
"""
Script d'entraînement du modèle XGBoost (classification binaire, AUC)

Ce script fait :
  1) Lecture des données PostgreSQL (SQLAlchemy)
  2) Préparation des données (encodages, features dérivées)
  3) Optimisation des hyperparamètres avec OPTUNA (TPE)
  4) Entraînement final avec early stopping (via xgboost.train)
  5) Évaluation sur le test + Sauvegarde des artefacts

Variables d'environnement utiles :
  - DATABASE_URL     : ex. postgresql+psycopg2://user:pwd@host:5432/db
  - OPTUNA_TRIALS    : nb d'essais Optuna (ex. 60 ; défaut=60)
  - OPTUNA_TIMEOUT   : temps max en secondes (ex. 1800 ; défaut=None)
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ✅ On utilise l'API bas niveau xgboost.train (compatible avec les versions plus anciennes)
import xgboost as xgb

# ✅ Optuna pour le tuning (TPE). On évite les callbacks d'intégration pour compat max.
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def safe_divide(num, den, fill_value: float = 0.0) -> pd.Series:
    """Division sûre avec gestion des NaN/infini et division par zéro."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
    out = pd.Series(out, index=num.index if isinstance(num, pd.Series) else None)
    return out.fillna(fill_value)


# -----------------------------------------------------------------------------
# I/O : PostgreSQL
# -----------------------------------------------------------------------------
def load_data_from_db() -> pd.DataFrame:
    """
    Charge les tables sirh, eval (ou performance), sondage et joint sur id_employee.
    """
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:password@localhost:5432/mydatabase",
    )
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

    with engine.begin() as conn:
        sirh = pd.read_sql(text('SELECT * FROM "sirh";'), conn)
        # Si ta table est "performance", remplace "eval" par "performance"
        eval_df = pd.read_sql(text('SELECT * FROM "eval";'), conn)
        sondage = pd.read_sql(text('SELECT * FROM "sondage";'), conn)

    # Harmonisation des clés
    if "id_employee" in sirh.columns:
        sirh["id_employee"] = pd.to_numeric(sirh["id_employee"], errors="coerce").astype("Int64")

    # "E_23" -> 23
    if "eval_number" in eval_df.columns:
        eval_df["id_employee"] = pd.to_numeric(eval_df["eval_number"].astype(str).str[2:], errors="coerce").astype("Int64")

    if "code_sondage" in sondage.columns:
        sondage["id_employee"] = pd.to_numeric(sondage["code_sondage"], errors="coerce").astype("Int64")

    # Jointures gauche successives
    df = sirh.merge(
        eval_df.drop(columns=["eval_number"], errors="ignore"),
        on="id_employee",
        how="left",
    ).merge(
        sondage.drop(columns=["code_sondage"], errors="ignore"),
        on="id_employee",
        how="left",
    )
    return df


# -----------------------------------------------------------------------------
# Préparation des données
# -----------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], Optional[OrdinalEncoder]]:
    """
    - Normalisations/corrections de colonnes
    - Encodages catégoriels (OneHot + Ordinal)
    - Features dérivées en bloc (évite fragmentation)
    - Nettoyage final
    """
    df_ = df.copy()

    # a) Normalisations / corrections
    col_aug_misspelled = "augementation_salaire_precedente"
    col_aug = "augmentation_salaire_precedente"
    src_col = col_aug_misspelled if col_aug_misspelled in df_.columns else (col_aug if col_aug in df_.columns else None)
    if src_col:
        df_["augmentation_taux"] = (
            df_[src_col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace(r"\s+", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
            .div(100)
        )
        df_.drop(columns=[src_col], inplace=True)
        df_.rename(columns={"augmentation_taux": col_aug}, inplace=True)

    if "nombre_heures_travailless" in df_.columns:
        df_.rename(columns={"nombre_heures_travailless": "nombre_heures_travaillees"}, inplace=True)

    # b) Cible
    if "a_quitte_l_entreprise" in df_.columns:
        df_["Attrition"] = (
            df_["a_quitte_l_entreprise"].astype(str).str.strip().str.lower()
            .map({"oui": 1, "non": 0}).fillna(0).astype(int)
        )

    df_.drop(
        columns=[
            "a_quitte_l_entreprise",
            "nombre_heures_travaillees",
            "id_employee",
            "ayant_enfants",
            "nombre_employee_sous_responsabilite",
        ],
        errors="ignore",
        inplace=True,
    )

    # c) Recodage oui/non
    if "heure_supplementaires" in df_.columns and df_["heure_supplementaires"].dtype == "object":
        df_["heure_supplementaires"] = (
            df_["heure_supplementaires"].astype(str).str.strip().str.lower()
            .map({"oui": 1, "non": 0}).astype("Int64")
        )

    # d) Encodages catégoriels
    colonnes_nominales = [c for c in ["departement", "poste", "domaine_etude", "statut_marital"] if c in df_.columns]
    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

    if colonnes_nominales:
        df_ohe = pd.DataFrame(
            ohe.fit_transform(df_[colonnes_nominales]),
            columns=ohe.get_feature_names_out(colonnes_nominales),
            index=df_.index,
        )
        df_.drop(columns=colonnes_nominales, inplace=True)
        df_ = pd.concat([df_, df_ohe], axis=1)
    else:
        ohe = None

    ordinal_encoder = None
    if "frequence_deplacement" in df_.columns:
        ordinal_encoder = OrdinalEncoder(categories=[["Aucun", "Occasionnel", "Frequent"]])
        df_[["frequence_deplacement"]] = ordinal_encoder.fit_transform(df_[["frequence_deplacement"]])

    # e) Dummies de secours
    df_model = pd.get_dummies(df_, drop_first=True)

    # f) Bool -> int
    bool_cols = df_model.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df_model[bool_cols] = df_model[bool_cols].astype(int)

    # g) Features dérivées en un seul bloc
    new_cols: Dict[str, pd.Series] = {}

    def make_ratio(dfm: pd.DataFrame, num: str, den: str) -> pd.Series:
        if num in dfm.columns and den in dfm.columns:
            return safe_divide(dfm[num], dfm[den], fill_value=0.0)
        return pd.Series(0.0, index=dfm.index)

    new_cols["ratio_anciennete"] = make_ratio(df_model, "annees_dans_l_entreprise", "annee_experience_totale")
    new_cols["ratio_poste"] = make_ratio(df_model, "annees_dans_le_poste_actuel", "annees_dans_l_entreprise")
    new_cols["ratio_formations"] = make_ratio(df_model, "nb_formations_suivies", "annees_dans_l_entreprise")

    if {"note_evaluation_actuelle", "note_evaluation_precedente"}.issubset(df_model.columns):
        new_cols["ecart_evaluation"] = (
            df_model["note_evaluation_actuelle"] - df_model["note_evaluation_precedente"]
        ).fillna(0.0)
        df_model = df_model.drop(columns="note_evaluation_precedente", errors="ignore")
    else:
        new_cols["ecart_evaluation"] = pd.Series(0.0, index=df_model.index)

    salaire_col = next((c for c in ["revenu_mensuel", "salaire_mensuel", "MonthlyIncome"] if c in df_model.columns), None)
    niveau_col = next((c for c in ["niveau_hierarchique_poste", "JobLevel"] if c in df_model.columns), None)
    if salaire_col and niveau_col:
        new_cols["ratio_salaire_niveau"] = make_ratio(df_model, salaire_col, niveau_col)
    else:
        new_cols["ratio_salaire_niveau"] = pd.Series(0.0, index=df_model.index)

    if "annees_depuis_la_derniere_promotion" in df_model.columns:
        new_cols["indice_recente_promo"] = 1.0 / (df_model["annees_depuis_la_derniere_promotion"].fillna(0) + 1.0)
        df_model = df_model.drop(columns="annees_depuis_la_derniere_promotion", errors="ignore")
    else:
        new_cols["indice_recente_promo"] = pd.Series(0.0, index=df_model.index)

    df_model = pd.concat([df_model, pd.DataFrame(new_cols)], axis=1)

    # h) Drop optionnels (corrélations fortes identifiées dans ton analyse)
    colonnes_a_supprimer = [
        "annees_dans_le_poste_actuel",
        "annes_sous_responsable_actuel",
        "poste_Ressources Humaines",
        "annee_experience_totale",
        "niveau_hierarchique_poste",
    ]
    df_model.drop(columns=[c for c in colonnes_a_supprimer if c in df_model.columns], inplace=True, errors="ignore")

    # i) Nettoyage final
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model.fillna(0.0, inplace=True)

    return df_model, ohe, ordinal_encoder


# -----------------------------------------------------------------------------
# Optuna + xgboost.train (compat max)
# -----------------------------------------------------------------------------
def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Retourne (#negatifs / #positifs) si possible, sinon 1.0."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return max(1.0, neg / pos) if pos > 0 else 1.0


def _optuna_objective_factory(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: Dict[str, Any],
):
    """
    Construit l'objective(trial) pour Optuna.
    Utilise xgboost.train avec DMatrix + early_stopping_rounds (pas de callbacks).
    Retourne l'AUC validation (meilleur score trouvé par XGBoost).
    """

    # Conversion en DMatrix (format natif XGBoost — rapide et stable)
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dvalid = xgb.DMatrix(X_val.values, label=y_val.values)

    def objective(trial: optuna.trial.Trial) -> float:
        # Espace de recherche des hyperparamètres
        params = {
            **base_params,  # objective, eval_metric, tree_method, etc.
            "eta": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),  # alias de learning_rate
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),  # L2
            "alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),     # L1
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        # Nombre d'arbres (boosting rounds)
        num_boost_round = trial.suggest_int("n_estimators", 300, 1200, step=100)

        # Entraînement avec early stopping (compat large)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],     # le log s'appellera "valid-auc"
            early_stopping_rounds=50,      # patience
            verbose_eval=False,
        )

        # Récupérer le meilleur score AUC sur la validation
        auc_val = float(getattr(booster, "best_score", np.nan))
        if np.isnan(auc_val):
            # Fallback : calcule l'AUC avec le meilleur nombre d'itérations
            best_iter = getattr(booster, "best_iteration", None)
            if best_iter is not None:
                proba_val = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
            else:
                proba_val = booster.predict(dvalid)
            auc_val = roc_auc_score(y_val.values, proba_val)

        return auc_val

    return objective


def tune_with_optuna(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    n_trials: int = 60,
    timeout: Optional[int] = None,
    seed: int = 1042,
) -> Dict[str, Any]:
    """
    Lancement de l'optimisation Optuna (TPE).
    Retourne les meilleurs hyperparamètres à réutiliser pour le fit final.
    """
    # Split interne pour guider early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=seed, stratify=y_train_full
    )

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",      # "gpu_hist" si GPU dispo
        "seed": seed,               # 'random_state' s'appelle 'seed' dans l'API bas niveau
        "verbosity": 0,
        "scale_pos_weight": _compute_scale_pos_weight(y_train_full),
        # "nthread": -1,  # selon la version ; tu peux décommenter si supporté
    }

    objective = _optuna_objective_factory(X_tr, y_tr, X_val, y_val, base_params)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_warmup_steps=10),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print("🔎 Meilleur AUC (val) :", study.best_value)
    print("🔧 Meilleurs hyperparamètres :", study.best_trial.params)

    # Recompose les params finaux pour xgb.train
    best = study.best_trial.params
    best_params = {
        **base_params,
        "eta": best["learning_rate"],
        "max_depth": best["max_depth"],
        "min_child_weight": best["min_child_weight"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "lambda": best["reg_lambda"],
        "alpha": best["reg_alpha"],
        "gamma": best["gamma"],
    }
    best_num_boost_round = best["n_estimators"]

    return {"best_params": best_params, "best_num_boost_round": best_num_boost_round, "study": study}


# -----------------------------------------------------------------------------
# Entraînement principal (fit final + évaluation + sauvegardes)
# -----------------------------------------------------------------------------
def train_model():
    """Pipeline complet : data -> prep -> tuning -> fit final -> éval -> sauvegardes."""
    # Logs plus lisibles
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # 1) Data
    print("📊 Chargement des données depuis PostgreSQL...")
    df = load_data_from_db()
    print(f"✅ {len(df)} lignes chargées")

    # 2) Préparation
    print("🔧 Préparation des données...")
    df_model, ohe, ordinal_encoder = prepare_data(df)

    if "Attrition" not in df_model.columns:
        raise ValueError("La colonne cible 'Attrition' est absente après préparation des données.")

    X = df_model.drop(columns=["Attrition"])
    y = df_model["Attrition"].astype(int)
    print(f"✅ {X.shape[1]} features préparées")

    # 3) Train/Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1042, stratify=y
    )

    # 4) Tuning Optuna (budget contrôlable par variables d'env)
    n_trials = int(os.getenv("OPTUNA_TRIALS", "60"))
    timeout_env = os.getenv("OPTUNA_TIMEOUT", None)
    timeout = int(timeout_env) if timeout_env is not None else None

    print(f"🧪 Lancement d'Optuna (n_trials={n_trials}, timeout={timeout}) ...")
    tune_result = tune_with_optuna(
        X_train_full, y_train_full, n_trials=n_trials, timeout=timeout, seed=1042
    )
    best_params = tune_result["best_params"]
    best_num_boost_round = tune_result["best_num_boost_round"]

    # 5) Fit final (refit) avec early stopping sur un nouveau split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=2042, stratify=y_train_full
    )

    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dvalid = xgb.DMatrix(X_val.values, label=y_val.values)

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # 6) Évaluation Test
    dtest = xgb.DMatrix(X_test.values, label=y_test.values)
    # ✅ Correction : utiliser iteration_range si best_iteration est dispo (plus de ntree_limit)
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        y_proba = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    else:
        y_proba = booster.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n📊 Métriques sur le jeu de test :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"AUC-PR: {average_precision_score(y_test, y_proba):.4f}")

    # 7) Sauvegardes des artefacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # a) Sauvegarder le modèle XGBoost au format JSON (recommandé, très compatible)
    booster_path = models_dir / "xgb_booster.json"
    booster.save_model(str(booster_path))

    # b) Sauvegarder aussi les préprocesseurs et la liste des features (ordre à respecter en prod)
    joblib.dump(ohe, models_dir / "onehot_encoder.joblib")
    joblib.dump(ordinal_encoder, models_dir / "ordinal_encoder.joblib")
    feature_names = list(X.columns)
    joblib.dump(feature_names, models_dir / "feature_names.joblib")

    # c) Sauvegarder les meilleurs hyperparams (audit/reprod.)
    joblib.dump(
        {"best_params": best_params, "best_num_boost_round": best_num_boost_round},
        models_dir / "xgb_best_params.joblib",
    )

    print(f"\n✅ Modèle XGBoost sauvegardé : {booster_path}")
    print(f"✅ {len(feature_names)} features sauvegardées")
    print(f"✅ Hyperparamètres Optuna sauvegardés (xgb_best_params.joblib)")

    return booster, ohe, ordinal_encoder, feature_names


if __name__ == "__main__":
    train_model()
