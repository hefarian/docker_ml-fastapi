
"""
Module de chargement et utilisation du modèle XGBoost pour l'attrition.
- Charge le Booster XGBoost + encoders + ordre des features
- Applique la même préparation minimale qu'au training pour 1 employé
- Prédit (label, proba) en garantissant l'ordre des colonnes
"""

from functools import lru_cache
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from schemas import EmployeeData
from settings import settings


class AttritionModel:
    """Modèle de prédiction d'attrition basé sur XGBoost (Booster)."""

    def __init__(
        self,
        booster_path: str,
        ohe_path: str,
        ordinal_path: str,
        feature_names_path: str,
    ):
        """
        Charge le Booster XGBoost et les artefacts de prétraitement/ordre des features.
        """
        # Modèle XGBoost (Booster) au format JSON
        self.booster = xgb.Booster()
        self.booster.load_model(booster_path)

        # Encoders & noms de colonnes
        self.ohe = joblib.load(ohe_path)  # peut être None si pas d'OHE au train
        self.ordinal_encoder = joblib.load(ordinal_path)  # peut être None si pas utilisé
        self.feature_names = joblib.load(feature_names_path)  # ordre attendu en prod (list[str])

    # --------------------
    # Utilitaires internes
    # --------------------
    @staticmethod
    def _safe_divide(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """Division sûre avec gestion NaN/Inf et division par zéro."""
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
        out = pd.Series(out, index=num.index if isinstance(num, pd.Series) else None)
        return out.fillna(fill_value)

    @staticmethod
    def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
        """Vérifie que toutes les colonnes demandées existent dans df."""
        return all(c in df.columns for c in cols)

    # --------------------
    # Transformation d'entrée
    # --------------------
    def _transform_employee_data(self, employee_data: EmployeeData) -> np.ndarray:
        """
        Transforme les données d'un employé (Pydantic) en vecteur de features
        aligné sur self.feature_names.
        """
        # 1) Pydantic -> dict -> DataFrame (1 ligne)
        data_dict = employee_data.model_dump()
        df = pd.DataFrame([data_dict])

        # 1.b) Optionnel: normalisation du champ 'augmentation_salaire_precedente'
        # si celui-ci arrive en chaîne avec un '%' (pour rester cohérent avec le train)
        if "augmentation_salaire_precedente" in df.columns:
            ser = df["augmentation_salaire_precedente"].astype(str)
            if ser.str.contains("%").any():
                df["augmentation_salaire_precedente"] = (
                    ser.str.replace("%", "", regex=False)
                       .str.replace(",", ".", regex=False)
                       .str.replace(r"\s+", "", regex=True)
                       .pipe(pd.to_numeric, errors="coerce")
                       .div(100.0)
                ).fillna(0.0)

        # 2) OneHot sur nominales (même liste que le train)
        colonnes_nominales = [c for c in ["departement", "poste", "domaine_etude", "statut_marital"] if c in df.columns]
        if self.ohe is not None and colonnes_nominales:
            df_ohe = pd.DataFrame(
                self.ohe.transform(df[colonnes_nominales]),
                columns=self.ohe.get_feature_names_out(colonnes_nominales),
                index=df.index,
            )
            df = df.drop(columns=colonnes_nominales)
            df = pd.concat([df, df_ohe], axis=1)

        # 3) Ordinal pour 'frequence_deplacement' si présent
        if self.ordinal_encoder is not None and "frequence_deplacement" in df.columns:
            # S'assure qu'on manipule bien des str (pour éviter les erreurs)
            df[["frequence_deplacement"]] = self.ordinal_encoder.transform(
                df[["frequence_deplacement"]]
            )

        # 4) Recode oui/non -> 1/0 pour 'heure_supplementaires' si string
        if "heure_supplementaires" in df.columns:
            if df["heure_supplementaires"].dtype == object:
                df["heure_supplementaires"] = (
                    df["heure_supplementaires"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"oui": 1, "non": 0})
                    .fillna(0)
                )

        # 5) get_dummies de secours si d'autres catégorielles restent
        df = pd.get_dummies(df, drop_first=True)

        # 6) Bool -> int
        bool_cols = df.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)

        # 7) Features dérivées (robustes aux colonnes manquantes)
        def add_ratio(safe_df: pd.DataFrame, num_col: str, den_col: str, out_col: str):
            if num_col in safe_df.columns and den_col in safe_df.columns:
                safe_df[out_col] = self._safe_divide(safe_df[num_col], safe_df[den_col], fill_value=0.0)
            else:
                safe_df[out_col] = 0.0

        add_ratio(df, "annees_dans_l_entreprise", "annee_experience_totale", "ratio_anciennete")
        add_ratio(df, "annees_dans_le_poste_actuel", "annees_dans_l_entreprise", "ratio_poste")

        if self._has_cols(df, ["note_evaluation_actuelle", "note_evaluation_precedente"]):
            df["ecart_evaluation"] = (
                (pd.to_numeric(df["note_evaluation_actuelle"], errors="coerce")
                 - pd.to_numeric(df["note_evaluation_precedente"], errors="coerce"))
                .fillna(0.0)
            )
        else:
            df["ecart_evaluation"] = 0.0
        df = df.drop(columns="note_evaluation_precedente", errors="ignore")

        add_ratio(df, "revenu_mensuel", "niveau_hierarchique_poste", "ratio_salaire_niveau")
        add_ratio(df, "nb_formations_suivies", "annees_dans_l_entreprise", "ratio_formations")

        if "annees_depuis_la_derniere_promotion" in df.columns:
            df["annees_depuis_la_derniere_promotion"] = pd.to_numeric(
                df["annees_depuis_la_derniere_promotion"], errors="coerce"
            )
            df["indice_recente_promo"] = 1.0 / (df["annees_depuis_la_derniere_promotion"].fillna(0) + 1.0)
        else:
            df["indice_recente_promo"] = 0.0
        df = df.drop(columns="annees_depuis_la_derniere_promotion", errors="ignore")

        # 8) Drop de colonnes corrélées (si présentes)
        colonnes_a_supprimer = [
            "annees_dans_le_poste_actuel",
            "annes_sous_responsable_actuel",
            "poste_Ressources Humaines",
            "annee_experience_totale",
            "niveau_hierarchique_poste",
        ]
        df = df.drop(columns=[c for c in colonnes_a_supprimer if c in df.columns], errors="ignore")

        # 9) Nettoyage final
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 10) S'assurer que toutes les features attendues existent
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0

        # 11) Réordonner les colonnes selon l'ordre attendu
        df = df[self.feature_names]

        # Renvoie la matrice numpy
        return df.values

    # --------------------
    # Prédiction
    # --------------------
    def predict(self, employee_data: EmployeeData) -> Tuple[int, float]:
        """
        Prédit l'attrition pour un employé.

        Returns:
            (label, proba) : (0/1, probabilité d’attrition)
        """
        # Transforme en features alignées
        X = self._transform_employee_data(employee_data)

        # XGBoost attend un DMatrix
        dmat = xgb.DMatrix(X)

        # Si le booster a une best_iteration (early stopping), on la respecte pour la prédiction
        best_iter = getattr(self.booster, "best_iteration", None)
        if best_iter is not None:
            proba = float(self.booster.predict(dmat, iteration_range=(0, best_iter + 1))[0])
        else:
            proba = float(self.booster.predict(dmat)[0])

        label = int(proba >= 0.5)
        return label, proba


# =============================================================================
# Fabrique / cache
# =============================================================================
@lru_cache(maxsize=1)
def get_model() -> AttritionModel:
    """
    Charge le modèle et les artefacts (avec cache).
    On part du dossier parent de settings.MODEL_PATH pour rester compatible
    avec ta config existante.
    """
    models_dir = Path(settings.MODEL_PATH).parent

    booster_path = models_dir / "xgb_booster.json"
    ohe_path = models_dir / "onehot_encoder.joblib"
    ordinal_path = models_dir / "ordinal_encoder.joblib"
    feature_names_path = models_dir / "feature_names.joblib"

    # Vérifie la présence des fichiers nécessaires
    for path in [booster_path, ohe_path, ordinal_path, feature_names_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier artefact manquant: {path}. "
                f"Veuillez exécuter le script d'entraînement pour générer le modèle et les artefacts."
            )

    return AttritionModel(
        booster_path=str(booster_path),
        ohe_path=str(ohe_path),
        ordinal_path=str(ordinal_path),
        feature_names_path=str(feature_names_path),
    )
