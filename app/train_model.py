# ============================================================================
# FICHIER : app/train_model.py
# ============================================================================
#
# QU'EST-CE QUE CE SCRIPT ?
# =========================
# Ce script entraîne un modèle de machine learning (XGBoost) pour prédire l'attrition
# des employés. L'attrition = départ d'un employé de l'entreprise.
#
# POURQUOI XGBOOST ?
# ===================
# XGBoost (eXtreme Gradient Boosting) est un algorithme très performant pour :
# - La classification binaire (l'employé reste ou part)
# - Les données tabulaires (tableaux avec colonnes)
# - La gestion des valeurs manquantes
#
# PIPELINE COMPLET :
# ==================
# 1. Charger les données depuis PostgreSQL (3 tables : sirh, eval, sondage)
# 2. Préparer les données (nettoyage, encodage, features dérivées)
# 3. Optimiser les hyperparamètres avec Optuna (recherche automatique)
# 4. Entraîner le modèle final avec les meilleurs hyperparamètres
# 5. Évaluer le modèle sur un jeu de test
# 6. Sauvegarder le modèle et tous les artefacts nécessaires
#
# VARIABLES D'ENVIRONNEMENT :
# ============================
# - DATABASE_URL : URL de connexion PostgreSQL
#   Exemple : postgresql+psycopg2://user:pwd@host:5432/db
# - OPTUNA_TRIALS : Nombre d'essais Optuna (défaut : 60)
#   Plus d'essais = meilleur modèle mais plus long
# - OPTUNA_TIMEOUT : Temps maximum en secondes (défaut : None = pas de limite)
#   Exemple : 1800 = 30 minutes maximum
#
# ============================================================================

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

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# joblib : pour sauvegarder/charger les modèles et préprocesseurs
import joblib

# numpy : calculs numériques (tableaux multidimensionnels)
import numpy as np

# pandas : manipulation de données tabulaires (DataFrames)
import pandas as pd

# sqlalchemy : ORM pour interagir avec PostgreSQL
from sqlalchemy import create_engine, text

# sklearn.metrics : métriques pour évaluer le modèle
from sklearn.metrics import (
    classification_report,      # Rapport détaillé (précision, rappel, F1)
    roc_auc_score,              # AUC-ROC (aire sous la courbe ROC)
    average_precision_score,    # AUC-PR (aire sous la courbe précision-rappel)
)

# sklearn.model_selection : outils pour diviser les données
from sklearn.model_selection import train_test_split

# sklearn.preprocessing : outils de préparation des données
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# xgboost : bibliothèque de machine learning (gradient boosting)
# On utilise l'API bas niveau xgboost.train (compatible avec les versions plus anciennes)
import xgboost as xgb

# optuna : bibliothèque d'optimisation d'hyperparamètres
# TPE = Tree-structured Parzen Estimator (algorithme de recherche)
# On évite les callbacks d'intégration pour compatibilité maximale
import optuna
from optuna.samplers import TPESampler      # Algorithme de recherche
from optuna.pruners import MedianPruner    # Arrêt prématuré des essais peu prometteurs


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def safe_divide(num, den, fill_value: float = 0.0) -> pd.Series:
    """
    Division sûre avec gestion des NaN/infini et division par zéro.
    
    QU'EST-CE QU'UNE DIVISION SÛRE ?
    =================================
    En mathématiques, diviser par zéro est impossible (erreur).
    En programmation, cela crée des valeurs infinies (inf) ou NaN (Not a Number).
    Cette fonction évite ces problèmes.
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Convertit les valeurs en nombres (gère les erreurs)
    2. Vérifie que le dénominateur est > 0 et fini
    3. Divise seulement si c'est sûr
    4. Remplace les NaN/inf par une valeur par défaut (fill_value)
    
    PARAMÈTRES :
    ===========
    - num : numérateur (peut être un nombre, une série pandas, etc.)
    - den : dénominateur (peut être un nombre, une série pandas, etc.)
    - fill_value : valeur à utiliser si division impossible (défaut : 0.0)
    
    RETOUR :
    ========
    pd.Series : série pandas avec les résultats de la division
    
    EXEMPLE :
    =========
    >>> safe_divide([10, 20, 30], [2, 0, 5], fill_value=0.0)
    [5.0, 0.0, 6.0]  # 20/0 devient 0.0 au lieu d'inf
    """
    # Convertir en nombres (errors="coerce" = remplace les erreurs par NaN)
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    
    # Division conditionnelle :
    # - Si den > 0 ET den est fini (pas inf), alors num / den
    # - Sinon, NaN
    # np.where : équivalent d'un if/else pour chaque élément
    out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
    
    # Convertir en série pandas (conserve l'index si num est une série)
    out = pd.Series(out, index=num.index if isinstance(num, pd.Series) else None)
    
    # Remplacer les NaN par fill_value (0.0 par défaut)
    return out.fillna(fill_value)


# ============================================================================
# CHARGEMENT DES DONNÉES DEPUIS POSTGRESQL
# ============================================================================

def load_data_from_db() -> pd.DataFrame:
    """
    Charge les tables sirh, eval (ou performance), sondage et joint sur id_employee.
    
    QU'EST-CE QU'UNE JOINTURE ?
    ===========================
    Une jointure combine plusieurs tables en une seule.
    Ici, on a 3 tables avec des informations différentes sur les employés :
    - sirh : informations RH (âge, salaire, département, etc.)
    - eval : évaluations de performance
    - sondage : résultats de sondages
    
    Toutes ces tables sont liées par "id_employee" (identifiant unique).
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Se connecter à PostgreSQL avec SQLAlchemy
    2. Charger chaque table séparément avec pd.read_sql()
    3. Harmoniser les clés (s'assurer que id_employee existe partout)
    4. Faire des jointures gauches (LEFT JOIN) pour combiner les tables
    5. Retourner un DataFrame unique avec toutes les colonnes
    
    RETOUR :
    ========
    pd.DataFrame : DataFrame pandas avec toutes les données combinées
    
    EXEMPLE DE STRUCTURE :
    ======================
    id_employee | age | revenu_mensuel | note_evaluation | a_quitte_l_entreprise
    1           | 35  | 5000          | 4.5             | Non
    2           | 28  | 3500          | 3.8             | Oui
    ...
    """
    # Récupérer l'URL de la base de données depuis les variables d'environnement
    # Si DATABASE_URL n'existe pas, utiliser une valeur par défaut (localhost)
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:password@localhost:5432/mydatabase",
    )
    
    # Créer un moteur SQLAlchemy pour se connecter à PostgreSQL
    # pool_pre_ping=True : vérifie que les connexions sont valides avant utilisation
    # future=True : utilise l'API moderne de SQLAlchemy
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

    # Ouvrir une connexion et charger les 3 tables
    # engine.begin() : crée une transaction (toutes les opérations réussissent ou échouent ensemble)
    with engine.begin() as conn:
        # Charger la table SIRH (Système d'Information des Ressources Humaines)
        # text('SELECT * FROM "sirh";') : requête SQL brute
        # Les guillemets doubles sont nécessaires car "sirh" est en minuscules
        sirh = pd.read_sql(text('SELECT * FROM "sirh";'), conn)
        
        # Charger la table eval (évaluations)
        # Note : Si votre table s'appelle "performance", remplacez "eval" par "performance"
        eval_df = pd.read_sql(text('SELECT * FROM "eval";'), conn)
        
        # Charger la table sondage (résultats de sondages)
        sondage = pd.read_sql(text('SELECT * FROM "sondage";'), conn)

    # ========================================================================
    # HARMONISATION DES CLÉS
    # ========================================================================
    # Problème : les 3 tables peuvent avoir des formats différents pour id_employee
    # Solution : convertir toutes les clés au même format (entier)
    
    # Table SIRH : id_employee devrait déjà être un entier
    if "id_employee" in sirh.columns:
        # pd.to_numeric : convertit en nombre (errors="coerce" = NaN si erreur)
        # .astype("Int64") : convertit en entier (Int64 permet les NaN)
        sirh["id_employee"] = pd.to_numeric(sirh["id_employee"], errors="coerce").astype("Int64")

    # Table eval : id_employee peut être dans "eval_number" avec format "E_23"
    # On extrait le nombre après "E_" (ex: "E_23" -> 23)
    if "eval_number" in eval_df.columns:
        # .astype(str) : convertit en chaîne
        # .str[2:] : prend tout après les 2 premiers caractères ("E_" -> reste "23")
        # pd.to_numeric : convertit "23" en nombre 23
        eval_df["id_employee"] = pd.to_numeric(eval_df["eval_number"].astype(str).str[2:], errors="coerce").astype("Int64")

    # Table sondage : id_employee peut être dans "code_sondage"
    if "code_sondage" in sondage.columns:
        sondage["id_employee"] = pd.to_numeric(sondage["code_sondage"], errors="coerce").astype("Int64")

    # ========================================================================
    # JOINTURES GAUCHES (LEFT JOIN)
    # ========================================================================
    # LEFT JOIN : garde toutes les lignes de la table de gauche (sirh)
    # et ajoute les colonnes des autres tables si elles existent
    
    # Première jointure : sirh + eval
    df = sirh.merge(
        eval_df.drop(columns=["eval_number"], errors="ignore"),  # Supprimer eval_number (plus besoin)
        on="id_employee",      # Clé de jointure
        how="left",            # LEFT JOIN (garde tous les employés de sirh)
    ).merge(
        # Deuxième jointure : résultat précédent + sondage
        sondage.drop(columns=["code_sondage"], errors="ignore"),  # Supprimer code_sondage
        on="id_employee",
        how="left",
    )
    
    return df


# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], Optional[OrdinalEncoder]]:
    """
    Prépare les données pour l'entraînement du modèle.
    
    QU'EST-CE QUE LA PRÉPARATION DES DONNÉES ?
    ===========================================
    Les modèles de machine learning ne peuvent pas travailler directement avec
    des données brutes. Il faut :
    1. Nettoyer les données (corriger les erreurs, gérer les valeurs manquantes)
    2. Encoder les variables catégorielles (texte -> nombres)
    3. Créer des features dérivées (nouvelles colonnes calculées)
    4. Supprimer les colonnes inutiles
    
    PIPELINE DE PRÉPARATION :
    =========================
    a) Normalisations/corrections de colonnes
    b) Création de la variable cible (Attrition)
    c) Recodage oui/non -> 1/0
    d) Encodages catégoriels (OneHot + Ordinal)
    e) Dummies de secours (pour les catégories restantes)
    f) Conversion bool -> int
    g) Features dérivées (ratios, écarts, etc.)
    h) Suppression de colonnes corrélées
    i) Nettoyage final (NaN, infini)
    
    RETOUR :
    ========
    Tuple contenant :
    - df_model : DataFrame préparé (prêt pour l'entraînement)
    - ohe : Encodeur OneHot (pour réutiliser en production)
    - ordinal_encoder : Encodeur ordinal (pour réutiliser en production)
    """
    # Créer une copie pour ne pas modifier le DataFrame original
    df_ = df.copy()

    # ========================================================================
    # a) NORMALISATIONS / CORRECTIONS DE COLONNES
    # ========================================================================
    
    # Problème : le nom de la colonne peut être mal orthographié
    # Solution : chercher les deux variantes possibles
    col_aug_misspelled = "augementation_salaire_precedente"  # Faute d'orthographe
    col_aug = "augmentation_salaire_precedente"              # Correct
    
    # Chercher quelle colonne existe
    src_col = col_aug_misspelled if col_aug_misspelled in df_.columns else (col_aug if col_aug in df_.columns else None)
    
    if src_col:
        # Normaliser le format de l'augmentation salariale
        # Exemple : "11%" ou "11,5%" -> 0.11 ou 0.115
        df_["augmentation_taux"] = (
            df_[src_col].astype(str)                    # Convertir en chaîne
            .str.replace("%", "", regex=False)            # Supprimer le %
            .str.replace(",", ".", regex=False)           # Remplacer , par .
            .str.replace(r"\s+", "", regex=True)          # Supprimer les espaces
            .pipe(pd.to_numeric, errors="coerce")         # Convertir en nombre
            .div(100)                                      # Diviser par 100 (11% -> 0.11)
        )
        # Supprimer l'ancienne colonne et renommer la nouvelle
        df_.drop(columns=[src_col], inplace=True)
        df_.rename(columns={"augmentation_taux": col_aug}, inplace=True)

    # Corriger le nom de la colonne (faute d'orthographe dans la base)
    if "nombre_heures_travailless" in df_.columns:
        df_.rename(columns={"nombre_heures_travailless": "nombre_heures_travaillees"}, inplace=True)

    # ========================================================================
    # b) CRÉATION DE LA VARIABLE CIBLE (ATTRITION)
    # ========================================================================
    # La variable cible = ce qu'on veut prédire
    # Ici : l'employé a-t-il quitté l'entreprise ? (Oui = 1, Non = 0)
    
    if "a_quitte_l_entreprise" in df_.columns:
        df_["Attrition"] = (
            df_["a_quitte_l_entreprise"].astype(str)      # Convertir en chaîne
            .str.strip()                                   # Supprimer les espaces
            .str.lower()                                   # Mettre en minuscules
            .map({"oui": 1, "non": 0})                    # Mapper oui->1, non->0
            .fillna(0)                                     # Remplacer NaN par 0
            .astype(int)                                   # Convertir en entier
        )

    # Supprimer les colonnes inutiles pour le modèle
    df_.drop(
        columns=[
            "a_quitte_l_entreprise",              # Déjà converti en Attrition
            "nombre_heures_travaillees",          # Non utilisé dans le modèle
            "id_employee",                        # Identifiant (pas une feature)
            "ayant_enfants",                      # Non utilisé
            "nombre_employee_sous_responsabilite", # Non utilisé
        ],
        errors="ignore",  # Ignorer si la colonne n'existe pas
        inplace=True,
    )

    # ========================================================================
    # c) RECODAGE OUI/NON -> 1/0
    # ========================================================================
    # Les modèles préfèrent les nombres (0/1) plutôt que les chaînes ("Oui"/"Non")
    
    if "heure_supplementaires" in df_.columns and df_["heure_supplementaires"].dtype == "object":
        df_["heure_supplementaires"] = (
            df_["heure_supplementaires"].astype(str)      # Convertir en chaîne
            .str.strip()                                   # Supprimer les espaces
            .str.lower()                                   # Mettre en minuscules
            .map({"oui": 1, "non": 0})                    # Mapper oui->1, non->0
            .astype("Int64")                               # Convertir en entier (avec NaN support)
        )

    # ========================================================================
    # d) ENCODAGES CATÉGORIELS
    # ========================================================================
    # Problème : les modèles ne comprennent pas le texte ("R&D", "Commercial", etc.)
    # Solution : convertir en nombres avec des encodages
    
    # Colonnes nominales (sans ordre) : département, poste, domaine_etude, statut_marital
    colonnes_nominales = [c for c in ["departement", "poste", "domaine_etude", "statut_marital"] if c in df_.columns]
    
    # OneHotEncoder : crée une colonne par catégorie
    # Exemple : département "R&D" -> colonne "departement_R&D" = 1, autres = 0
    # drop="first" : supprime la première colonne (évite la colinéarité)
    # sparse_output=False : retourne un tableau dense (pas une matrice creuse)
    # handle_unknown="ignore" : si une nouvelle catégorie apparaît, la traite comme zéros
    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # Compatibilité avec les anciennes versions de scikit-learn
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

    if colonnes_nominales:
        # Appliquer l'encodage OneHot
        # fit_transform : apprend les catégories ET transforme les données
        df_ohe = pd.DataFrame(
            ohe.fit_transform(df_[colonnes_nominales]),    # Encodage
            columns=ohe.get_feature_names_out(colonnes_nominales),  # Noms des colonnes
            index=df_.index,                               # Conserver l'index
        )
        # Supprimer les colonnes originales et ajouter les colonnes encodées
        df_.drop(columns=colonnes_nominales, inplace=True)
        df_ = pd.concat([df_, df_ohe], axis=1)
    else:
        ohe = None  # Pas d'encodage si pas de colonnes nominales

    # OrdinalEncoder : pour les colonnes avec un ordre naturel
    # Exemple : "Aucun" < "Occasionnel" < "Frequent"
    ordinal_encoder = None
    if "frequence_deplacement" in df_.columns:
        # categories : définit l'ordre des catégories
        ordinal_encoder = OrdinalEncoder(categories=[["Aucun", "Occasionnel", "Frequent"]])
        # fit_transform : apprend l'ordre ET transforme
        df_[["frequence_deplacement"]] = ordinal_encoder.fit_transform(df_[["frequence_deplacement"]])

    # ========================================================================
    # e) DUMMIES DE SECOURS
    # ========================================================================
    # Pour les colonnes catégorielles restantes (non encodées précédemment)
    # pd.get_dummies : crée des colonnes binaires (0/1) pour chaque catégorie
    # drop_first=True : supprime la première colonne (évite la colinéarité)
    df_model = pd.get_dummies(df_, drop_first=True)

    # ========================================================================
    # f) CONVERSION BOOL -> INT
    # ========================================================================
    # Les colonnes booléennes (True/False) doivent être converties en 1/0
    bool_cols = df_model.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df_model[bool_cols] = df_model[bool_cols].astype(int)

    # ========================================================================
    # g) FEATURES DÉRIVÉES
    # ========================================================================
    # Créer de nouvelles colonnes calculées à partir des colonnes existantes
    # Ces features peuvent améliorer les performances du modèle
    
    new_cols: Dict[str, pd.Series] = {}  # Dictionnaire pour stocker les nouvelles colonnes

    # Fonction helper pour créer des ratios
    def make_ratio(dfm: pd.DataFrame, num: str, den: str) -> pd.Series:
        """Crée un ratio (numérateur / dénominateur) de manière sûre."""
        if num in dfm.columns and den in dfm.columns:
            return safe_divide(dfm[num], dfm[den], fill_value=0.0)
        return pd.Series(0.0, index=dfm.index)

    # Ratio d'ancienneté : années dans l'entreprise / années d'expérience totale
    # Plus proche de 1 = l'employé a passé toute sa carrière dans l'entreprise
    new_cols["ratio_anciennete"] = make_ratio(df_model, "annees_dans_l_entreprise", "annee_experience_totale")
    
    # Ratio poste : années dans le poste actuel / années dans l'entreprise
    # Plus proche de 1 = l'employé est dans le même poste depuis longtemps
    new_cols["ratio_poste"] = make_ratio(df_model, "annees_dans_le_poste_actuel", "annees_dans_l_entreprise")
    
    # Ratio formations : nombre de formations / années dans l'entreprise
    # Plus élevé = l'employé se forme beaucoup
    new_cols["ratio_formations"] = make_ratio(df_model, "nb_formations_suivies", "annees_dans_l_entreprise")

    # Écart d'évaluation : note actuelle - note précédente
    # Positif = amélioration, négatif = détérioration
    if {"note_evaluation_actuelle", "note_evaluation_precedente"}.issubset(df_model.columns):
        new_cols["ecart_evaluation"] = (
            df_model["note_evaluation_actuelle"] - df_model["note_evaluation_precedente"]
        ).fillna(0.0)
        # Supprimer la note précédente (déjà utilisée dans l'écart)
        df_model = df_model.drop(columns="note_evaluation_precedente", errors="ignore")
    else:
        new_cols["ecart_evaluation"] = pd.Series(0.0, index=df_model.index)

    # Ratio salaire/niveau : revenu mensuel / niveau hiérarchique
    # Plus élevé = mieux payé par rapport à son niveau
    salaire_col = next((c for c in ["revenu_mensuel", "salaire_mensuel", "MonthlyIncome"] if c in df_model.columns), None)
    niveau_col = next((c for c in ["niveau_hierarchique_poste", "JobLevel"] if c in df_model.columns), None)
    if salaire_col and niveau_col:
        new_cols["ratio_salaire_niveau"] = make_ratio(df_model, salaire_col, niveau_col)
    else:
        new_cols["ratio_salaire_niveau"] = pd.Series(0.0, index=df_model.index)

    # Indice de promotion récente : 1 / (années depuis promotion + 1)
    # Plus proche de 1 = promotion très récente
    if "annees_depuis_la_derniere_promotion" in df_model.columns:
        new_cols["indice_recente_promo"] = 1.0 / (df_model["annees_depuis_la_derniere_promotion"].fillna(0) + 1.0)
        df_model = df_model.drop(columns="annees_depuis_la_derniere_promotion", errors="ignore")
    else:
        new_cols["indice_recente_promo"] = pd.Series(0.0, index=df_model.index)

    # Ajouter toutes les nouvelles colonnes au DataFrame
    df_model = pd.concat([df_model, pd.DataFrame(new_cols)], axis=1)

    # ========================================================================
    # h) SUPPRESSION DE COLONNES CORRÉLÉES
    # ========================================================================
    # Certaines colonnes sont très corrélées (redondantes)
    # Les supprimer évite le surapprentissage (overfitting)
    colonnes_a_supprimer = [
        "annees_dans_le_poste_actuel",      # Déjà utilisé dans ratio_poste
        "annes_sous_responsable_actuel",     # Peu informatif
        "poste_Ressources Humaines",         # Catégorie rare
        "annee_experience_totale",          # Déjà utilisé dans ratio_anciennete
        "niveau_hierarchique_poste",         # Déjà utilisé dans ratio_salaire_niveau
    ]
    df_model.drop(columns=[c for c in colonnes_a_supprimer if c in df_model.columns], inplace=True, errors="ignore")

    # ========================================================================
    # i) NETTOYAGE FINAL
    # ========================================================================
    # Remplacer les valeurs infinies et NaN par 0.0
    # Les modèles ne peuvent pas gérer infini ou NaN
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model.fillna(0.0, inplace=True)

    return df_model, ohe, ordinal_encoder


# ============================================================================
# OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA
# ============================================================================

def _compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Calcule le poids pour équilibrer les classes (déséquilibre positif/négatif).
    
    QU'EST-CE QUE LE DÉSÉQUILIBRE DE CLASSES ?
    ==========================================
    Si on a 90% d'employés qui restent et 10% qui partent, le modèle peut
    toujours prédire "reste" et avoir 90% de précision (mais inutile !).
    
    scale_pos_weight permet de donner plus d'importance aux exemples positifs (départ).
    
    RETOUR :
    ========
    float : ratio (#négatifs / #positifs)
    Exemple : si 90% restent et 10% partent, retourne 9.0
    """
    pos = int((y == 1).sum())  # Nombre d'employés qui partent
    neg = int((y == 0).sum())  # Nombre d'employés qui restent
    # Retourner le ratio, minimum 1.0 (pas de déséquilibre si pos = 0)
    return max(1.0, neg / pos) if pos > 0 else 1.0


def _optuna_objective_factory(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: Dict[str, Any],
):
    """
    Construit la fonction objective pour Optuna.
    
    QU'EST-CE QU'UNE FONCTION OBJECTIVE ?
    =====================================
    C'est la fonction qu'Optuna essaie d'optimiser (maximiser).
    Ici, on maximise l'AUC (Area Under Curve) sur le jeu de validation.
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Optuna propose des hyperparamètres (learning_rate, max_depth, etc.)
    2. On entraîne un modèle avec ces hyperparamètres
    3. On calcule l'AUC sur le jeu de validation
    4. On retourne l'AUC (Optuna cherche à la maximiser)
    
    PARAMÈTRES :
    ===========
    - X_tr, y_tr : données d'entraînement (features et cible)
    - X_val, y_val : données de validation (pour évaluer les hyperparamètres)
    - base_params : paramètres de base (objectif, métrique, etc.)
    
    RETOUR :
    ========
    function : fonction objective(trial) que Optuna peut appeler
    """
    
    # Conversion en DMatrix (format natif XGBoost)
    # DMatrix est optimisé pour XGBoost (plus rapide que DataFrame)
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dvalid = xgb.DMatrix(X_val.values, label=y_val.values)

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Fonction objective appelée par Optuna pour chaque essai.
        
        PARAMÈTRES :
        ============
        - trial : objet Optuna qui propose des hyperparamètres à tester
        
        RETOUR :
        ========
        float : AUC sur le jeu de validation (à maximiser)
        """
        # ====================================================================
        # ESPACE DE RECHERCHE DES HYPERPARAMÈTRES
        # ====================================================================
        # Optuna propose des valeurs pour chaque hyperparamètre
        # trial.suggest_* : demande à Optuna de proposer une valeur
        
        params = {
            **base_params,  # Paramètres de base (objectif, métrique, etc.)
            
            # learning_rate (alias "eta") : vitesse d'apprentissage
            # Plus petit = apprentissage plus lent mais plus stable
            # log=True : recherche sur échelle logarithmique (0.001 à 0.3)
            "eta": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            
            # max_depth : profondeur maximale des arbres
            # Plus profond = modèle plus complexe (risque de surapprentissage)
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            
            # min_child_weight : poids minimum des feuilles
            # Plus élevé = modèle plus simple (évite le surapprentissage)
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            
            # subsample : proportion d'échantillons utilisés pour chaque arbre
            # 0.5 = utilise 50% des données (réduit le surapprentissage)
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            
            # colsample_bytree : proportion de features utilisées pour chaque arbre
            # 0.5 = utilise 50% des colonnes (réduit le surapprentissage)
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            
            # lambda (reg_lambda) : régularisation L2
            # Plus élevé = modèle plus simple (évite le surapprentissage)
            "lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            
            # alpha (reg_alpha) : régularisation L1
            # Plus élevé = modèle plus simple (sélection de features)
            "alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            
            # gamma : seuil minimum de gain pour diviser un nœud
            # Plus élevé = modèle plus simple (moins de divisions)
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        
        # Nombre d'arbres (boosting rounds)
        # Plus d'arbres = modèle plus complexe (mais risque de surapprentissage)
        num_boost_round = trial.suggest_int("n_estimators", 300, 1200, step=100)

        # ====================================================================
        # ENTRAÎNEMENT AVEC EARLY STOPPING
        # ====================================================================
        # early_stopping_rounds : arrête l'entraînement si AUC ne s'améliore pas
        # Exemple : si AUC ne s'améliore pas pendant 50 rounds, arrêter
        booster = xgb.train(
            params=params,                    # Hyperparamètres proposés par Optuna
            dtrain=dtrain,                     # Données d'entraînement
            num_boost_round=num_boost_round,  # Nombre maximum d'arbres
            evals=[(dvalid, "valid")],         # Jeu de validation (pour early stopping)
            early_stopping_rounds=50,          # Patience (50 rounds sans amélioration)
            verbose_eval=False,                # Ne pas afficher les logs
        )

        # ====================================================================
        # RÉCUPÉRATION DU MEILLEUR SCORE AUC
        # ====================================================================
        # XGBoost stocke le meilleur score AUC dans booster.best_score
        auc_val = float(getattr(booster, "best_score", np.nan))
        
        if np.isnan(auc_val):
            # Fallback : calculer l'AUC manuellement si best_score n'existe pas
            best_iter = getattr(booster, "best_iteration", None)
            if best_iter is not None:
                # Utiliser le meilleur nombre d'itérations trouvé par early stopping
                proba_val = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
            else:
                # Utiliser toutes les itérations
                proba_val = booster.predict(dvalid)
            # Calculer l'AUC avec sklearn
            auc_val = roc_auc_score(y_val.values, proba_val)

        # Retourner l'AUC (Optuna cherche à maximiser cette valeur)
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
    Lance l'optimisation Optuna pour trouver les meilleurs hyperparamètres.
    
    QU'EST-CE QU'OPTUNA ?
    =====================
    Optuna est une bibliothèque d'optimisation d'hyperparamètres.
    Elle teste automatiquement différentes combinaisons pour trouver les meilleures.
    
    ALGORITHME TPE (Tree-structured Parzen Estimator) :
    ===================================================
    - Apprend des essais précédents pour proposer de meilleures valeurs
    - Plus intelligent qu'une recherche aléatoire ou une grille
    
    PARAMÈTRES :
    ============
    - X_train_full, y_train_full : toutes les données d'entraînement
    - n_trials : nombre d'essais Optuna (défaut : 60)
    - timeout : temps maximum en secondes (défaut : None = pas de limite)
    - seed : graine aléatoire pour la reproductibilité
    
    RETOUR :
    ========
    Dict contenant :
    - best_params : meilleurs hyperparamètres trouvés
    - best_num_boost_round : meilleur nombre d'arbres
    - study : objet Optuna (pour analyse)
    """
    # Diviser les données d'entraînement en train/validation
    # 15% pour la validation (pour guider early stopping)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.15,           # 15% pour validation
        random_state=seed,        # Graine pour reproductibilité
        stratify=y_train_full     # Conserver la proportion de classes
    )

    # ========================================================================
    # PARAMÈTRES DE BASE (non optimisés)
    # ========================================================================
    base_params = {
        "objective": "binary:logistic",  # Classification binaire (logistique)
        "eval_metric": "auc",            # Métrique à optimiser (AUC)
        "tree_method": "hist",           # Méthode de construction d'arbres (rapide)
        # "gpu_hist" si GPU disponible (plus rapide)
        "seed": seed,                    # Graine aléatoire
        "verbosity": 0,                  # Pas de logs
        "scale_pos_weight": _compute_scale_pos_weight(y_train_full),  # Équilibrage des classes
    }

    # Créer la fonction objective
    objective = _optuna_objective_factory(X_tr, y_tr, X_val, y_val, base_params)

    # ========================================================================
    # CRÉER L'ÉTUDE OPTUNA
    # ========================================================================
    study = optuna.create_study(
        direction="maximize",              # Maximiser l'AUC
        sampler=TPESampler(seed=seed),     # Algorithme TPE
        pruner=MedianPruner(n_warmup_steps=10),  # Arrêter les essais peu prometteurs
    )

    # ========================================================================
    # LANCER L'OPTIMISATION
    # ========================================================================
    study.optimize(
        objective,                    # Fonction à optimiser
        n_trials=n_trials,           # Nombre d'essais
        timeout=timeout,             # Temps maximum
        show_progress_bar=True,      # Afficher la barre de progression
        gc_after_trial=True,         # Nettoyer la mémoire après chaque essai
    )

    # Afficher les résultats
    print("🔎 Meilleur AUC (val) :", study.best_value)
    print("🔧 Meilleurs hyperparamètres :", study.best_trial.params)

    # ========================================================================
    # RECOMPOSER LES PARAMÈTRES FINAUX
    # ========================================================================
    # Convertir les hyperparamètres d'Optuna au format XGBoost
    best = study.best_trial.params
    best_params = {
        **base_params,
        "eta": best["learning_rate"],        # learning_rate -> eta
        "max_depth": best["max_depth"],
        "min_child_weight": best["min_child_weight"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "lambda": best["reg_lambda"],        # reg_lambda -> lambda
        "alpha": best["reg_alpha"],          # reg_alpha -> alpha
        "gamma": best["gamma"],
    }
    best_num_boost_round = best["n_estimators"]

    return {"best_params": best_params, "best_num_boost_round": best_num_boost_round, "study": study}


# ============================================================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================================================

def train_model():
    """
    Pipeline complet d'entraînement du modèle.
    
    ÉTAPES :
    ========
    1. Charger les données depuis PostgreSQL
    2. Préparer les données (nettoyage, encodage, features)
    3. Diviser en train/test (80/20)
    4. Optimiser les hyperparamètres avec Optuna
    5. Entraîner le modèle final avec les meilleurs hyperparamètres
    6. Évaluer sur le jeu de test
    7. Sauvegarder le modèle et tous les artefacts
    
    RETOUR :
    ========
    Tuple contenant :
    - booster : modèle XGBoost entraîné
    - ohe : encodeur OneHot (pour la production)
    - ordinal_encoder : encodeur ordinal (pour la production)
    - feature_names : liste des noms de features (ordre important)
    """
    # Ignorer les warnings pour des logs plus propres
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # ========================================================================
    # ÉTAPE 1 : CHARGER LES DONNÉES
    # ========================================================================
    print("📊 Chargement des données depuis PostgreSQL...")
    df = load_data_from_db()
    print(f"✅ {len(df)} lignes chargées")

    # ========================================================================
    # ÉTAPE 2 : PRÉPARER LES DONNÉES
    # ========================================================================
    print("🔧 Préparation des données...")
    df_model, ohe, ordinal_encoder = prepare_data(df)

    # Vérifier que la variable cible existe
    if "Attrition" not in df_model.columns:
        raise ValueError("La colonne cible 'Attrition' est absente après préparation des données.")

    # Séparer les features (X) et la cible (y)
    X = df_model.drop(columns=["Attrition"])  # Toutes les colonnes sauf Attrition
    y = df_model["Attrition"].astype(int)     # Variable cible (0 ou 1)
    print(f"✅ {X.shape[1]} features préparées")

    # ========================================================================
    # ÉTAPE 3 : DIVISER EN TRAIN/TEST
    # ========================================================================
    # 80% pour l'entraînement, 20% pour le test final
    # stratify=y : conserve la proportion de classes (évite le déséquilibre)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, 
        test_size=0.2,           # 20% pour le test
        random_state=1042,       # Graine pour reproductibilité
        stratify=y               # Conserver la proportion de classes
    )

    # ========================================================================
    # ÉTAPE 4 : OPTIMISER LES HYPERPARAMÈTRES AVEC OPTUNA
    # ========================================================================
    # Récupérer les paramètres depuis les variables d'environnement
    n_trials = int(os.getenv("OPTUNA_TRIALS", "60"))      # Nombre d'essais
    timeout_env = os.getenv("OPTUNA_TIMEOUT", None)       # Timeout
    timeout = int(timeout_env) if timeout_env is not None else None

    print(f"🧪 Lancement d'Optuna (n_trials={n_trials}, timeout={timeout}) ...")
    tune_result = tune_with_optuna(
        X_train_full, y_train_full, 
        n_trials=n_trials, 
        timeout=timeout, 
        seed=1042
    )
    best_params = tune_result["best_params"]
    best_num_boost_round = tune_result["best_num_boost_round"]

    # ========================================================================
    # ÉTAPE 5 : ENTRAÎNEMENT FINAL
    # ========================================================================
    # Diviser à nouveau train_full en train/val pour l'entraînement final
    # (différent du split Optuna pour éviter le surapprentissage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.15,           # 15% pour validation
        random_state=2042,        # Graine différente
        stratify=y_train_full
    )

    # Convertir en DMatrix (format XGBoost)
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
    dvalid = xgb.DMatrix(X_val.values, label=y_val.values)

    # Entraîner le modèle final avec les meilleurs hyperparamètres
    booster = xgb.train(
        params=best_params,                    # Meilleurs hyperparamètres trouvés
        dtrain=dtrain,                         # Données d'entraînement
        num_boost_round=best_num_boost_round,  # Meilleur nombre d'arbres
        evals=[(dvalid, "valid")],             # Jeu de validation
        early_stopping_rounds=50,               # Early stopping
        verbose_eval=False,                     # Pas de logs
    )

    # ========================================================================
    # ÉTAPE 6 : ÉVALUATION SUR LE JEU DE TEST
    # ========================================================================
    # Convertir le jeu de test en DMatrix
    dtest = xgb.DMatrix(X_test.values, label=y_test.values)
    
    # Faire des prédictions avec le meilleur nombre d'itérations
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        # Utiliser le meilleur nombre d'itérations (trouvé par early stopping)
        y_proba = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    else:
        # Utiliser toutes les itérations si best_iteration n'existe pas
        y_proba = booster.predict(dtest)
    
    # Convertir les probabilités en prédictions binaires (0 ou 1)
    # Seuil = 0.5 : si proba >= 0.5, prédire 1 (part), sinon 0 (reste)
    y_pred = (y_proba >= 0.5).astype(int)

    # Afficher les métriques
    print("\n📊 Métriques sur le jeu de test :")
    print(classification_report(y_test, y_pred))  # Précision, rappel, F1
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")  # AUC-ROC
    print(f"AUC-PR: {average_precision_score(y_test, y_proba):.4f}")  # AUC-PR

    # ========================================================================
    # ÉTAPE 7 : SAUVEGARDE DES ARTEFACTS
    # ========================================================================
    # Créer le dossier models/ s'il n'existe pas
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # a) Sauvegarder le modèle XGBoost au format JSON
    # Format JSON = très compatible, facile à charger
    booster_path = models_dir / "xgb_booster.json"
    booster.save_model(str(booster_path))

    # b) Sauvegarder les préprocesseurs et la liste des features
    # IMPORTANT : ces fichiers sont nécessaires pour faire des prédictions en production
    # L'ordre des features doit être exactement le même qu'à l'entraînement
    joblib.dump(ohe, models_dir / "onehot_encoder.joblib")              # Encodeur OneHot
    joblib.dump(ordinal_encoder, models_dir / "ordinal_encoder.joblib")  # Encodeur ordinal
    feature_names = list(X.columns)                                      # Liste des features (ordre)
    joblib.dump(feature_names, models_dir / "feature_names.joblib")

    # c) Sauvegarder les meilleurs hyperparamètres (pour audit/reproductibilité)
    joblib.dump(
        {"best_params": best_params, "best_num_boost_round": best_num_boost_round},
        models_dir / "xgb_best_params.joblib",
    )

    # Afficher un résumé
    print(f"\n✅ Modèle XGBoost sauvegardé : {booster_path}")
    print(f"✅ {len(feature_names)} features sauvegardées")
    print(f"✅ Hyperparamètres Optuna sauvegardés (xgb_best_params.joblib)")

    return booster, ohe, ordinal_encoder, feature_names


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
# Si ce script est exécuté directement (pas importé), lancer l'entraînement
if __name__ == "__main__":
    train_model()
