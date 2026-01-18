# ============================================================================
# FICHIER : app/model.py
# ============================================================================
#
# QU'EST-CE QUE CE FICHIER ?
# ===========================
# Ce fichier contient la classe AttritionModel qui :
# 1. Charge le modèle XGBoost entraîné et ses préprocesseurs
# 2. Transforme les données d'un employé en features pour le modèle
# 3. Fait des prédictions (l'employé va-t-il quitter ?)
#
# POURQUOI UNE CLASSE SÉPARÉE ?
# ==============================
# - Séparation des responsabilités : l'API (main.py) ne connaît pas les détails du modèle
# - Réutilisabilité : on peut utiliser le modèle ailleurs (scripts, notebooks)
# - Testabilité : plus facile de tester le modèle indépendamment
#
# ============================================================================

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


# ============================================================================
# CLASSE : AttritionModel
# ============================================================================
class AttritionModel:
    """
    Modèle de prédiction d'attrition basé sur XGBoost (Booster).
    
    QU'EST-CE QUE CETTE CLASSE ?
    ============================
    Cette classe encapsule :
    - Le modèle XGBoost entraîné (Booster)
    - Les préprocesseurs (OneHotEncoder, OrdinalEncoder)
    - La liste des features dans l'ordre attendu
    
    COMMENT ÇA MARCHE ?
    ===================
    1. À l'initialisation : charge le modèle et les préprocesseurs depuis les fichiers
    2. Pour une prédiction :
       a. Transforme les données de l'employé en features (même pipeline qu'à l'entraînement)
       b. Fait la prédiction avec XGBoost
       c. Retourne la prédiction (0 ou 1) et la probabilité (0.0 à 1.0)
    """

    def __init__(
        self,
        booster_path: str,
        ohe_path: str,
        ordinal_path: str,
        feature_names_path: str,
    ):
        """
        Charge le Booster XGBoost et les artefacts de prétraitement/ordre des features.
        
        QU'EST-CE QU'UN ARTEFACT ?
        ===========================
        Un artefact est un fichier nécessaire pour utiliser le modèle :
        - booster_path : le modèle XGBoost lui-même (format JSON)
        - ohe_path : l'encodeur OneHot (pour encoder les catégories)
        - ordinal_path : l'encodeur ordinal (pour encoder les fréquences)
        - feature_names_path : la liste des features dans l'ordre attendu
        
        POURQUOI L'ORDRE DES FEATURES EST IMPORTANT ?
        =============================================
        XGBoost attend les features dans le même ordre qu'à l'entraînement.
        Si l'ordre change, les prédictions seront incorrectes !
        
        PARAMÈTRES :
        ===========
        - booster_path : chemin vers le fichier xgb_booster.json
        - ohe_path : chemin vers onehot_encoder.joblib
        - ordinal_path : chemin vers ordinal_encoder.joblib
        - feature_names_path : chemin vers feature_names.joblib
        """
        # ====================================================================
        # CHARGEMENT DU MODÈLE XGBOOST
        # ====================================================================
        # Modèle XGBoost (Booster) au format JSON
        # xgb.Booster() : crée un objet Booster vide
        # load_model() : charge le modèle depuis le fichier JSON
        self.booster = xgb.Booster()
        self.booster.load_model(booster_path)

        # ====================================================================
        # CHARGEMENT DES PRÉPROCESSEURS
        # ====================================================================
        # Encoders & noms de colonnes
        # joblib.load() : charge un objet Python sauvegardé avec joblib
        # Ces encodeurs doivent être les mêmes qu'à l'entraînement !
        self.ohe = joblib.load(ohe_path)  # peut être None si pas d'OHE au train
        self.ordinal_encoder = joblib.load(ordinal_path)  # peut être None si pas utilisé
        self.feature_names = joblib.load(feature_names_path)  # ordre attendu en prod (list[str])

    # ========================================================================
    # MÉTHODES UTILITAIRES INTERNES
    # ========================================================================
    @staticmethod
    def _safe_divide(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """
        Division sûre avec gestion NaN/Inf et division par zéro.
        
        QU'EST-CE QU'UNE DIVISION SÛRE ?
        =================================
        En mathématiques, diviser par zéro est impossible.
        En programmation, cela crée des valeurs infinies (inf) ou NaN.
        Cette fonction évite ces problèmes.
        
        EXEMPLE :
        =========
        >>> _safe_divide([10, 20], [2, 0], fill_value=0.0)
        [5.0, 0.0]  # 20/0 devient 0.0 au lieu d'inf
        """
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
        out = pd.Series(out, index=num.index if isinstance(num, pd.Series) else None)
        return out.fillna(fill_value)

    @staticmethod
    def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
        """
        Vérifie que toutes les colonnes demandées existent dans df.
        
        UTILITÉ :
        =========
        Évite les erreurs si une colonne est absente (données incomplètes).
        """
        return all(c in df.columns for c in cols)

    # ========================================================================
    # TRANSFORMATION DES DONNÉES
    # ========================================================================
    def _transform_employee_data(self, employee_data: EmployeeData) -> np.ndarray:
        """
        Transforme les données d'un employé (Pydantic) en vecteur de features
        aligné sur self.feature_names.
        
        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode reproduit EXACTEMENT le même pipeline de préparation
        que lors de l'entraînement. C'est crucial pour que les prédictions soient correctes.
        
        PIPELINE DE TRANSFORMATION :
        ============================
        1. Convertir Pydantic -> DataFrame
        2. Normaliser augmentation_salaire_precedente
        3. Encoder les catégories (OneHot)
        4. Encoder frequence_deplacement (Ordinal)
        5. Recoder heure_supplementaires (oui/non -> 1/0)
        6. get_dummies pour les catégories restantes
        7. Convertir bool -> int
        8. Créer les features dérivées (ratios, écarts)
        9. Supprimer les colonnes corrélées
        10. Nettoyer (NaN, infini)
        11. S'assurer que toutes les features attendues existent
        12. Réordonner selon l'ordre attendu
        
        RETOUR :
        ========
        np.ndarray : matrice numpy avec les features dans l'ordre attendu
        """
        # ====================================================================
        # ÉTAPE 1 : CONVERTIR PYDANTIC -> DATAFRAME
        # ====================================================================
        # 1) Pydantic -> dict -> DataFrame (1 ligne)
        # model_dump() : convertit l'objet Pydantic en dictionnaire
        data_dict = employee_data.model_dump()
        # pd.DataFrame([data_dict]) : crée un DataFrame avec 1 ligne
        df = pd.DataFrame([data_dict])

        # ====================================================================
        # ÉTAPE 1.b : NORMALISER augmentation_salaire_precedente
        # ====================================================================
        # Optionnel: normalisation du champ 'augmentation_salaire_precedente'
        # si celui-ci arrive en chaîne avec un '%' (pour rester cohérent avec le train)
        if "augmentation_salaire_precedente" in df.columns:
            ser = df["augmentation_salaire_precedente"].astype(str)
            if ser.str.contains("%").any():
                # Convertir "11%" -> 0.11
                df["augmentation_salaire_precedente"] = (
                    ser.str.replace("%", "", regex=False)
                       .str.replace(",", ".", regex=False)
                       .str.replace(r"\s+", "", regex=True)
                       .pipe(pd.to_numeric, errors="coerce")
                       .div(100.0)
                ).fillna(0.0)

        # ====================================================================
        # ÉTAPE 2 : ENCODAGE ONEHOT
        # ====================================================================
        # 2) OneHot sur nominales (même liste que le train)
        # Colonnes nominales : pas d'ordre naturel (département, poste, etc.)
        colonnes_nominales = [c for c in ["departement", "poste", "domaine_etude", "statut_marital"] if c in df.columns]
        if self.ohe is not None and colonnes_nominales:
            # transform() : applique l'encodage (sans réapprendre, utilise celui de l'entraînement)
            df_ohe = pd.DataFrame(
                self.ohe.transform(df[colonnes_nominales]),
                columns=self.ohe.get_feature_names_out(colonnes_nominales),
                index=df.index,
            )
            # Supprimer les colonnes originales et ajouter les colonnes encodées
            df = df.drop(columns=colonnes_nominales)
            df = pd.concat([df, df_ohe], axis=1)

        # ====================================================================
        # ÉTAPE 3 : ENCODAGE ORDINAL
        # ====================================================================
        # 3) Ordinal pour 'frequence_deplacement' si présent
        # Ordinal : les catégories ont un ordre (Aucun < Occasionnel < Frequent)
        if self.ordinal_encoder is not None and "frequence_deplacement" in df.columns:
            # S'assure qu'on manipule bien des str (pour éviter les erreurs)
            df[["frequence_deplacement"]] = self.ordinal_encoder.transform(
                df[["frequence_deplacement"]]
            )

        # ====================================================================
        # ÉTAPE 4 : RECODAGE heure_supplementaires
        # ====================================================================
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

        # ====================================================================
        # ÉTAPE 5 : GET_DUMMIES DE SECOURS
        # ====================================================================
        # 5) get_dummies de secours si d'autres catégorielles restent
        # Pour les catégories non encodées précédemment
        df = pd.get_dummies(df, drop_first=True)

        # ====================================================================
        # ÉTAPE 6 : CONVERSION BOOL -> INT
        # ====================================================================
        # 6) Bool -> int
        # Les colonnes booléennes doivent être converties en 0/1
        bool_cols = df.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)

        # ====================================================================
        # ÉTAPE 7 : FEATURES DÉRIVÉES
        # ====================================================================
        # 7) Features dérivées (robustes aux colonnes manquantes)
        # Fonction helper pour créer des ratios de manière sûre
        def add_ratio(safe_df: pd.DataFrame, num_col: str, den_col: str, out_col: str):
            """Ajoute un ratio (num/den) au DataFrame de manière sûre."""
            if num_col in safe_df.columns and den_col in safe_df.columns:
                safe_df[out_col] = self._safe_divide(safe_df[num_col], safe_df[den_col], fill_value=0.0)
            else:
                safe_df[out_col] = 0.0

        # Créer les mêmes features dérivées qu'à l'entraînement
        add_ratio(df, "annees_dans_l_entreprise", "annee_experience_totale", "ratio_anciennete")
        add_ratio(df, "annees_dans_le_poste_actuel", "annees_dans_l_entreprise", "ratio_poste")

        # Écart d'évaluation
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

        # Indice de promotion récente
        if "annees_depuis_la_derniere_promotion" in df.columns:
            df["annees_depuis_la_derniere_promotion"] = pd.to_numeric(
                df["annees_depuis_la_derniere_promotion"], errors="coerce"
            )
            df["indice_recente_promo"] = 1.0 / (df["annees_depuis_la_derniere_promotion"].fillna(0) + 1.0)
        else:
            df["indice_recente_promo"] = 0.0
        df = df.drop(columns="annees_depuis_la_derniere_promotion", errors="ignore")

        # ====================================================================
        # ÉTAPE 8 : SUPPRESSION DE COLONNES CORRÉLÉES
        # ====================================================================
        # 8) Drop de colonnes corrélées (si présentes)
        # Ces colonnes ont été supprimées à l'entraînement (redondantes)
        colonnes_a_supprimer = [
            "annees_dans_le_poste_actuel",
            "annes_sous_responsable_actuel",
            "poste_Ressources Humaines",
            "annee_experience_totale",
            "niveau_hierarchique_poste",
        ]
        df = df.drop(columns=[c for c in colonnes_a_supprimer if c in df.columns], errors="ignore")

        # ====================================================================
        # ÉTAPE 9 : NETTOYAGE FINAL
        # ====================================================================
        # 9) Nettoyage final
        # Remplacer les valeurs infinies et NaN par 0.0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ====================================================================
        # ÉTAPE 10 : S'ASSURER QUE TOUTES LES FEATURES EXISTENT
        # ====================================================================
        # 10) S'assurer que toutes les features attendues existent
        # Si une feature manque (ex: nouvelle catégorie), on la crée avec valeur 0.0
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0

        # ====================================================================
        # ÉTAPE 11 : RÉORDONNER LES COLONNES
        # ====================================================================
        # 11) Réordonner les colonnes selon l'ordre attendu
        # CRUCIAL : XGBoost attend les features dans le même ordre qu'à l'entraînement
        df = df[self.feature_names]

        # Renvoie la matrice numpy (format attendu par XGBoost)
        return df.values

    # ========================================================================
    # MÉTHODE DE PRÉDICTION
    # ========================================================================
    def predict(self, employee_data: EmployeeData) -> Tuple[int, float]:
        """
        Prédit l'attrition pour un employé.
        
        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        1. Transforme les données de l'employé en features
        2. Fait la prédiction avec XGBoost
        3. Retourne la prédiction (0 ou 1) et la probabilité (0.0 à 1.0)
        
        PARAMÈTRES :
        ===========
        - employee_data : données de l'employé (objet EmployeeData Pydantic)
        
        RETOUR :
        ========
        Tuple[int, float] :
        - int : prédiction (0 = reste, 1 = quitte)
        - float : probabilité d'attrition (0.0 à 1.0)
        
        EXEMPLE :
        =========
        >>> model = AttritionModel(...)
        >>> prediction, proba = model.predict(employee_data)
        >>> print(f"Prédiction: {prediction}, Probabilité: {proba:.2%}")
        Prédiction: 1, Probabilité: 75.23%
        """
        # ====================================================================
        # ÉTAPE 1 : TRANSFORMER LES DONNÉES
        # ====================================================================
        # Transforme en features alignées
        # _transform_employee_data() : applique tout le pipeline de préparation
        X = self._transform_employee_data(employee_data)

        # ====================================================================
        # ÉTAPE 2 : CRÉER UN DMATRIX
        # ====================================================================
        # XGBoost attend un DMatrix (format optimisé pour XGBoost)
        dmat = xgb.DMatrix(X)

        # ====================================================================
        # ÉTAPE 3 : FAIRE LA PRÉDICTION
        # ====================================================================
        # Si le booster a une best_iteration (early stopping), on la respecte pour la prédiction
        # best_iteration : meilleur nombre d'itérations trouvé par early stopping
        best_iter = getattr(self.booster, "best_iteration", None)
        if best_iter is not None:
            # Utiliser seulement les meilleures itérations (évite le surapprentissage)
            proba = float(self.booster.predict(dmat, iteration_range=(0, best_iter + 1))[0])
        else:
            # Utiliser toutes les itérations si best_iteration n'existe pas
            proba = float(self.booster.predict(dmat)[0])

        # ====================================================================
        # ÉTAPE 4 : CONVERTIR LA PROBABILITÉ EN PRÉDICTION BINAIRE
        # ====================================================================
        # Seuil = 0.5 : si proba >= 0.5, prédire 1 (quitte), sinon 0 (reste)
        label = int(proba >= 0.5)
        return label, proba


# ============================================================================
# FONCTION : get_model (Factory avec cache)
# ============================================================================
@lru_cache(maxsize=1)
def get_model() -> AttritionModel:
    """
    Charge le modèle et les artefacts (avec cache).
    
    QU'EST-CE QUE @lru_cache ?
    ==========================
    @lru_cache est un décorateur Python qui met en cache le résultat d'une fonction.
    - Première fois : la fonction s'exécute et charge le modèle
    - Fois suivantes : retourne directement le modèle mis en cache
    - maxsize=1 : garde seulement 1 résultat en cache
    
    POURQUOI C'EST IMPORTANT ?
    ===========================
    - Évite de recharger le modèle à chaque requête (très lent)
    - Économise la mémoire (une seule instance en mémoire)
    - Pattern "singleton" : une seule instance partagée
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Première fois qu'on appelle get_model() :
       - Charge les fichiers depuis le disque
       - Crée une instance AttritionModel
       - Met en cache l'instance
    2. Fois suivantes :
       - Retourne directement l'instance mise en cache
       - Pas de rechargement des fichiers
    
    RETOUR :
    ========
    AttritionModel : instance du modèle (singleton)
    
    EXCEPTIONS :
    ===========
    FileNotFoundError : si un fichier artefact est manquant
    """
    # Déterminer le dossier des modèles
    # On part du chemin du modèle (paramétré) et on remonte au dossier parent
    models_dir = Path(settings.MODEL_PATH).parent

    # Chemins des fichiers artefacts
    booster_path = models_dir / "xgb_booster.json"
    ohe_path = models_dir / "onehot_encoder.joblib"
    ordinal_path = models_dir / "ordinal_encoder.joblib"
    feature_names_path = models_dir / "feature_names.joblib"

    # ========================================================================
    # VÉRIFICATION DE LA PRÉSENCE DES FICHIERS
    # ========================================================================
    # Vérifie la présence des fichiers nécessaires
    # Si un fichier manque, on lève une exception explicite
    for path in [booster_path, ohe_path, ordinal_path, feature_names_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier artefact manquant: {path}. "
                f"Veuillez exécuter le script d'entraînement pour générer le modèle et les artefacts."
            )

    # ========================================================================
    # CRÉATION DE L'INSTANCE
    # ========================================================================
    # Créer et retourner l'instance AttritionModel
    # Cette instance sera mise en cache par @lru_cache
    return AttritionModel(
        booster_path=str(booster_path),
        ohe_path=str(ohe_path),
        ordinal_path=str(ordinal_path),
        feature_names_path=str(feature_names_path),
    )
