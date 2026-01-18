"""
Schémas Pydantic pour l'API de prédiction d'attrition

QU'EST-CE QU'UN SCHÉMA PYDANTIC ?
=================================
Un schéma définit la structure et les règles de validation des données.

Exemple concret :
- Si quelqu'un envoie {"age": -5}, Pydantic va rejeter car age doit être >= 18
- Si quelqu'un envoie {"age": "trente-cinq"}, Pydantic va rejeter car age doit être un nombre
- Si quelqu'un oublie le champ "age", Pydantic va rejeter car il est obligatoire

POURQUOI C'EST IMPORTANT ?
===========================
1. Validation automatique : on n'a pas besoin de vérifier manuellement chaque champ
2. Documentation automatique : FastAPI génère la doc Swagger à partir de ces schémas
3. Sécurité : empêche les données malformées d'atteindre le modèle
4. Clarté : on voit exactement quelles données sont attendues

COMMENT ÇA MARCHE ?
===================
- BaseModel : classe de base de Pydantic (tous nos schémas en héritent)
- Field(...) : définit un champ obligatoire avec des contraintes
- Field(..., ge=18, le=100) : ge = "greater or equal" (>=), le = "less or equal" (<=)
- Optional[int] : le champ peut être None (optionnel)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================================================
# SCHÉMA : Données d'un employé (entrée de l'API)
# ============================================================================
class EmployeeData(BaseModel):
    """
    Données d'un employé pour la prédiction d'attrition.
    
    Cette classe définit TOUTES les informations nécessaires sur un employé
    pour que le modèle puisse faire une prédiction.
    
    Chaque champ a :
    - Un type (int, str, float) : le type de données attendu
    - Des contraintes (ge=18, le=100) : les valeurs acceptées
    - Une description : explication du champ (visible dans la doc Swagger)
    
    Exemple d'utilisation :
        employee = EmployeeData(
            age=35,
            genre="M",
            revenu_mensuel=5000,
            ...
        )
    """
    
    # Âge de l'employé (en années)
    # ge=18 : l'âge doit être >= 18 (pas de mineurs)
    # le=100 : l'âge doit être <= 100 (valeur réaliste)
    age: int = Field(..., ge=18, le=100, description="Âge de l'employé")
    
    # Genre de l'employé
    # Pas de contrainte particulière, juste une chaîne de caractères
    genre: str = Field(..., description="Genre (ex: 'M', 'F')")
    
    # Revenu mensuel en euros
    # ge=0 : le revenu ne peut pas être négatif
    revenu_mensuel: float = Field(..., ge=0, description="Revenu mensuel")
    
    # Statut marital (ex: "Marié(e)", "Célibataire", "Divorcé(e)")
    statut_marital: str = Field(..., description="Statut marital")
    
    # Département dans l'entreprise (ex: "R&D", "Commercial", "RH")
    departement: str = Field(..., description="Département")
    
    # Poste occupé (ex: "Data Scientist", "Manager", "Consultant")
    poste: str = Field(..., description="Poste occupé")
    
    # Nombre d'expériences professionnelles précédentes
    # ge=0 : ne peut pas être négatif
    nombre_experiences_precedentes: int = Field(..., ge=0, description="Nombre d'expériences précédentes")
    
    # Nombre total d'années d'expérience professionnelle
    annee_experience_totale: int = Field(..., ge=0, description="Années d'expérience totale")
    
    # Nombre d'années passées dans l'entreprise actuelle
    annees_dans_l_entreprise: int = Field(..., ge=0, description="Années dans l'entreprise")
    
    # Nombre d'années dans le poste actuel
    annees_dans_le_poste_actuel: int = Field(..., ge=0, description="Années dans le poste actuel")
    
    # Satisfaction concernant l'environnement de travail (échelle 1-5)
    # ge=1 : minimum 1, le=5 : maximum 5
    satisfaction_employee_environnement: int = Field(..., ge=1, le=5, description="Satisfaction environnement (1-5)")
    
    # Note de l'évaluation précédente (sur 5)
    note_evaluation_precedente: float = Field(..., ge=0, le=5, description="Note évaluation précédente")
    
    # Niveau hiérarchique du poste (1 = bas, plus élevé = plus haut niveau)
    niveau_hierarchique_poste: int = Field(..., ge=1, description="Niveau hiérarchique")
    
    # Satisfaction concernant la nature du travail (1-5)
    satisfaction_employee_nature_travail: int = Field(..., ge=1, le=5, description="Satisfaction nature travail (1-5)")
    
    # Satisfaction concernant l'équipe (1-5)
    satisfaction_employee_equipe: int = Field(..., ge=1, le=5, description="Satisfaction équipe (1-5)")
    
    # Satisfaction concernant l'équilibre vie pro/perso (1-5)
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=1, le=5, description="Satisfaction équilibre pro/perso (1-5)")
    
    # Note de l'évaluation actuelle (sur 5)
    note_evaluation_actuelle: float = Field(..., ge=0, le=5, description="Note évaluation actuelle")
    
    # Fait-il des heures supplémentaires ? ("Oui" ou "Non")
    heure_supplementaires: str = Field(..., description="Heures supplémentaires ('Oui' ou 'Non')")
    
    # Nombre de participations au PEE (Plan d'Épargne Entreprise)
    nombre_participation_pee: int = Field(..., ge=0, description="Nombre de participations PEE")
    
    # Nombre de formations suivies
    nb_formations_suivies: int = Field(..., ge=0, description="Nombre de formations suivies")
    
    # Distance entre le domicile et le travail (en kilomètres)
    distance_domicile_travail: int = Field(..., ge=0, description="Distance domicile-travail (km)")
    
    # Niveau d'éducation (1 = bas, plus élevé = niveau plus haut)
    niveau_education: int = Field(..., ge=1, description="Niveau d'éducation")
    
    # Domaine d'étude (ex: "Data Science", "Informatique", "Commerce")
    domaine_etude: str = Field(..., description="Domaine d'étude")
    
    # Fréquence des déplacements professionnels
    # Valeurs possibles : "Aucun", "Occasionnel", "Frequent"
    frequence_deplacement: str = Field(..., description="Fréquence de déplacement ('Aucun', 'Occasionnel', 'Frequent')")
    
    # Nombre d'années depuis la dernière promotion
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0, description="Années depuis dernière promotion")
    
    # Nombre d'années sous le responsable actuel
    annes_sous_responsable_actuel: int = Field(..., ge=0, description="Années sous responsable actuel")
    
    # Augmentation salariale précédente (en proportion, pas en pourcentage)
    # Exemple : 0.11 = 11%, 0.05 = 5%
    # ge=0 : pas négatif, le=1 : pas plus de 100%
    augmentation_salaire_precedente: float = Field(..., ge=0, le=1, description="Augmentation salaire précédente (proportion, ex: 0.11 pour 11%)")


# ============================================================================
# SCHÉMA : Requête de prédiction (ce que le client envoie à l'API)
# ============================================================================
class PredictRequest(BaseModel):
    """
    Requête de prédiction avec données d'employé.
    
    Structure de la requête HTTP POST /predict :
    {
        "employee_data": {
            "age": 35,
            "genre": "M",
            ...
        }
    }
    
    Le champ "employee_data" contient toutes les informations de l'employé
    définies dans la classe EmployeeData ci-dessus.
    """
    employee_data: EmployeeData  # Contient toutes les données de l'employé


# ============================================================================
# SCHÉMA : Réponse de prédiction (ce que l'API renvoie au client)
# ============================================================================
class PredictResponse(BaseModel):
    """
    Réponse de prédiction retournée par l'API.
    
    Structure de la réponse HTTP :
    {
        "prediction": 0,           # 0 = reste, 1 = quitte
        "probability": 0.2345,     # Probabilité d'attrition (0.0 à 1.0)
        "model_version": "1.0.0",  # Version du modèle utilisé
        "prediction_id": 123       # ID de l'enregistrement en base (optionnel)
    }
    """
    # Prédiction : 0 = l'employé reste, 1 = l'employé quitte
    prediction: int = Field(..., description="Prédiction (0 = reste, 1 = quitte)")
    
    # Probabilité d'attrition (entre 0.0 et 1.0)
    # 0.0 = aucune chance de partir, 1.0 = départ certain
    # ge=0 : minimum 0, le=1 : maximum 1
    probability: float = Field(..., ge=0, le=1, description="Probabilité d'attrition")
    
    # Version du modèle utilisé (pour traçabilité)
    model_version: str = Field(..., description="Version du modèle")
    
    # ID de la prédiction enregistrée en base de données (si l'enregistrement a réussi)
    # Optional[int] : peut être None si l'enregistrement a échoué
    prediction_id: Optional[int] = Field(None, description="ID de la prédiction enregistrée en DB")


# ============================================================================
# SCHÉMA : Enregistrement d'une prédiction (pour la base de données)
# ============================================================================
class PredictionRecord(BaseModel):
    """
    Enregistrement d'une prédiction dans la base de données.
    
    Cette classe représente une ligne de la table "predictions" en base de données.
    Elle contient :
    - Les données d'entrée (input_data) : ce qui a été envoyé au modèle
    - La prédiction : le résultat du modèle
    - La probabilité : la confiance du modèle
    - La date de création : quand la prédiction a été faite
    """
    # ID unique de la prédiction (généré automatiquement par la base)
    id: Optional[int] = None
    
    # Données d'entrée (dictionnaire JSON)
    # Dict[str, Any] : dictionnaire avec des clés string et valeurs de n'importe quel type
    input_data: Dict[str, Any] = Field(..., description="Données d'entrée")
    
    # Prédiction : 0 ou 1
    prediction: int = Field(..., description="Prédiction (0 ou 1)")
    
    # Probabilité : entre 0.0 et 1.0
    probability: float = Field(..., ge=0, le=1, description="Probabilité")
    
    # Date et heure de création de la prédiction
    # Optional[datetime] : peut être None si pas encore enregistré
    created_at: Optional[datetime] = None


# ============================================================================
# SCHÉMA : Réponse du endpoint de santé
# ============================================================================
class HealthResponse(BaseModel):
    """
    Réponse du endpoint /health.
    
    Ce endpoint permet de vérifier que l'API fonctionne correctement.
    
    Structure de la réponse :
    {
        "status": "ok",           # "ok" si tout va bien, "degraded" sinon
        "model_loaded": true,      # Le modèle est-il chargé ?
        "db_connected": true       # La base de données est-elle accessible ?
    }
    """
    # Statut global : "ok" si tout fonctionne, "degraded" si problème
    status: str
    
    # Le modèle de machine learning est-il chargé et prêt ?
    # False = le modèle n'est pas disponible (fichiers manquants, erreur de chargement)
    model_loaded: bool = False
    
    # La base de données PostgreSQL est-elle accessible ?
    # False = impossible de se connecter à la base
    db_connected: bool = False
