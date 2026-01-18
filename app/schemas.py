"""
Schémas Pydantic pour l'API de prédiction d'attrition
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Schéma pour les données d'entrée brutes (avant transformation)
class EmployeeData(BaseModel):
    """Données d'un employé pour la prédiction d'attrition"""
    age: int = Field(..., ge=18, le=100, description="Âge de l'employé")
    genre: str = Field(..., description="Genre (ex: 'M', 'F')")
    revenu_mensuel: float = Field(..., ge=0, description="Revenu mensuel")
    statut_marital: str = Field(..., description="Statut marital")
    departement: str = Field(..., description="Département")
    poste: str = Field(..., description="Poste occupé")
    nombre_experiences_precedentes: int = Field(..., ge=0, description="Nombre d'expériences précédentes")
    annee_experience_totale: int = Field(..., ge=0, description="Années d'expérience totale")
    annees_dans_l_entreprise: int = Field(..., ge=0, description="Années dans l'entreprise")
    annees_dans_le_poste_actuel: int = Field(..., ge=0, description="Années dans le poste actuel")
    satisfaction_employee_environnement: int = Field(..., ge=1, le=5, description="Satisfaction environnement (1-5)")
    note_evaluation_precedente: float = Field(..., ge=0, le=5, description="Note évaluation précédente")
    niveau_hierarchique_poste: int = Field(..., ge=1, description="Niveau hiérarchique")
    satisfaction_employee_nature_travail: int = Field(..., ge=1, le=5, description="Satisfaction nature travail (1-5)")
    satisfaction_employee_equipe: int = Field(..., ge=1, le=5, description="Satisfaction équipe (1-5)")
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=1, le=5, description="Satisfaction équilibre pro/perso (1-5)")
    note_evaluation_actuelle: float = Field(..., ge=0, le=5, description="Note évaluation actuelle")
    heure_supplementaires: str = Field(..., description="Heures supplémentaires ('Oui' ou 'Non')")
    nombre_participation_pee: int = Field(..., ge=0, description="Nombre de participations PEE")
    nb_formations_suivies: int = Field(..., ge=0, description="Nombre de formations suivies")
    distance_domicile_travail: int = Field(..., ge=0, description="Distance domicile-travail (km)")
    niveau_education: int = Field(..., ge=1, description="Niveau d'éducation")
    domaine_etude: str = Field(..., description="Domaine d'étude")
    frequence_deplacement: str = Field(..., description="Fréquence de déplacement ('Aucun', 'Occasionnel', 'Frequent')")
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0, description="Années depuis dernière promotion")
    annes_sous_responsable_actuel: int = Field(..., ge=0, description="Années sous responsable actuel")
    augmentation_salaire_precedente: float = Field(..., ge=0, le=1, description="Augmentation salaire précédente (proportion, ex: 0.11 pour 11%)")

class PredictRequest(BaseModel):
    """Requête de prédiction avec données d'employé"""
    employee_data: EmployeeData

class PredictResponse(BaseModel):
    """Réponse de prédiction"""
    prediction: int = Field(..., description="Prédiction (0 = reste, 1 = quitte)")
    probability: float = Field(..., ge=0, le=1, description="Probabilité d'attrition")
    model_version: str = Field(..., description="Version du modèle")
    prediction_id: Optional[int] = Field(None, description="ID de la prédiction enregistrée en DB")

class PredictionRecord(BaseModel):
    """Enregistrement d'une prédiction dans la base de données"""
    id: Optional[int] = None
    input_data: Dict[str, Any] = Field(..., description="Données d'entrée")
    prediction: int = Field(..., description="Prédiction (0 ou 1)")
    probability: float = Field(..., ge=0, le=1, description="Probabilité")
    created_at: Optional[datetime] = None

class HealthResponse(BaseModel):
    """Réponse du endpoint de santé"""
    status: str
    model_loaded: bool = False
    db_connected: bool = False
