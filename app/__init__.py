# Package app
"""
Fichier __init__.py pour le package 'app'

QU'EST-CE QU'UN __init__.py ?
=============================
Le fichier __init__.py transforme un simple dossier en "package" Python.
Un package est un dossier qui contient des modules Python et peut être importé.

POURQUOI C'EST NÉCESSAIRE ?
===========================
Sans ce fichier, Python ne reconnaît pas le dossier "app" comme un package.
Avec ce fichier, on peut faire :
    from app.main import app
    from app.schemas import EmployeeData

SANS ce fichier, ces imports échoueraient avec "ModuleNotFoundError".

CONTENU DE CE FICHIER
=====================
Ce fichier est vide (ou presque) car on n'a pas besoin d'initialiser quoi que ce soit
au moment de l'import du package. On pourrait y mettre :
- Des imports pour faciliter l'utilisation : from app.schemas import EmployeeData
- Des constantes partagées
- Du code d'initialisation

Mais pour ce projet, un fichier vide suffit.
"""
