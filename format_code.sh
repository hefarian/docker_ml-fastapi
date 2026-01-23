#!/bin/bash
# Script pour formater le code avec Black et isort
# Usage: ./format_code.sh

echo "========================================"
echo "Formatage du code avec Black et isort"
echo "========================================"

# Chercher Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "ERREUR: Python introuvable"
    echo "Installez Python ou ajoutez-le au PATH"
    exit 1
fi

echo "Python trouvé: $PYTHON_CMD"

# Installer Black et isort si nécessaire
echo ""
echo "[1/3] Installation de Black et isort..."
$PYTHON_CMD -m pip install --quiet black isort

# Formater avec Black
echo ""
echo "[2/3] Formatage avec Black (line-length=127)..."
$PYTHON_CMD -m black --line-length 127 app/ tests/

# Trier les imports avec isort
echo ""
echo "[3/3] Tri des imports avec isort..."
$PYTHON_CMD -m isort app/ tests/

echo ""
echo "========================================"
echo "Formatage terminé avec succès!"
echo "========================================"
echo ""
echo "Vous pouvez maintenant commiter les changements:"
echo "  git add ."
echo "  git commit -m 'Format code with Black and isort'"
echo ""
