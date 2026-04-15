
#!/bin/bash

# Script de démarrage du module de graphe de patients

echo "========================================="
echo "DÉMARRAGE DU MODULE GRAPHE DE PATIENTS"
echo "========================================="

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 n'est pas installé${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python3 trouvé${NC}"

# Créer un environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Création de l'environnement virtuel...${NC}"
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo -e "${YELLOW}Activation de l'environnement virtuel...${NC}"
source venv/bin/activate

# Installer les dépendances
echo -e "${YELLOW}Installation des dépendances...${NC}"
pip install -r requirements.txt

# Vérifier si les données existent
if [ ! -d "output" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Aucune donnée d'entraînement trouvée${NC}"
    echo "Vous devez d'abord entraîner le modèle avec:"
    echo "  python train.py --max_samples 1000"
    echo ""
    read -p "Voulez-vous lancer l'entraînement maintenant? (o/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo -e "${YELLOW}Lancement de l'entraînement (échantillon réduit)...${NC}"
        python train.py --max_samples 1000 --epochs 50
    else
        echo -e "${RED}Arrêt du script. Lancez d'abord l'entraînement.${NC}"
        exit 1
    fi
fi

# Lancer l'API
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}DÉMARRAGE DE L'API${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "L'API sera accessible sur: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""

python api.py --host 0.0.0.0 --port 8000