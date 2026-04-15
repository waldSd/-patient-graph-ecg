"""
API FastAPI pour servir les données du graphe de patients
"""
import os
import json
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# ========== Modèles Pydantic ==========

class Node(BaseModel):
    """Nœud du graphe (patient)"""
    id: int
    representativeness: Optional[float] = None
    atypicality: Optional[float] = None
    is_prototype: Optional[bool] = False
    is_atypical: Optional[bool] = False
    # Métadonnées supplémentaires du dataset PTB-XL
    patient_id: Optional[int] = None
    age: Optional[float] = None
    sex: Optional[int] = None


class Edge(BaseModel):
    """Arête du graphe (similarité entre patients)"""
    source: int
    target: int
    weight: float
    distance: float


class GraphData(BaseModel):
    """Données complètes du graphe"""
    nodes: List[Node]
    edges: List[Edge]
    stats: Dict


class PatientNeighbors(BaseModel):
    """Voisins d'un patient"""
    patient_id: int
    neighbors: List[Dict]


class Metrics(BaseModel):
    """Métriques de représentativité et atypicité"""
    representativeness: List[float]
    atypicality: List[float]
    prototypes: List[int]
    atypical: List[int]


# ========== Application FastAPI ==========

app = FastAPI(
    title="Patient Graph API",
    description="API pour visualiser et interroger le graphe de patients basé sur les embeddings ECG",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis le frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Variables globales ==========

GRAPH_DATA_PATH = os.getenv("GRAPH_DATA_PATH", "./output/graph/graph_data.json")
METRICS_PATH = os.getenv("METRICS_PATH", "./output/graph/metrics.json")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./output/embeddings/embeddings.npy")

graph_data = None
metrics_data = None
embeddings = None


# ========== Fonctions de chargement ==========

def load_data():
    """Charge les données au démarrage"""
    global graph_data, metrics_data, embeddings

    try:
        # Charger les données du graphe
        if os.path.exists(GRAPH_DATA_PATH):
            with open(GRAPH_DATA_PATH, 'r') as f:
                graph_data = json.load(f)
            print(f"✓ Données du graphe chargées : {len(graph_data['nodes'])} nœuds")
        else:
            print(f"⚠️  Fichier de graphe non trouvé : {GRAPH_DATA_PATH}")

        # Charger les métriques
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
            print(f"✓ Métriques chargées")
        else:
            print(f"⚠️  Fichier de métriques non trouvé : {METRICS_PATH}")

        # Charger les embeddings
        if os.path.exists(EMBEDDINGS_PATH):
            embeddings = np.load(EMBEDDINGS_PATH)
            print(f"✓ Embeddings chargés : {embeddings.shape}")
        else:
            print(f"⚠️  Fichier d'embeddings non trouvé : {EMBEDDINGS_PATH}")

    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")


@app.on_event("startup")
async def startup_event():
    """Charger les données au démarrage du serveur"""
    load_data()


# ========== Endpoints ==========

@app.get("/")
async def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "Patient Graph API",
        "version": "1.0.0",
        "endpoints": {
            "graph": "/api/graph",
            "metrics": "/api/metrics",
            "patient": "/api/patient/{patient_id}",
            "neighbors": "/api/patient/{patient_id}/neighbors",
            "prototypes": "/api/prototypes",
            "atypical": "/api/atypical",
            "stats": "/api/stats"
        }
    }


@app.get("/api/graph", response_model=GraphData)
async def get_graph(
    limit_nodes: Optional[int] = Query(None, description="Limiter le nombre de nœuds retournés"),
    min_weight: Optional[float] = Query(0.0, description="Poids minimum des arêtes")
):
    """
    Retourne les données complètes du graphe

    - **limit_nodes**: Limite le nombre de nœuds (pour performance)
    - **min_weight**: Filtre les arêtes par poids minimum
    """
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Données du graphe non disponibles")

    # Copier les données
    filtered_data = graph_data.copy()

    # Filtrer par nombre de nœuds si demandé
    if limit_nodes and limit_nodes < len(filtered_data['nodes']):
        # Garder les prototypes et atypiques en priorité
        priority_nodes = set(filtered_data.get('prototypes', []) + filtered_data.get('atypical', []))
        regular_nodes = [n for n in filtered_data['nodes'] if n['id'] not in priority_nodes]

        # Limiter les nœuds réguliers
        max_regular = max(0, limit_nodes - len(priority_nodes))
        selected_nodes = [n for n in filtered_data['nodes'] if n['id'] in priority_nodes] + regular_nodes[:max_regular]

        # Filtrer les arêtes pour ne garder que celles entre les nœuds sélectionnés
        selected_ids = {n['id'] for n in selected_nodes}
        filtered_data['nodes'] = selected_nodes
        filtered_data['edges'] = [
            e for e in filtered_data['edges']
            if e['source'] in selected_ids and e['target'] in selected_ids
        ]

    # Filtrer par poids d'arête
    if min_weight > 0:
        filtered_data['edges'] = [
            e for e in filtered_data['edges']
            if e['weight'] >= min_weight
        ]

    return filtered_data


@app.get("/api/metrics", response_model=Metrics)
async def get_metrics():
    """Retourne les métriques de représentativité et atypicité pour tous les patients"""
    if metrics_data is None:
        raise HTTPException(status_code=503, detail="Métriques non disponibles")

    return metrics_data


@app.get("/api/patient/{patient_id}")
async def get_patient(patient_id: int):
    """Retourne les informations détaillées d'un patient"""
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Données du graphe non disponibles")

    # Trouver le nœud correspondant
    node = next((n for n in graph_data['nodes'] if n['id'] == patient_id), None)

    if node is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} non trouvé")

    return node


@app.get("/api/patient/{patient_id}/neighbors")
async def get_patient_neighbors(
    patient_id: int,
    k: int = Query(5, description="Nombre de voisins à retourner")
):
    """Retourne les k voisins les plus proches d'un patient"""
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Données du graphe non disponibles")

    # Vérifier que le patient existe
    node = next((n for n in graph_data['nodes'] if n['id'] == patient_id), None)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} non trouvé")

    # Trouver toutes les arêtes connectées au patient
    edges = [
        e for e in graph_data['edges']
        if e['source'] == patient_id or e['target'] == patient_id
    ]

    # Extraire les voisins avec leur poids
    neighbors = []
    for edge in edges:
        neighbor_id = edge['target'] if edge['source'] == patient_id else edge['source']
        neighbor_node = next((n for n in graph_data['nodes'] if n['id'] == neighbor_id), None)

        if neighbor_node:
            neighbors.append({
                'id': neighbor_id,
                'weight': edge['weight'],
                'distance': edge['distance'],
                'node': neighbor_node
            })

    # Trier par poids (similarité) décroissant
    neighbors.sort(key=lambda x: x['weight'], reverse=True)

    # Limiter au top k
    neighbors = neighbors[:k]

    return {
        'patient_id': patient_id,
        'neighbors': neighbors
    }


@app.get("/api/prototypes")
async def get_prototypes():
    """Retourne la liste des patients prototypiques"""
    if graph_data is None or 'prototypes' not in graph_data:
        raise HTTPException(status_code=503, detail="Données des prototypes non disponibles")

    prototype_ids = graph_data['prototypes']
    prototypes = [
        n for n in graph_data['nodes']
        if n['id'] in prototype_ids
    ]

    return {
        'count': len(prototypes),
        'prototypes': prototypes
    }


@app.get("/api/atypical")
async def get_atypical():
    """Retourne la liste des patients atypiques"""
    if graph_data is None or 'atypical' not in graph_data:
        raise HTTPException(status_code=503, detail="Données des patients atypiques non disponibles")

    atypical_ids = graph_data['atypical']
    atypical = [
        n for n in graph_data['nodes']
        if n['id'] in atypical_ids
    ]

    return {
        'count': len(atypical),
        'atypical': atypical
    }


@app.get("/api/stats")
async def get_stats():
    """Retourne les statistiques globales du graphe"""
    if graph_data is None:
        raise HTTPException(status_code=503, detail="Données du graphe non disponibles")

    stats = graph_data.get('stats', {})

    # Ajouter des statistiques supplémentaires
    if metrics_data:
        representativeness = np.array(metrics_data['representativeness'])
        atypicality = np.array(metrics_data['atypicality'])

        stats.update({
            'avg_representativeness': float(representativeness.mean()),
            'avg_atypicality': float(atypicality.mean()),
            'n_prototypes': len(metrics_data.get('prototypes', [])),
            'n_atypical': len(metrics_data.get('atypical', []))
        })

    if embeddings is not None:
        stats['embedding_dim'] = int(embeddings.shape[1])

    return stats


@app.post("/api/reload")
async def reload_data():
    """Recharge les données depuis les fichiers (utile après un nouvel entraînement)"""
    load_data()
    return {"message": "Données rechargées avec succès"}


# ========== Lancement du serveur ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lancer l'API du graphe de patients")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    parser.add_argument('--reload', action='store_true', help='Auto-reload')

    args = parser.parse_args()

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )