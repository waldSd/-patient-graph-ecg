"""
Génère des données de démonstration fictives pour tester l'interface
sans avoir à télécharger le dataset PTB-XL ni entraîner le modèle.

ATTENTION : Ces données sont SIMULÉES, pas réelles !
Pour la vraie démo de la thèse, utilisez le dataset PTB-XL.
"""
import os
import json
import numpy as np
from graph_builder import PatientGraphBuilder

print("=" * 80)
print("GÉNÉRATION DE DONNÉES DE DÉMONSTRATION (SIMULÉES)")
print("=" * 80)
print("\n⚠️  ATTENTION : Ces données sont fictives !")
print("Pour la vraie démo, téléchargez PTB-XL et lancez train.py\n")

# Créer les dossiers de sortie
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/embeddings", exist_ok=True)
os.makedirs("output/graph", exist_ok=True)

# Paramètres
n_patients = 200  # Nombre de patients simulés
embedding_dim = 64

print(f"Génération de {n_patients} patients simulés...")

# Générer des embeddings simulés avec structure de clusters
np.random.seed(42)
embeddings = []

# Créer 5 clusters principaux (simule différents profils cardiaques)
n_clusters = 5
cluster_size = n_patients // n_clusters

for cluster_id in range(n_clusters):
    # Centre du cluster
    cluster_center = np.random.randn(embedding_dim) * 2

    # Générer des patients autour du centre
    for i in range(cluster_size):
        # Patient normal du cluster
        if i < cluster_size - 2:
            noise = np.random.randn(embedding_dim) * 0.5
            embedding = cluster_center + noise
        # Dernier patient = prototype (très proche du centre)
        elif i == cluster_size - 2:
            noise = np.random.randn(embedding_dim) * 0.1
            embedding = cluster_center + noise
        # Avant-dernier = patient un peu atypique
        else:
            noise = np.random.randn(embedding_dim) * 2
            embedding = cluster_center + noise

        embeddings.append(embedding)

# Ajouter quelques patients vraiment atypiques (outliers)
n_outliers = n_patients - len(embeddings)
for _ in range(n_outliers):
    outlier = np.random.randn(embedding_dim) * 5  # Très éloignés
    embeddings.append(outlier)

embeddings = np.array(embeddings)

# Normaliser
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"✓ Embeddings générés : shape {embeddings.shape}")

# Sauvegarder les embeddings
np.save("output/embeddings/embeddings.npy", embeddings)
print("✓ Embeddings sauvegardés")

# Créer des métadonnées fictives
import pandas as pd

metadata = pd.DataFrame({
    'patient_id': range(n_patients),
    'age': np.random.randint(20, 90, n_patients),
    'sex': np.random.randint(0, 2, n_patients),
    'ecg_id': range(n_patients)
})

metadata.to_csv("output/embeddings/metadata.csv", index=False)
print("✓ Métadonnées sauvegardées")

# Construire le graphe
print("\nConstruction du graphe...")
builder = PatientGraphBuilder(embeddings=embeddings, metadata=metadata)

# Calculer similarité
builder.compute_similarity_matrix(metric='cosine')

# Construire graphe K-NN
builder.build_knn_graph(k=10)

# Calculer métriques
representativeness = builder.compute_representativeness()
atypicality = builder.compute_atypicality()

# Identifier prototypes et atypiques
n_special = min(10, n_patients // 10)
prototypes = builder.identify_prototypes(n_prototypes=n_special)
atypical_patients = builder.identify_atypical(n_atypical=n_special)

# Exporter les données du graphe
print("\nExport des résultats...")
graph_data = builder.export_graph_data()

# Ajouter les informations supplémentaires
graph_data['prototypes'] = prototypes
graph_data['atypical'] = atypical_patients

# Sauvegarder le graphe (JSON)
with open("output/graph/graph_data.json", 'w') as f:
    json.dump(graph_data, f, indent=2)

print("✓ graph_data.json créé")

# Sauvegarder le graphe (GraphML)
builder.save_graph("output/graph/patient_graph.graphml")

# Sauvegarder les métriques
metrics = {
    'representativeness': representativeness.tolist(),
    'atypicality': atypicality.tolist(),
    'prototypes': prototypes,
    'atypical': atypical_patients
}

with open("output/graph/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ metrics.json créé")

# Afficher le résumé
print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)

print(f"""
✅ Données de démonstration générées avec succès !

Fichiers créés :
  - output/embeddings/embeddings.npy
  - output/embeddings/metadata.csv
  - output/graph/graph_data.json
  - output/graph/patient_graph.graphml
  - output/graph/metrics.json

Statistiques :
  - Patients : {n_patients}
  - Dimension embeddings : {embedding_dim}
  - Connexions : {graph_data['stats']['n_connections']}
  - Prototypes : {len(prototypes)}
  - Atypiques : {len(atypical_patients)}

Vous pouvez maintenant :
  1. Lancer l'API : python api.py
  2. Lancer le frontend : cd ../empathy-health-main && npm run dev
  3. Accéder à : http://localhost:5173/services/graphe-patients

⚠️  RAPPEL : Ces données sont SIMULÉES
Pour la vraie démo de thèse :
  1. Téléchargez PTB-XL : https://physionet.org/content/ptb-xl/1.0.3/
  2. Lancez : python train.py --max_samples 5000
""")

print("=" * 80)