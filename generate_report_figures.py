"""
Génère toutes les figures et statistiques pour le rapport LaTeX
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# Configuration matplotlib
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Créer le dossier rapport/figures
output_dir = Path("rapport/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ========== CHARGER LES DONNÉES ==========
print("Chargement des données...")

# Métadonnées du dataset
metadata = pd.read_csv("output/embeddings/metadata.csv")

# Embeddings
embeddings = np.load("output/embeddings/embeddings.npy")

# Métriques
with open("output/graph/metrics.json", 'r') as f:
    metrics = json.load(f)

# Historique d'entraînement
with open("output/models/ecg_autoencoder_history.json", 'r') as f:
    history = json.load(f)

# Graphe data
with open("output/graph/graph_data.json", 'r') as f:
    graph_data = json.load(f)

print(f"✓ Données chargées")
print(f"  - {len(metadata)} patients")
print(f"  - {len(embeddings)} embeddings")
print(f"  - {len(metrics['prototypes'])} prototypes")
print(f"  - {len(metrics['atypical'])} atypiques")

# ========== FIGURE 1: DÉMOGRAPHIE ==========
print("\nGénération Figure 1: Démographie...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Distribution de l'âge
ax1.hist(metadata['age'], bins=30, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Âge (années)', fontsize=12)
ax1.set_ylabel('Nombre de patients', fontsize=12)
ax1.set_title('Distribution de l\'âge des patients', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(metadata['age'].mean(), color='red', linestyle='--',
            label=f'Moyenne: {metadata["age"].mean():.1f} ans')
ax1.legend()

# Distribution par sexe
sex_counts = metadata['sex'].value_counts()
ax2.bar(['Hommes', 'Femmes'], [sex_counts.get(0, 0), sex_counts.get(1, 0)],
        color=['#3498db', '#e74c3c'], edgecolor='black')
ax2.set_ylabel('Nombre de patients', fontsize=12)
ax2.set_title('Répartition par sexe', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "01_demographics.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 1 sauvegardée")

# ========== FIGURE 2: EXEMPLE D'ECG ==========
print("Génération Figure 2: Exemple d'ECG...")

# Charger un ECG exemple
from data_loader import PTBXLDataLoader
loader = PTBXLDataLoader(data_dir='./ptb-xl-data')
example_signal = loader.load_signal(metadata.iloc[0]['filename_lr'])

if example_signal is not None:
    fig, axes = plt.subplots(12, 1, figsize=(14, 10), sharex=True)
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(example_signal[:, i], linewidth=0.8, color='#2c3e50')
        ax.set_ylabel(lead_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)

    axes[-1].set_xlabel('Temps (échantillons à 100 Hz)', fontsize=12)
    fig.suptitle('Exemple d\'ECG 12-dérivations du dataset PTB-XL',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_dir / "02_example_ecg.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 sauvegardée")

# ========== FIGURE 3: COURBES D'ENTRAÎNEMENT ==========
print("Génération Figure 3: Courbes d'entraînement...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(history['loss']) + 1)

# Loss
ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss')
ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Époque', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# MAE
ax2.plot(epochs, history['mae'], 'b-', linewidth=2, label='Train MAE')
ax2.plot(epochs, history['val_mae'], 'r-', linewidth=2, label='Validation MAE')
ax2.set_xlabel('Époque', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Évolution du MAE', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "03_training_curves.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 3 sauvegardée")

# ========== FIGURE 4: DISTRIBUTION DES SIMILARITÉS ==========
print("Génération Figure 4: Distribution des similarités...")

# Extraire les similarités (weights) des arêtes
weights = [edge['weight'] for edge in graph_data['edges']]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='#27ae60')
ax.set_xlabel('Similarité cosinus', fontsize=12)
ax.set_ylabel('Nombre de connexions', fontsize=12)
ax.set_title('Distribution des similarités entre patients connectés',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(np.mean(weights), color='red', linestyle='--',
          label=f'Moyenne: {np.mean(weights):.4f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / "04_similarity_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 4 sauvegardée")

# ========== FIGURE 5: DISTRIBUTION DES MÉTRIQUES ==========
print("Génération Figure 5: Distribution des métriques...")

representativeness = np.array(metrics['representativeness'])
atypicality = np.array(metrics['atypicality'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Représentativité
ax1.hist(representativeness, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
ax1.set_xlabel('Score de représentativité', fontsize=12)
ax1.set_ylabel('Nombre de patients', fontsize=12)
ax1.set_title('Distribution de la représentativité', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(representativeness.mean(), color='red', linestyle='--',
           label=f'Moyenne: {representativeness.mean():.4f}')
ax1.legend()

# Atypicité
ax2.hist(atypicality, bins=30, edgecolor='black', alpha=0.7, color='#e74c3c')
ax2.set_xlabel('Score d\'atypicité', fontsize=12)
ax2.set_ylabel('Nombre de patients', fontsize=12)
ax2.set_title('Distribution de l\'atypicité', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(atypicality.mean(), color='red', linestyle='--',
           label=f'Moyenne: {atypicality.mean():.4f}')
ax2.legend()

plt.tight_layout()
plt.savefig(output_dir / "05_metrics_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 5 sauvegardée")

# ========== FIGURE 6: PROTOTYPES VS ATYPIQUES (2D) ==========
print("Génération Figure 6: Prototypes vs Atypiques...")

# Réduction de dimension UMAP
try:
    from umap import UMAP

    print("  Réduction UMAP en cours...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
    embeddings_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Tous les patients en gris
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
              c='lightgray', s=50, alpha=0.5, label='Patients normaux')

    # Prototypes en vert
    proto_idx = metrics['prototypes']
    ax.scatter(embeddings_2d[proto_idx, 0], embeddings_2d[proto_idx, 1],
              c='green', s=200, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes (n={len(proto_idx)})', zorder=10)

    # Atypiques en rouge
    atyp_idx = metrics['atypical']
    ax.scatter(embeddings_2d[atyp_idx, 0], embeddings_2d[atyp_idx, 1],
              c='red', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques (n={len(atyp_idx)})', zorder=10)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Carte des patients : Prototypes vs Atypiques (UMAP)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "06_prototypes_vs_atypical.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 6 sauvegardée (UMAP)")

except Exception as e:
    print(f"⚠️  UMAP failed: {e}, using PCA instead...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
              c='lightgray', s=50, alpha=0.5, label='Patients normaux')

    proto_idx = metrics['prototypes']
    ax.scatter(embeddings_2d[proto_idx, 0], embeddings_2d[proto_idx, 1],
              c='green', s=200, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes (n={len(proto_idx)})', zorder=10)

    atyp_idx = metrics['atypical']
    ax.scatter(embeddings_2d[atyp_idx, 0], embeddings_2d[atyp_idx, 1],
              c='red', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques (n={len(atyp_idx)})', zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Carte des patients : Prototypes vs Atypiques (PCA)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "06_prototypes_vs_atypical.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 6 sauvegardée (PCA)")

# ========== GÉNÉRER LES STATISTIQUES POUR LATEX ==========
print("\nGénération des statistiques pour LaTeX...")

statistics = {
    "dataset": {
        "n_patients": len(metadata),
        "sampling_rate": 100,
        "signal_duration": 10,
        "n_leads": 12,
        "mean_age": float(metadata['age'].mean()),
        "median_age": float(metadata['age'].median()),
        "n_male": int((metadata['sex'] == 0).sum()),
        "n_female": int((metadata['sex'] == 1).sum())
    },
    "model": {
        "embedding_dim": embeddings.shape[1],
        "compression_ratio": (1000 * 12) / embeddings.shape[1],
        "final_train_loss": float(history['loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
        "final_train_mae": float(history['mae'][-1]),
        "final_val_mae": float(history['val_mae'][-1]),
        "n_epochs": len(history['loss'])
    },
    "graph": {
        "n_nodes": graph_data['stats']['n_patients'],
        "n_edges": graph_data['stats']['n_connections'],
        "avg_degree": graph_data['stats']['avg_degree'],
        "k_neighbors": 10,
        "avg_similarity": float(np.mean(weights))
    },
    "metrics": {
        "avg_representativeness": float(representativeness.mean()),
        "min_representativeness": float(representativeness.min()),
        "max_representativeness": float(representativeness.max()),
        "avg_atypicality": float(atypicality.mean()),
        "min_atypicality": float(atypicality.min()),
        "max_atypicality": float(atypicality.max()),
        "n_prototypes": len(metrics['prototypes']),
        "n_atypical": len(metrics['atypical'])
    }
}

with open("rapport/statistics.json", 'w') as f:
    json.dump(statistics, f, indent=2)

print(f"✓ Statistiques sauvegardées : rapport/statistics.json")

print("\n" + "="*80)
print("TOUTES LES FIGURES ET STATISTIQUES SONT GÉNÉRÉES !")
print("="*80)
print(f"\nFigures disponibles dans : {output_dir}/")
print(f"Statistiques disponibles dans : rapport/statistics.json")
print("\nProchaine étape : Générer le rapport LaTeX avec:")
print("  cd rapport/")
print("  python generate_latex_report.py")