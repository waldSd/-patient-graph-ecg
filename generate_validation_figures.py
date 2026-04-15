"""
Génère les figures de validation train/test pour le rapport LaTeX
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

print("="*80)
print("GÉNÉRATION DES FIGURES DE VALIDATION TRAIN/TEST")
print("="*80)

# ========== CHARGER LES DONNÉES ==========
print("\nChargement des données de validation...")

# Embeddings train et test
train_embeddings = np.load("output_validated/embeddings/train_embeddings.npy")
test_embeddings = np.load("output_validated/embeddings/test_embeddings.npy")

# Métadonnées
train_metadata = pd.read_csv("output_validated/embeddings/train_metadata.csv")
test_metadata = pd.read_csv("output_validated/embeddings/test_metadata.csv")

# Métriques
with open("output_validated/graph/train_metrics.json", 'r') as f:
    train_metrics = json.load(f)

with open("output_validated/graph/test_metrics.json", 'r') as f:
    test_metrics = json.load(f)

with open("output_validated/graph/train_test_comparison.json", 'r') as f:
    comparison = json.load(f)

# Historique d'entraînement
with open("output_validated/models/ecg_autoencoder_history.json", 'r') as f:
    history = json.load(f)

print(f"✓ Données chargées")
print(f"  - Train: {len(train_embeddings)} patients")
print(f"  - Test: {len(test_embeddings)} patients")

# ========== FIGURE 1: COMPARAISON TRAIN VS TEST - MÉTRIQUES ==========
print("\nGénération Figure 1: Comparaison train vs test...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Représentativité
categories = ['Train', 'Test']
repr_values = [
    comparison['train']['avg_representativeness'],
    comparison['test']['avg_representativeness']
]

bars1 = ax1.bar(categories, repr_values, color=['#3498db', '#e74c3c'],
                edgecolor='black', alpha=0.8, width=0.6)
ax1.set_ylabel('Score moyen de représentativité', fontsize=12)
ax1.set_title('Représentativité : Train vs Test', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars1, repr_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Atypicité
atyp_values = [
    comparison['train']['avg_atypicality'],
    comparison['test']['avg_atypicality']
]

bars2 = ax2.bar(categories, atyp_values, color=['#3498db', '#e74c3c'],
                edgecolor='black', alpha=0.8, width=0.6)
ax2.set_ylabel('Score moyen d\'atypicité', fontsize=12)
ax2.set_title('Atypicité : Train vs Test', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 0.15])
ax2.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars2, atyp_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "07_train_test_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 1 sauvegardée")

# ========== FIGURE 2: DISTRIBUTIONS COMPARÉES ==========
print("Génération Figure 2: Distributions comparées...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Représentativité - Train
train_repr = np.array(train_metrics['representativeness'])
axes[0, 0].hist(train_repr, bins=40, edgecolor='black', alpha=0.7, color='#3498db')
axes[0, 0].set_xlabel('Score de représentativité', fontsize=11)
axes[0, 0].set_ylabel('Nombre de patients', fontsize=11)
axes[0, 0].set_title('Représentativité - Train Set (n=800)', fontsize=12, fontweight='bold')
axes[0, 0].axvline(train_repr.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {train_repr.mean():.4f}')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Représentativité - Test
test_repr = np.array(test_metrics['representativeness'])
axes[0, 1].hist(test_repr, bins=40, edgecolor='black', alpha=0.7, color='#e74c3c')
axes[0, 1].set_xlabel('Score de représentativité', fontsize=11)
axes[0, 1].set_ylabel('Nombre de patients', fontsize=11)
axes[0, 1].set_title('Représentativité - Test Set (n=200)', fontsize=12, fontweight='bold')
axes[0, 1].axvline(test_repr.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {test_repr.mean():.4f}')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Atypicité - Train
train_atyp = np.array(train_metrics['atypicality'])
axes[1, 0].hist(train_atyp, bins=40, edgecolor='black', alpha=0.7, color='#3498db')
axes[1, 0].set_xlabel('Score d\'atypicité', fontsize=11)
axes[1, 0].set_ylabel('Nombre de patients', fontsize=11)
axes[1, 0].set_title('Atypicité - Train Set (n=800)', fontsize=12, fontweight='bold')
axes[1, 0].axvline(train_atyp.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {train_atyp.mean():.4f}')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Atypicité - Test
test_atyp = np.array(test_metrics['atypicality'])
axes[1, 1].hist(test_atyp, bins=40, edgecolor='black', alpha=0.7, color='#e74c3c')
axes[1, 1].set_xlabel('Score d\'atypicité', fontsize=11)
axes[1, 1].set_ylabel('Nombre de patients', fontsize=11)
axes[1, 1].set_title('Atypicité - Test Set (n=200)', fontsize=12, fontweight='bold')
axes[1, 1].axvline(test_atyp.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Moyenne: {test_atyp.mean():.4f}')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "08_distributions_train_test.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 2 sauvegardée")

# ========== FIGURE 3: UMAP TRAIN + TEST ==========
print("Génération Figure 3: Visualisation UMAP train+test...")

try:
    from umap import UMAP

    # Combiner embeddings pour UMAP
    all_embeddings = np.vstack([train_embeddings, test_embeddings])

    print("  Réduction UMAP en cours...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
    embeddings_2d = reducer.fit_transform(all_embeddings)

    # Séparer train et test
    train_2d = embeddings_2d[:len(train_embeddings)]
    test_2d = embeddings_2d[len(train_embeddings):]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Train en bleu
    ax.scatter(train_2d[:, 0], train_2d[:, 1],
              c='#3498db', s=60, alpha=0.6, label='Train (n=800)', edgecolors='none')

    # Test en rouge
    ax.scatter(test_2d[:, 0], test_2d[:, 1],
              c='#e74c3c', s=60, alpha=0.6, label='Test (n=200)', edgecolors='none')

    # Prototypes train en étoile verte
    proto_train_idx = train_metrics['prototypes']
    ax.scatter(train_2d[proto_train_idx, 0], train_2d[proto_train_idx, 1],
              c='green', s=250, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes Train (n={len(proto_train_idx)})', zorder=10)

    # Atypiques train en triangle violet
    atyp_train_idx = train_metrics['atypical']
    ax.scatter(train_2d[atyp_train_idx, 0], train_2d[atyp_train_idx, 1],
              c='purple', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques Train (n={len(atyp_train_idx)})', zorder=10)

    # Prototypes test en étoile jaune
    proto_test_idx = test_metrics['prototypes']
    ax.scatter(test_2d[proto_test_idx, 0], test_2d[proto_test_idx, 1],
              c='yellow', s=250, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes Test (n={len(proto_test_idx)})', zorder=10)

    # Atypiques test en triangle orange
    atyp_test_idx = test_metrics['atypical']
    ax.scatter(test_2d[atyp_test_idx, 0], test_2d[atyp_test_idx, 1],
              c='orange', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques Test (n={len(atyp_test_idx)})', zorder=10)

    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.set_title('Carte des patients : Train vs Test avec prototypes et atypiques (UMAP)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "09_umap_train_test.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 3 sauvegardée (UMAP)")

except Exception as e:
    print(f"⚠️  UMAP failed: {e}, using PCA instead...")
    from sklearn.decomposition import PCA

    all_embeddings = np.vstack([train_embeddings, test_embeddings])
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    train_2d = embeddings_2d[:len(train_embeddings)]
    test_2d = embeddings_2d[len(train_embeddings):]

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(train_2d[:, 0], train_2d[:, 1],
              c='#3498db', s=60, alpha=0.6, label='Train (n=800)', edgecolors='none')

    ax.scatter(test_2d[:, 0], test_2d[:, 1],
              c='#e74c3c', s=60, alpha=0.6, label='Test (n=200)', edgecolors='none')

    proto_train_idx = train_metrics['prototypes']
    ax.scatter(train_2d[proto_train_idx, 0], train_2d[proto_train_idx, 1],
              c='green', s=250, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes Train (n={len(proto_train_idx)})', zorder=10)

    atyp_train_idx = train_metrics['atypical']
    ax.scatter(train_2d[atyp_train_idx, 0], train_2d[atyp_train_idx, 1],
              c='purple', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques Train (n={len(atyp_train_idx)})', zorder=10)

    proto_test_idx = test_metrics['prototypes']
    ax.scatter(test_2d[proto_test_idx, 0], test_2d[proto_test_idx, 1],
              c='yellow', s=250, marker='*', edgecolors='black', linewidth=1.5,
              label=f'Prototypes Test (n={len(proto_test_idx)})', zorder=10)

    atyp_test_idx = test_metrics['atypical']
    ax.scatter(test_2d[atyp_test_idx, 0], test_2d[atyp_test_idx, 1],
              c='orange', s=150, marker='^', edgecolors='black', linewidth=1.5,
              label=f'Atypiques Test (n={len(atyp_test_idx)})', zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
    ax.set_title('Carte des patients : Train vs Test avec prototypes et atypiques (PCA)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "09_umap_train_test.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 3 sauvegardée (PCA)")

# ========== FIGURE 4: COURBES D'ENTRAÎNEMENT AVEC VALIDATION ==========
print("Génération Figure 4: Courbes d'entraînement...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(history['loss']) + 1)

# Loss
ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Test Loss (Validation)', marker='s', markersize=4)
ax1.set_xlabel('Époque', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Évolution de la Loss (Train vs Test)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Ajouter annotation de l'early stopping si présent
if len(epochs) < 50:
    ax1.axvline(len(epochs), color='green', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Early stopping (epoch {len(epochs)})')

# MAE
ax2.plot(epochs, history['mae'], 'b-', linewidth=2, label='Train MAE', marker='o', markersize=4)
ax2.plot(epochs, history['val_mae'], 'r-', linewidth=2, label='Test MAE (Validation)', marker='s', markersize=4)
ax2.set_xlabel('Époque', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Évolution du MAE (Train vs Test)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "10_training_validation_curves.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure 4 sauvegardée")

# ========== GÉNÉRER LES STATISTIQUES DE VALIDATION ==========
print("\nGénération des statistiques de validation...")

validation_stats = {
    "train_set": {
        "n_patients": len(train_embeddings),
        "avg_representativeness": float(train_repr.mean()),
        "std_representativeness": float(train_repr.std()),
        "avg_atypicality": float(train_atyp.mean()),
        "std_atypicality": float(train_atyp.std()),
        "n_prototypes": len(train_metrics['prototypes']),
        "n_atypical": len(train_metrics['atypical'])
    },
    "test_set": {
        "n_patients": len(test_embeddings),
        "avg_representativeness": float(test_repr.mean()),
        "std_representativeness": float(test_repr.std()),
        "avg_atypicality": float(test_atyp.mean()),
        "std_atypicality": float(test_atyp.std()),
        "n_prototypes": len(test_metrics['prototypes']),
        "n_atypical": len(test_metrics['atypical'])
    },
    "comparison": {
        "repr_difference": float(abs(train_repr.mean() - test_repr.mean())),
        "repr_difference_pct": float(abs(train_repr.mean() - test_repr.mean()) / train_repr.mean() * 100),
        "atyp_difference": float(abs(train_atyp.mean() - test_atyp.mean())),
        "atyp_difference_pct": float(abs(train_atyp.mean() - test_atyp.mean()) / train_atyp.mean() * 100)
    },
    "training": {
        "final_train_loss": float(history['loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
        "final_train_mae": float(history['mae'][-1]),
        "final_val_mae": float(history['val_mae'][-1]),
        "n_epochs": len(history['loss']),
        "early_stopped": len(history['loss']) < 50
    }
}

with open("rapport/validation_statistics.json", 'w') as f:
    json.dump(validation_stats, f, indent=2)

print(f"✓ Statistiques de validation sauvegardées : rapport/validation_statistics.json")

print("\n" + "="*80)
print("TOUTES LES FIGURES DE VALIDATION SONT GÉNÉRÉES !")
print("="*80)
print(f"\nFigures disponibles dans : {output_dir}/")
print("  - 07_train_test_comparison.png : Comparaison des métriques moyennes")
print("  - 08_distributions_train_test.png : Distributions des scores")
print("  - 09_umap_train_test.png : Visualisation 2D avec prototypes/atypiques")
print("  - 10_training_validation_curves.png : Courbes d'entraînement")
print(f"\nStatistiques disponibles dans : rapport/validation_statistics.json")

# Afficher un résumé
print("\n" + "="*80)
print("RÉSUMÉ DE LA VALIDATION")
print("="*80)
print(f"\nTrain set (n={validation_stats['train_set']['n_patients']}):")
print(f"  Représentativité: {validation_stats['train_set']['avg_representativeness']:.4f} ± {validation_stats['train_set']['std_representativeness']:.4f}")
print(f"  Atypicité: {validation_stats['train_set']['avg_atypicality']:.4f} ± {validation_stats['train_set']['std_atypicality']:.4f}")

print(f"\nTest set (n={validation_stats['test_set']['n_patients']}):")
print(f"  Représentativité: {validation_stats['test_set']['avg_representativeness']:.4f} ± {validation_stats['test_set']['std_representativeness']:.4f}")
print(f"  Atypicité: {validation_stats['test_set']['avg_atypicality']:.4f} ± {validation_stats['test_set']['std_atypicality']:.4f}")

print(f"\nDifférences train-test:")
print(f"  Représentativité: {validation_stats['comparison']['repr_difference']:.4f} ({validation_stats['comparison']['repr_difference_pct']:.2f}%)")
print(f"  Atypicité: {validation_stats['comparison']['atyp_difference']:.4f} ({validation_stats['comparison']['atyp_difference_pct']:.2f}%)")

print(f"\nEntraînement:")
print(f"  Époques: {validation_stats['training']['n_epochs']}/50")
print(f"  Early stopping: {'Oui' if validation_stats['training']['early_stopped'] else 'Non'}")
print(f"  Final train loss: {validation_stats['training']['final_train_loss']:.6f}")
print(f"  Final val loss: {validation_stats['training']['final_val_loss']:.6f}")

print("\n✓ Prêt pour la mise à jour du rapport LaTeX")