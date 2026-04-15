"""
Pipeline d'entraînement avec validation scientifique rigoureuse train/test
"""
import argparse
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from data_loader import PTBXLDataLoader
from autoencoder import ECGAutoencoder
from graph_builder import PatientGraphBuilder


def evaluate_new_patients(
    train_graph_builder: PatientGraphBuilder,
    test_embeddings: np.ndarray,
    test_metadata=None
):
    """
    Évalue de nouveaux patients (test set) par rapport au graphe train

    Args:
        train_graph_builder: GraphBuilder construit sur train set
        test_embeddings: Embeddings des patients test
        test_metadata: Métadonnées des patients test

    Returns:
        dict: Métriques pour les patients test
    """
    print("\n" + "="*80)
    print("ÉVALUATION DES NOUVEAUX PATIENTS (TEST SET)")
    print("="*80)

    n_test = len(test_embeddings)
    train_embeddings = train_graph_builder.embeddings

    # Calculer la similarité entre chaque patient test et tous les patients train
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\nCalcul des similarités test → train...")
    test_to_train_similarity = cosine_similarity(test_embeddings, train_embeddings)

    print(f"✓ Matrice de similarité calculée: {test_to_train_similarity.shape}")

    # Pour chaque patient test, trouver ses k plus proches voisins dans le train
    k = 20  # Même k que pour les métriques

    test_representativeness = np.zeros(n_test)
    test_atypicality = np.zeros(n_test)
    test_nearest_neighbors = []

    for i in range(n_test):
        # Similarités de ce patient test avec tous les patients train
        similarities = test_to_train_similarity[i]

        # Trouver les k plus proches voisins
        top_k_idx = np.argsort(similarities)[-k:]
        top_k_similarities = similarities[top_k_idx]

        # Représentativité = moyenne des similarités aux k voisins
        test_representativeness[i] = top_k_similarities.mean()

        # Atypicité = moyenne des distances aux k voisins
        test_atypicality[i] = (1 - top_k_similarities).mean()

        # Stocker les voisins
        test_nearest_neighbors.append({
            'patient_idx': i,
            'neighbor_indices': top_k_idx.tolist(),
            'similarities': top_k_similarities.tolist()
        })

    # Normaliser entre 0 et 1
    test_representativeness = (test_representativeness - test_representativeness.min()) / \
                              (test_representativeness.max() - test_representativeness.min() + 1e-8)
    test_atypicality = (test_atypicality - test_atypicality.min()) / \
                      (test_atypicality.max() - test_atypicality.min() + 1e-8)

    print(f"\n✓ Métriques calculées pour {n_test} patients test")
    print(f"  Représentativité moyenne: {test_representativeness.mean():.4f}")
    print(f"  Atypicité moyenne: {test_atypicality.mean():.4f}")

    # Identifier prototypes et atypiques dans le test set
    n_special = min(10, n_test // 10)
    test_prototypes = np.argsort(test_representativeness)[-n_special:][::-1].tolist()
    test_atypical = np.argsort(test_atypicality)[-n_special:][::-1].tolist()

    print(f"\n✓ {n_special} patients prototypiques identifiés (test):")
    for idx in test_prototypes[:5]:
        print(f"  Patient test {idx} - score: {test_representativeness[idx]:.4f}")

    print(f"\n✓ {n_special} patients atypiques identifiés (test):")
    for idx in test_atypical[:5]:
        print(f"  Patient test {idx} - score: {test_atypicality[idx]:.4f}")

    return {
        'representativeness': test_representativeness.tolist(),
        'atypicality': test_atypicality.tolist(),
        'prototypes': test_prototypes,
        'atypical': test_atypical,
        'nearest_neighbors': test_nearest_neighbors,
        'stats': {
            'avg_representativeness': float(test_representativeness.mean()),
            'min_representativeness': float(test_representativeness.min()),
            'max_representativeness': float(test_representativeness.max()),
            'avg_atypicality': float(test_atypicality.mean()),
            'min_atypicality': float(test_atypicality.min()),
            'max_atypicality': float(test_atypicality.max())
        }
    }


def main(args):
    print("="*80)
    print("PIPELINE AVEC VALIDATION SCIENTIFIQUE TRAIN/TEST")
    print("="*80)
    print(f"\nDémarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========== 1. CHARGEMENT DES DONNÉES ==========
    print("="*80)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES PTB-XL")
    print("="*80)

    loader = PTBXLDataLoader(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate
    )

    # Charger les métadonnées
    metadata = loader.load_metadata()

    # Charger les signaux
    signals, valid_metadata = loader.load_all_signals(max_samples=args.max_samples)

    # Prétraiter
    signals = loader.preprocess_signals(signals)

    # Split RIGOUREUX train/test (80/20)
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(signals))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train = signals[train_idx]
    X_test = signals[test_idx]
    metadata_train = valid_metadata.iloc[train_idx].reset_index(drop=True)
    metadata_test = valid_metadata.iloc[test_idx].reset_index(drop=True)

    print(f"✓ Split train/test RIGOUREUX : {len(X_train)} / {len(X_test)}")
    print(f"\n✓ Données prêtes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # ========== 2. ENTRAÎNEMENT AUTO-ENCODEUR ==========
    print("\n" + "="*80)
    print("ÉTAPE 2: ENTRAÎNEMENT DE L'AUTO-ENCODEUR")
    print("="*80)

    # Créer ou charger le modèle
    model_path = os.path.join(args.output_dir, "models", "ecg_autoencoder")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    input_shape = (X_train.shape[1], X_train.shape[2])

    if os.path.exists(f"{model_path}_autoencoder.keras") and not args.retrain:
        print(f"\n⚠️  Un modèle existe déjà à {model_path}")
        print("Chargement du modèle existant (utilisez --retrain pour ré-entraîner)")
        ae = ECGAutoencoder(input_shape=input_shape, embedding_dim=args.embedding_dim)
        ae.load(model_path)
    else:
        print("\nCréation d'un nouveau modèle...")
        ae = ECGAutoencoder(input_shape=input_shape, embedding_dim=args.embedding_dim)
        ae.compile_model(learning_rate=0.001)

        # Entraîner UNIQUEMENT sur train
        history = ae.train(
            X_train=X_train,
            X_val=X_test,  # Validation sur test
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Sauvegarder
        ae.save(model_path)

        # Sauvegarder l'historique
        with open(f"{model_path}_history.json", 'w') as f:
            json.dump(history.history, f)

    # ========== 3. EXTRACTION DES EMBEDDINGS ==========
    print("\n" + "="*80)
    print("ÉTAPE 3: EXTRACTION DES EMBEDDINGS")
    print("="*80)

    train_embeddings = ae.get_embeddings(X_train)
    test_embeddings = ae.get_embeddings(X_test)

    print(f"✓ Embeddings train extraits : {train_embeddings.shape}")
    print(f"✓ Embeddings test extraits : {test_embeddings.shape}")

    # Sauvegarder
    embeddings_path = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    np.save(os.path.join(embeddings_path, "train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(embeddings_path, "test_embeddings.npy"), test_embeddings)
    metadata_train.to_csv(os.path.join(embeddings_path, "train_metadata.csv"), index=False)
    metadata_test.to_csv(os.path.join(embeddings_path, "test_metadata.csv"), index=False)

    print(f"✓ Embeddings sauvegardés : {embeddings_path}")

    # ========== 4. CONSTRUCTION DU GRAPHE (TRAIN UNIQUEMENT) ==========
    print("\n" + "="*80)
    print("ÉTAPE 4: CONSTRUCTION DU GRAPHE (TRAIN SET UNIQUEMENT)")
    print("="*80)

    train_builder = PatientGraphBuilder(
        embeddings=train_embeddings,
        metadata=metadata_train
    )

    # Matrice de similarité
    train_builder.compute_similarity_matrix(metric='cosine')

    # Graphe K-NN
    train_builder.build_knn_graph(k=args.k_neighbors)

    # ========== 5. MÉTRIQUES SUR TRAIN SET ==========
    print("\n" + "="*80)
    print("ÉTAPE 5: CALCUL DES MÉTRIQUES (TRAIN SET)")
    print("="*80)

    train_representativeness = train_builder.compute_representativeness()
    train_atypicality = train_builder.compute_atypicality()

    n_special = min(10, len(train_embeddings) // 10)
    train_prototypes = train_builder.identify_prototypes(n_prototypes=n_special)
    train_atypical_patients = train_builder.identify_atypical(n_atypical=n_special)

    # ========== 6. ÉVALUATION SUR TEST SET ==========
    print("\n" + "="*80)
    print("ÉTAPE 6: ÉVALUATION SUR TEST SET (NOUVEAUX PATIENTS)")
    print("="*80)

    test_metrics = evaluate_new_patients(
        train_graph_builder=train_builder,
        test_embeddings=test_embeddings,
        test_metadata=metadata_test
    )

    # ========== 7. EXPORT DES RÉSULTATS ==========
    print("\n" + "="*80)
    print("ÉTAPE 7: EXPORT DES RÉSULTATS")
    print("="*80)

    # Exporter le graphe train
    graph_data = train_builder.export_graph_data()
    graph_data['prototypes'] = train_prototypes
    graph_data['atypical'] = train_atypical_patients

    graph_path = os.path.join(args.output_dir, "graph")
    os.makedirs(graph_path, exist_ok=True)

    with open(os.path.join(graph_path, "train_graph_data.json"), 'w') as f:
        json.dump(graph_data, f, indent=2)

    train_builder.save_graph(os.path.join(graph_path, "train_patient_graph.graphml"))

    # Sauvegarder les métriques train
    train_metrics_data = {
        'representativeness': train_representativeness.tolist(),
        'atypicality': train_atypicality.tolist(),
        'prototypes': train_prototypes,
        'atypical': train_atypical_patients,
        'stats': {
            'avg_representativeness': float(train_representativeness.mean()),
            'min_representativeness': float(train_representativeness.min()),
            'max_representativeness': float(train_representativeness.max()),
            'avg_atypicality': float(train_atypicality.mean()),
            'min_atypicality': float(train_atypicality.min()),
            'max_atypicality': float(train_atypicality.max())
        }
    }

    with open(os.path.join(graph_path, "train_metrics.json"), 'w') as f:
        json.dump(train_metrics_data, f, indent=2)

    # Sauvegarder les métriques test
    with open(os.path.join(graph_path, "test_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"✓ Résultats sauvegardés dans : {args.output_dir}")

    # ========== 8. COMPARAISON TRAIN VS TEST ==========
    print("\n" + "="*80)
    print("COMPARAISON TRAIN VS TEST")
    print("="*80)

    print(f"\n📊 REPRÉSENTATIVITÉ:")
    print(f"  Train - Moyenne: {train_metrics_data['stats']['avg_representativeness']:.4f}")
    print(f"  Test  - Moyenne: {test_metrics['stats']['avg_representativeness']:.4f}")
    print(f"  Différence: {abs(train_metrics_data['stats']['avg_representativeness'] - test_metrics['stats']['avg_representativeness']):.4f}")

    print(f"\n📊 ATYPICITÉ:")
    print(f"  Train - Moyenne: {train_metrics_data['stats']['avg_atypicality']:.4f}")
    print(f"  Test  - Moyenne: {test_metrics['stats']['avg_atypicality']:.4f}")
    print(f"  Différence: {abs(train_metrics_data['stats']['avg_atypicality'] - test_metrics['stats']['avg_atypicality']):.4f}")

    # Sauvegarder la comparaison
    comparison = {
        'train': train_metrics_data['stats'],
        'test': test_metrics['stats'],
        'differences': {
            'representativeness': abs(train_metrics_data['stats']['avg_representativeness'] - test_metrics['stats']['avg_representativeness']),
            'atypicality': abs(train_metrics_data['stats']['avg_atypicality'] - test_metrics['stats']['avg_atypicality'])
        }
    }

    with open(os.path.join(graph_path, "train_test_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*80)
    print("RÉSUMÉ FINAL")
    print("="*80)

    print(f"\nDonnées:")
    print(f"  - Patients train: {len(train_embeddings)}")
    print(f"  - Patients test: {len(test_embeddings)}")
    print(f"  - Dimension des embeddings: {args.embedding_dim}")

    print(f"\nGraphe (train only):")
    print(f"  - Nœuds: {graph_data['stats']['n_patients']}")
    print(f"  - Arêtes: {graph_data['stats']['n_connections']}")
    print(f"  - Degré moyen: {graph_data['stats']['avg_degree']:.2f}")

    print(f"\nPatients spéciaux:")
    print(f"  - Prototypes train: {len(train_prototypes)}")
    print(f"  - Atypiques train: {len(train_atypical_patients)}")
    print(f"  - Prototypes test: {len(test_metrics['prototypes'])}")
    print(f"  - Atypiques test: {len(test_metrics['atypical'])}")

    print(f"\nFichiers générés:")
    print(f"  - Modèle: {model_path}")
    print(f"  - Embeddings: {embeddings_path}")
    print(f"  - Graphe: {graph_path}")

    print(f"\nTerminé : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline avec validation train/test')

    parser.add_argument('--data_dir', type=str, default='./ptb-xl-data',
                       help='Répertoire du dataset PTB-XL')
    parser.add_argument('--output_dir', type=str, default='./output_validated',
                       help='Répertoire de sortie')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Nombre maximum d\'ECG à charger')
    parser.add_argument('--sampling_rate', type=int, default=100,
                       help='Fréquence d\'échantillonnage (100 ou 500 Hz)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Dimension de l\'embedding')
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help='Nombre de voisins pour le graphe K-NN')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille des batchs')
    parser.add_argument('--retrain', action='store_true',
                       help='Ré-entraîner même si un modèle existe')

    args = parser.parse_args()
    main(args)