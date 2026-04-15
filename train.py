"""
Script principal d'entraînement du pipeline complet :
1. Chargement des données PTB-XL
2. Entraînement de l'auto-encodeur
3. Extraction des embeddings
4. Construction du graphe de patients
5. Calcul des métriques
"""
import os
import numpy as np
import json
import argparse
from datetime import datetime

from data_loader import PTBXLDataLoader
from autoencoder import ECGAutoencoder
from graph_builder import PatientGraphBuilder


def main(args):
    """Pipeline complet d'entraînement"""

    print("=" * 80)
    print("PIPELINE DE CONSTRUCTION DU GRAPHE DE PATIENTS")
    print("=" * 80)
    print(f"\nDémarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========== 1. CHARGEMENT DES DONNÉES ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES PTB-XL")
    print("=" * 80)

    loader = PTBXLDataLoader(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate
    )

    # Charger les métadonnées
    metadata = loader.load_metadata()

    # Charger les signaux
    signals, valid_metadata = loader.load_all_signals(max_samples=args.max_samples)

    # Prétraiter
    normalized_signals = loader.preprocess_signals(signals)

    # Split train/test
    X_train, X_test, meta_train, meta_test = loader.get_train_test_split(
        test_size=args.test_size,
        random_state=42
    )

    print(f"\n✓ Données prêtes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # ========== 2. ENTRAÎNEMENT DE L'AUTO-ENCODEUR ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 2: ENTRAÎNEMENT DE L'AUTO-ENCODEUR")
    print("=" * 80)

    input_shape = (X_train.shape[1], X_train.shape[2])  # (longueur, 12 dérivations)

    # Vérifier si un modèle existe déjà
    model_path = os.path.join(args.output_dir, "models", "ecg_autoencoder")
    model_exists = os.path.exists(f"{model_path}_encoder.keras")

    if model_exists and not args.retrain:
        print(f"\n⚠️  Un modèle existe déjà à {model_path}")
        print("Chargement du modèle existant (utilisez --retrain pour ré-entraîner)")

        ae = ECGAutoencoder(
            input_shape=input_shape,
            embedding_dim=args.embedding_dim,
            filters=args.filters
        )
        ae.load(model_path)
    else:
        print("\nCréation d'un nouveau modèle...")

        ae = ECGAutoencoder(
            input_shape=input_shape,
            embedding_dim=args.embedding_dim,
            filters=args.filters
        )

        ae.compile_model(learning_rate=args.learning_rate)

        # Entraîner
        history = ae.train(
            X_train=X_train,
            X_val=X_test,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Sauvegarder
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ae.save(model_path)

        # Sauvegarder l'historique
        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        with open(f"{model_path}_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

    # ========== 3. EXTRACTION DES EMBEDDINGS ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 3: EXTRACTION DES EMBEDDINGS")
    print("=" * 80)

    # Utiliser tout le dataset (train + test) pour le graphe
    all_signals = np.concatenate([X_train, X_test], axis=0)
    all_metadata = valid_metadata  # Déjà combiné

    embeddings = ae.get_embeddings(all_signals)

    # Sauvegarder les embeddings
    embeddings_path = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    np.save(os.path.join(embeddings_path, "embeddings.npy"), embeddings)
    all_metadata.to_csv(os.path.join(embeddings_path, "metadata.csv"), index=False)

    print(f"✓ Embeddings sauvegardés : {embeddings_path}")

    # ========== 4. CONSTRUCTION DU GRAPHE ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 4: CONSTRUCTION DU GRAPHE DE PATIENTS")
    print("=" * 80)

    builder = PatientGraphBuilder(
        embeddings=embeddings,
        metadata=all_metadata
    )

    # Calculer la similarité
    builder.compute_similarity_matrix(metric=args.similarity_metric)

    # Construire le graphe K-NN
    graph = builder.build_knn_graph(
        k=args.k_neighbors,
        weight_threshold=args.weight_threshold
    )

    # ========== 5. CALCUL DES MÉTRIQUES ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 5: CALCUL DES MÉTRIQUES")
    print("=" * 80)

    # Représentativité et atypicité
    representativeness = builder.compute_representativeness()
    atypicality = builder.compute_atypicality()

    # Identifier les prototypes et atypiques
    n_special = min(10, len(embeddings) // 10)
    prototypes = builder.identify_prototypes(n_prototypes=n_special)
    atypical_patients = builder.identify_atypical(n_atypical=n_special)

    # ========== 6. EXPORT DES RÉSULTATS ==========
    print("\n" + "=" * 80)
    print("ÉTAPE 6: EXPORT DES RÉSULTATS")
    print("=" * 80)

    # Exporter les données du graphe
    graph_data = builder.export_graph_data()

    # Ajouter les informations supplémentaires
    graph_data['prototypes'] = prototypes
    graph_data['atypical'] = atypical_patients

    # Sauvegarder le graphe
    graph_path = os.path.join(args.output_dir, "graph")
    os.makedirs(graph_path, exist_ok=True)

    # Format JSON pour l'API
    with open(os.path.join(graph_path, "graph_data.json"), 'w') as f:
        json.dump(graph_data, f, indent=2)

    # Format GraphML pour NetworkX
    builder.save_graph(os.path.join(graph_path, "patient_graph.graphml"))

    # Sauvegarder les métriques
    metrics = {
        'representativeness': representativeness.tolist(),
        'atypicality': atypicality.tolist(),
        'prototypes': prototypes,
        'atypical': atypical_patients
    }

    with open(os.path.join(graph_path, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Résultats sauvegardés dans : {args.output_dir}")

    # ========== 7. RÉSUMÉ FINAL ==========
    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)

    print(f"""
Données:
  - Patients: {len(embeddings)}
  - Dimension des embeddings: {args.embedding_dim}

Graphe:
  - Nœuds: {graph_data['stats']['n_patients']}
  - Arêtes: {graph_data['stats']['n_connections']}
  - Degré moyen: {graph_data['stats']['avg_degree']:.2f}

Patients spéciaux:
  - Prototypes identifiés: {len(prototypes)}
  - Atypiques identifiés: {len(atypical_patients)}

Fichiers générés:
  - Modèle: {model_path}
  - Embeddings: {embeddings_path}
  - Graphe: {graph_path}
    """)

    print(f"\nTerminé : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de construction du graphe de patients")

    # Données
    parser.add_argument('--data_dir', type=str, default='./ptb-xl-data',
                        help='Répertoire du dataset PTB-XL')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='Fréquence d\'échantillonnage (100 ou 500 Hz)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Nombre maximum d\'échantillons à charger (None = tous)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion du test set')

    # Auto-encodeur
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimension de l\'embedding')
    parser.add_argument('--filters', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Nombre de filtres par couche convolutive')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--retrain', action='store_true',
                        help='Ré-entraîner même si un modèle existe')

    # Graphe
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Nombre de voisins pour le graphe K-NN')
    parser.add_argument('--similarity_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Métrique de similarité')
    parser.add_argument('--weight_threshold', type=float, default=0.0,
                        help='Seuil minimum de poids pour créer une arête')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Répertoire de sortie')

    args = parser.parse_args()

    # Lancer le pipeline
    main(args)