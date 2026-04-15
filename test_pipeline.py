"""
Script de test rapide pour vérifier le pipeline complet
"""
import numpy as np
import sys

print("=" * 80)
print("TEST DU PIPELINE GRAPHE DE PATIENTS")
print("=" * 80)

# Test 1: Imports
print("\n1. Test des imports...")
try:
    from data_loader import PTBXLDataLoader
    from autoencoder import ECGAutoencoder
    from graph_builder import PatientGraphBuilder
    print("   ✓ Tous les modules importés avec succès")
except Exception as e:
    print(f"   ✗ Erreur d'import: {e}")
    sys.exit(1)

# Test 2: Auto-encodeur avec données simulées
print("\n2. Test de l'auto-encodeur...")
try:
    # Créer des données simulées
    n_samples = 50
    signal_length = 1000
    n_leads = 12

    X_train = np.random.randn(n_samples, signal_length, n_leads).astype(np.float32)
    X_test = np.random.randn(10, signal_length, n_leads).astype(np.float32)

    # Créer et compiler le modèle
    ae = ECGAutoencoder(
        input_shape=(signal_length, n_leads),
        embedding_dim=32,
        filters=[16, 32]  # Architecture réduite pour le test
    )
    ae.compile_model()

    # Entraîner brièvement
    history = ae.train(
        X_train=X_train,
        X_val=X_test,
        epochs=2,
        batch_size=16
    )

    # Extraire embeddings
    embeddings = ae.get_embeddings(X_train)

    print(f"   ✓ Auto-encodeur fonctionnel")
    print(f"     - Embeddings shape: {embeddings.shape}")
    print(f"     - Loss final: {history.history['loss'][-1]:.4f}")

except Exception as e:
    print(f"   ✗ Erreur auto-encodeur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Construction du graphe
print("\n3. Test de la construction du graphe...")
try:
    # Construire le graphe
    builder = PatientGraphBuilder(embeddings=embeddings)

    # Calculer similarité
    sim_matrix = builder.compute_similarity_matrix(metric='cosine')

    # Construire graphe K-NN
    graph = builder.build_knn_graph(k=5)

    # Calculer métriques
    representativeness = builder.compute_representativeness()
    atypicality = builder.compute_atypicality()

    # Identifier prototypes et atypiques
    prototypes = builder.identify_prototypes(n_prototypes=5)
    atypical = builder.identify_atypical(n_atypical=5)

    # Exporter
    graph_data = builder.export_graph_data()

    print(f"   ✓ Graphe construit avec succès")
    print(f"     - Nœuds: {graph_data['stats']['n_patients']}")
    print(f"     - Arêtes: {graph_data['stats']['n_connections']}")
    print(f"     - Prototypes: {len(prototypes)}")
    print(f"     - Atypiques: {len(atypical)}")

except Exception as e:
    print(f"   ✗ Erreur construction graphe: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Data Loader (optionnel si dataset disponible)
print("\n4. Test du data loader (optionnel)...")
try:
    loader = PTBXLDataLoader(data_dir='./ptb-xl-data')

    # Vérifier si le dataset existe
    import os
    if os.path.exists('./ptb-xl-data'):
        metadata = loader.load_metadata()
        print(f"   ✓ Dataset trouvé: {len(metadata)} enregistrements")
    else:
        print("   ⚠️  Dataset PTB-XL non trouvé (normal si pas encore téléchargé)")
        print("      Téléchargez-le depuis: https://physionet.org/content/ptb-xl/1.0.3/")

except Exception as e:
    print(f"   ⚠️  Avertissement data loader: {e}")

# Résumé final
print("\n" + "=" * 80)
print("RÉSUMÉ DES TESTS")
print("=" * 80)
print("""
✓ Auto-encodeur CNN 1D : OK
✓ Extraction d'embeddings : OK
✓ Construction du graphe : OK
✓ Calcul des métriques : OK
✓ Identification prototypes/atypiques : OK

Prochaines étapes :
1. Télécharger le dataset PTB-XL
2. Lancer l'entraînement complet : python train.py --max_samples 1000
3. Démarrer l'API : python api.py
4. Accéder à l'interface : http://localhost:5173/services/graphe-patients

""")

print("✅ Tous les tests sont passés ! Le pipeline est fonctionnel.")
print("=" * 80)