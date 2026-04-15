"""Test rapide pour vérifier que le dataset PTB-XL est bien chargé"""
from data_loader import PTBXLDataLoader

print("=" * 60)
print("TEST DE CHARGEMENT DU DATASET PTB-XL")
print("=" * 60)

# Créer le loader
loader = PTBXLDataLoader(data_dir='./ptb-xl-data')

# Charger les métadonnées
print("\n1. Chargement des métadonnées...")
metadata = loader.load_metadata()

print(f"\n✅ {len(metadata)} ECG trouvés dans le dataset")
print(f"\nColonnes disponibles : {list(metadata.columns[:10])}")
print(f"\nPremiers enregistrements :")
print(metadata.head())

# Tester le chargement d'un signal
print("\n2. Test de chargement d'un signal ECG...")
first_record = metadata.iloc[0]['filename_lr']
print(f"Chargement de : {first_record}")

signal = loader.load_signal(first_record)
if signal is not None:
    print(f"✅ Signal chargé : shape {signal.shape}")
    print(f"   Durée : {signal.shape[0]/100:.1f} secondes")
    print(f"   Dérivations : {signal.shape[1]}")
else:
    print("❌ Erreur lors du chargement du signal")

print("\n" + "=" * 60)
print("DATASET PRÊT À L'EMPLOI !")
print("=" * 60)
print("\nVous pouvez maintenant lancer l'entraînement avec :")
print("  python train.py --max_samples 1000 --epochs 50")