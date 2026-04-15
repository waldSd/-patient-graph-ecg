"""
Exploration détaillée du dataset PTB-XL pour comprendre ce qu'on manipule
"""
import pandas as pd
import numpy as np
from data_loader import PTBXLDataLoader
import ast

print("=" * 80)
print("EXPLORATION DÉTAILLÉE DU DATASET PTB-XL")
print("=" * 80)

# Créer le loader
loader = PTBXLDataLoader(data_dir='./ptb-xl-data')
metadata = loader.load_metadata()

print(f"\n📊 STATISTIQUES GLOBALES")
print(f"{'='*80}")
print(f"Nombre total d'ECG : {len(metadata)}")
print(f"Nombre de patients uniques : {metadata['patient_id'].nunique()}")
print(f"Période de collecte : Octobre 1989 - Juin 1996 (7 ans)")
print(f"Durée de chaque ECG : 10 secondes")
print(f"Nombre de dérivations : 12 (I, II, III, AVL, AVR, AVF, V1-V6)")

# DÉMOGRAPHIE
print(f"\n👥 DÉMOGRAPHIE DES PATIENTS")
print(f"{'='*80}")
print(f"Âge moyen : {metadata['age'].mean():.1f} ans")
print(f"Âge médian : {metadata['age'].median():.0f} ans")
print(f"Âge min/max : {metadata['age'].min():.0f} / {metadata['age'].max():.0f} ans")

sex_counts = metadata['sex'].value_counts()
print(f"\nRépartition par sexe :")
print(f"  - Hommes (0) : {sex_counts.get(0, 0)} ({sex_counts.get(0, 0)/len(metadata)*100:.1f}%)")
print(f"  - Femmes (1) : {sex_counts.get(1, 0)} ({sex_counts.get(1, 0)/len(metadata)*100:.1f}%)")

# DIAGNOSTICS
print(f"\n🩺 DIAGNOSTICS (SCP-ECG)")
print(f"{'='*80}")

# Charger les statements SCP
scp_statements = pd.read_csv('./ptb-xl-data/scp_statements.csv')
print(f"Nombre de codes diagnostiques différents : {len(scp_statements)}")

# Analyser les diagnostics dans les metadata
# Les scp_codes sont stockés comme dictionnaires dans une colonne
print("\nExemple de codes diagnostiques d'un patient :")
# Prendre le premier ECG avec des codes
for idx, row in metadata.iterrows():
    if pd.notna(row['scp_codes']):
        scp_dict = ast.literal_eval(row['scp_codes'])
        print(f"  Patient {int(row['patient_id'])} (ECG {row['ecg_id']}) :")
        for code, likelihood in list(scp_dict.items())[:3]:  # Top 3
            # Trouver la description
            desc_row = scp_statements[scp_statements['Unnamed: 0'] == code]
            if not desc_row.empty:
                desc = desc_row.iloc[0]['description']
                print(f"    - {code}: {desc} (probabilité: {likelihood})")
        break

# Compter les grandes catégories
print(f"\n📈 DISTRIBUTION DES PATHOLOGIES (Superclasses)")
print(f"{'='*80}")

# Créer un mapping des catégories principales
diagnostic_superclass = scp_statements[['Unnamed: 0', 'diagnostic_class']].set_index('Unnamed: 0')['diagnostic_class'].to_dict()

# Compter chaque superclasse
superclass_counts = {}
for idx, row in metadata.iterrows():
    if pd.notna(row['scp_codes']):
        scp_dict = ast.literal_eval(row['scp_codes'])
        for code in scp_dict.keys():
            if code in diagnostic_superclass:
                superclass = diagnostic_superclass[code]
                if pd.notna(superclass):
                    superclass_counts[superclass] = superclass_counts.get(superclass, 0) + 1

# Trier et afficher
print("\nNombre d'ECG par catégorie diagnostique :")
for category, count in sorted(superclass_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {category:25s} : {count:5d} ECG ({count/len(metadata)*100:5.1f}%)")

# QUALITÉ DU SIGNAL
print(f"\n📡 QUALITÉ DU SIGNAL")
print(f"{'='*80}")

quality_cols = ['static_noise', 'burst_noise', 'baseline_drift', 'electrodes_problems']
print("\nProblèmes de qualité détectés :")
for col in quality_cols:
    if col in metadata.columns:
        n_problems = metadata[col].notna().sum()
        print(f"  {col:25s} : {n_problems:5d} ECG ({n_problems/len(metadata)*100:5.1f}%)")

# SPLITS TRAIN/TEST
print(f"\n🔀 SPLITS RECOMMANDÉS POUR L'ENTRAÎNEMENT")
print(f"{'='*80}")
fold_counts = metadata['strat_fold'].value_counts().sort_index()
print("\nRépartition par fold (stratifiée par patient) :")
for fold, count in fold_counts.items():
    usage = "TRAIN" if fold <= 8 else ("VALIDATION" if fold == 9 else "TEST")
    print(f"  Fold {fold:2d} : {count:5d} ECG - Usage recommandé : {usage}")

print(f"\nTotaux recommandés :")
train_count = metadata[metadata['strat_fold'] <= 8].shape[0]
val_count = metadata[metadata['strat_fold'] == 9].shape[0]
test_count = metadata[metadata['strat_fold'] == 10].shape[0]
print(f"  TRAIN (folds 1-8)  : {train_count:5d} ECG ({train_count/len(metadata)*100:.1f}%)")
print(f"  VALIDATION (fold 9): {val_count:5d} ECG ({val_count/len(metadata)*100:.1f}%)")
print(f"  TEST (fold 10)     : {test_count:5d} ECG ({test_count/len(metadata)*100:.1f}%)")

# EXEMPLE CONCRET : CHARGER UN SIGNAL
print(f"\n📊 EXEMPLE CONCRET : UN SIGNAL ECG")
print(f"{'='*80}")

# Prendre un ECG normal
normal_ecgs = metadata[metadata['scp_codes'].str.contains('NORM', na=False)]
if len(normal_ecgs) > 0:
    example_row = normal_ecgs.iloc[0]
    print(f"\nChargement d'un ECG normal...")
    print(f"  Patient ID : {int(example_row['patient_id'])}")
    print(f"  Age : {int(example_row['age'])} ans")
    print(f"  Sexe : {'Homme' if example_row['sex'] == 0 else 'Femme'}")

    # Charger le signal
    signal = loader.load_signal(example_row['filename_lr'])
    if signal is not None:
        print(f"\n  Signal chargé avec succès !")
        print(f"  Shape : {signal.shape} (temps x dérivations)")
        print(f"  Durée : {signal.shape[0] / 100:.1f} secondes (100 Hz)")
        print(f"  Dérivations : 12 (I, II, III, AVL, AVR, AVF, V1, V2, V3, V4, V5, V6)")
        print(f"\n  Statistiques du signal :")
        print(f"    - Amplitude min : {signal.min():.2f} mV")
        print(f"    - Amplitude max : {signal.max():.2f} mV")
        print(f"    - Amplitude moyenne : {signal.mean():.2f} mV")
        print(f"    - Écart-type : {signal.std():.2f} mV")

        print(f"\n  Aperçu des premières valeurs (dérivation I) :")
        print(f"    {signal[:10, 0]}")

# CE QU'ON VA FAIRE AVEC
print(f"\n🎯 CE QU'ON VA FAIRE AVEC CE DATASET")
print(f"{'='*80}")
print("""
1. EXTRACTION D'EMBEDDINGS (Représentations Latentes)
   ---------------------------------------------------
   • Chaque ECG (1000 points × 12 dérivations) sera compressé
   • Par un auto-encodeur CNN 1D entraîné à reconstruire le signal
   • En un vecteur compact de 64 dimensions (embedding)
   • Cet embedding capture les "patterns cardiaques" essentiels

   EXEMPLE :
   ECG de 12 000 valeurs → Auto-encodeur → Embedding de 64 valeurs

2. CONSTRUCTION DU GRAPHE DE PATIENTS
   ------------------------------------
   • Calculer la similarité entre tous les embeddings
   • Si deux patients ont des embeddings proches → ECG similaires
   • Créer une arête entre eux (graphe K-NN)
   • Poids de l'arête = similarité cosinus

   EXEMPLE :
   Patient A (embedding: [0.2, 0.5, ...]) ≈ Patient B (embedding: [0.3, 0.4, ...])
   → Similarité = 0.92 → Créer une arête forte

3. IDENTIFICATION DES PROTOTYPES
   --------------------------------
   • Patients PROTOTYPIQUES = proches de beaucoup d'autres
   • Score de représentativité élevé
   • Représentent des profils "typiques"
   • Exemple : "ECG normal classique", "Infarctus typique"

4. DÉTECTION DES CAS ATYPIQUES
   -----------------------------
   • Patients ATYPIQUES = éloignés de tous les autres
   • Score d'atypicité élevé
   • Cas rares, combinaisons inhabituelles
   • Nécessitent attention particulière du clinicien

5. RAISONNEMENT CLINIQUE PAR ANALOGIE
   ------------------------------------
   • Nouveau patient → Trouver ses k voisins dans le graphe
   • "Votre ECG ressemble à ces 5 patients"
   • Aide le médecin à comparer avec des cas connus
   • Aligné avec la théorie des prototypes en psychologie cognitive
""")

print("\n" + "=" * 80)
print("PRÊT POUR L'ENTRAÎNEMENT !")
print("=" * 80)
print("""
Avec python train.py --max_samples 1000 --epochs 50, on va :

1. Charger 1000 ECG réels de ce dataset
2. Entraîner l'auto-encodeur à les compresser intelligemment
3. Extraire les 1000 embeddings résultants
4. Construire le graphe de similarité
5. Calculer représentativité et atypicité pour chaque patient
6. Identifier automatiquement les prototypes et cas atypiques

Temps estimé : 15-20 minutes sur CPU
Résultat : Un graphe explorable dans l'interface web !
""")