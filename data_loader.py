"""
Module de chargement et prétraitement du dataset PTB-XL
"""
import os
import numpy as np
import pandas as pd
import wfdb
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class PTBXLDataLoader:
    """Charge et prétraite le dataset PTB-XL de PhysioNet"""

    def __init__(self, data_dir: str = './ptb-xl-data', sampling_rate: int = 100):
        """
        Args:
            data_dir: Répertoire contenant les données PTB-XL
            sampling_rate: Fréquence d'échantillonnage (100 ou 500 Hz)
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.metadata = None
        self.signals = None

    def download_dataset(self):
        """Télécharge le dataset PTB-XL depuis PhysioNet"""
        import subprocess

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("Téléchargement du dataset PTB-XL...")

            # Télécharger via wget ou curl
            url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

            try:
                subprocess.run([
                    "wget", "-O", f"{self.data_dir}/ptb-xl.zip", url
                ], check=True)

                subprocess.run([
                    "unzip", f"{self.data_dir}/ptb-xl.zip", "-d", self.data_dir
                ], check=True)

                print("✓ Dataset téléchargé et décompressé")
            except Exception as e:
                print(f"Erreur lors du téléchargement : {e}")
                print("\nVeuillez télécharger manuellement le dataset depuis :")
                print("https://physionet.org/content/ptb-xl/1.0.3/")

    def load_metadata(self) -> pd.DataFrame:
        """Charge les métadonnées des patients et diagnostics"""
        metadata_file = os.path.join(self.data_dir, 'ptbxl_database.csv')

        if not os.path.exists(metadata_file):
            # Chercher dans les sous-dossiers décompressés
            possible_paths = [
                os.path.join(self.data_dir, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3', 'ptbxl_database.csv'),
                os.path.join(self.data_dir, 'ptb-xl', 'ptbxl_database.csv')
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    metadata_file = path
                    self.data_dir = os.path.dirname(path)
                    break
            else:
                raise FileNotFoundError(
                    f"Fichier de métadonnées non trouvé. Assurez-vous que le dataset est dans {self.data_dir}"
                )

        self.metadata = pd.read_csv(metadata_file)
        print(f"✓ Métadonnées chargées : {len(self.metadata)} enregistrements")
        return self.metadata

    def load_signal(self, record_name: str) -> Optional[np.ndarray]:
        """
        Charge un signal ECG individuel

        Args:
            record_name: Nom du record (ex: 'records100/00000/00001_lr')

        Returns:
            Signal ECG de shape (longueur_signal, 12) pour les 12 dérivations
        """
        try:
            # Construire le chemin complet
            record_path = os.path.join(self.data_dir, record_name)

            # Charger le signal avec wfdb
            record = wfdb.rdsamp(record_path)
            signal = record[0]  # record[0] contient le signal, record[1] contient les métadonnées

            return signal
        except Exception as e:
            print(f"Erreur lors du chargement de {record_name}: {e}")
            return None

    def load_all_signals(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Charge tous les signaux ECG

        Args:
            max_samples: Nombre maximum d'échantillons à charger (None = tous)

        Returns:
            Tuple (signals, metadata) où signals a la shape (n_patients, longueur_signal, 12)
        """
        if self.metadata is None:
            self.load_metadata()

        # Sélectionner les records avec le bon sampling rate
        if self.sampling_rate == 100:
            filename_col = 'filename_lr'  # low resolution
        else:
            filename_col = 'filename_hr'  # high resolution

        # Limiter le nombre d'échantillons si demandé
        metadata_subset = self.metadata.head(max_samples) if max_samples else self.metadata

        signals_list = []
        valid_indices = []

        print(f"Chargement de {len(metadata_subset)} signaux ECG...")

        for idx, row in metadata_subset.iterrows():
            if idx % 1000 == 0:
                print(f"  Progression: {idx}/{len(metadata_subset)}")

            record_name = row[filename_col]
            signal = self.load_signal(record_name)

            if signal is not None:
                signals_list.append(signal)
                valid_indices.append(idx)

        # Convertir en array numpy
        self.signals = np.array(signals_list)
        valid_metadata = self.metadata.loc[valid_indices].reset_index(drop=True)

        print(f"✓ {len(signals_list)} signaux chargés avec succès")
        print(f"  Shape des signaux: {self.signals.shape}")

        return self.signals, valid_metadata

    def preprocess_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        Normalise les signaux ECG

        Args:
            signals: Array de shape (n_patients, longueur_signal, 12)

        Returns:
            Signaux normalisés
        """
        # Normalisation par signal (chaque ECG individuellement)
        normalized_signals = np.zeros_like(signals)

        for i in range(len(signals)):
            # Normalisation Z-score par dérivation
            mean = signals[i].mean(axis=0, keepdims=True)
            std = signals[i].std(axis=0, keepdims=True)
            std[std == 0] = 1  # Éviter division par zéro
            normalized_signals[i] = (signals[i] - mean) / std

        print("✓ Signaux normalisés")
        return normalized_signals

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Sépare les données en train/test

        Returns:
            X_train, X_test, y_train, y_test, metadata_train, metadata_test
        """
        if self.signals is None or self.metadata is None:
            raise ValueError("Chargez d'abord les signaux avec load_all_signals()")

        # Utiliser l'index comme "label" temporaire pour le split
        indices = np.arange(len(self.signals))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # On peut stratifier sur le diagnostic si besoin
        )

        X_train = self.signals[train_idx]
        X_test = self.signals[test_idx]

        metadata_train = self.metadata.iloc[train_idx].reset_index(drop=True)
        metadata_test = self.metadata.iloc[test_idx].reset_index(drop=True)

        print(f"✓ Split train/test : {len(X_train)} / {len(X_test)}")

        return X_train, X_test, metadata_train, metadata_test


if __name__ == "__main__":
    # Test du module
    loader = PTBXLDataLoader(data_dir='./ptb-xl-data')

    # Télécharger si nécessaire
    # loader.download_dataset()

    # Charger les métadonnées
    metadata = loader.load_metadata()
    print("\nAperçu des métadonnées:")
    print(metadata.head())
    print(f"\nColonnes: {metadata.columns.tolist()}")

    # Charger un échantillon de signaux
    print("\n--- Test de chargement d'un échantillon ---")
    signals, meta = loader.load_all_signals(max_samples=10)

    # Prétraiter
    normalized = loader.preprocess_signals(signals)
    print(f"Signaux normalisés - min: {normalized.min():.3f}, max: {normalized.max():.3f}")