"""
Auto-encodeur CNN 1D pour extraction d'embeddings à partir de signaux ECG
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple


class ECGAutoencoder:
    """Auto-encodeur convolutif 1D pour signaux ECG 12-dérivations"""

    def __init__(
            self,
            input_shape: Tuple[int, int],
            embedding_dim: int = 64,
            filters: list = None
    ):
        """
        Args:
            input_shape: Shape de l'input (longueur_signal, n_derivations)
            embedding_dim: Dimension de l'embedding latent
            filters: Liste du nombre de filtres par couche convolutive
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.filters = filters or [32, 64, 128, 256]

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self._build_model()

    def _build_encoder(self, input_layer):
        """Construit l'encodeur (partie compression)"""
        x = input_layer

        # Blocs convolutifs avec pooling
        for i, num_filters in enumerate(self.filters):
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'encoder_conv_{i + 1}'
            )(x)

            x = layers.BatchNormalization(name=f'encoder_bn_{i + 1}')(x)

            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'encoder_conv_{i + 1}_b'
            )(x)

            x = layers.MaxPooling1D(
                pool_size=2,
                name=f'encoder_pool_{i + 1}'
            )(x)

            x = layers.Dropout(0.2, name=f'encoder_dropout_{i + 1}')(x)

        # Aplatir et réduire à l'embedding
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

        # Couche dense pour l'embedding
        embedding = layers.Dense(
            self.embedding_dim,
            activation='relu',
            name='embedding'
        )(x)

        return embedding

    def _build_decoder(self, embedding_layer):
        """Construit le décodeur (partie reconstruction)"""
        # Calculer la taille après tous les poolings
        pooling_factor = 2 ** len(self.filters)
        reduced_length = self.input_shape[0] // pooling_factor

        # Projeter l'embedding vers une shape compatible
        x = layers.Dense(
            reduced_length * self.filters[-1],
            activation='relu',
            name='decoder_dense'
        )(embedding_layer)

        x = layers.Reshape(
            (reduced_length, self.filters[-1]),
            name='decoder_reshape'
        )(x)

        # Blocs de déconvolution (upsampling)
        for i, num_filters in enumerate(reversed(self.filters)):
            x = layers.UpSampling1D(
                size=2,
                name=f'decoder_upsample_{i + 1}'
            )(x)

            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'decoder_conv_{i + 1}'
            )(x)

            x = layers.BatchNormalization(name=f'decoder_bn_{i + 1}')(x)

            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                name=f'decoder_conv_{i + 1}_b'
            )(x)

        # Couche de sortie pour reconstruire les 12 dérivations
        output = layers.Conv1D(
            filters=self.input_shape[1],  # 12 dérivations
            kernel_size=3,
            padding='same',
            activation='linear',
            name='decoder_output'
        )(x)

        # Ajuster la taille pour correspondre exactement à l'input
        # Calculer le padding nécessaire
        current_length = reduced_length * pooling_factor
        padding_needed = self.input_shape[0] - current_length
        if padding_needed > 0:
            # Ajouter du padding pour atteindre la taille exacte
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            output = layers.ZeroPadding1D(
                padding=(padding_left, padding_right),
                name='decoder_final_padding'
            )(output)
        elif padding_needed < 0:
            # Couper l'excédent
            output = layers.Cropping1D(
                cropping=(-padding_needed // 2, -padding_needed - (-padding_needed // 2)),
                name='decoder_final_cropping'
            )(output)

        return output

    def _build_model(self):
        """Construit l'auto-encodeur complet"""
        # Input
        input_layer = layers.Input(
            shape=self.input_shape,
            name='ecg_input'
        )

        # Encoder
        embedding = self._build_encoder(input_layer)
        self.encoder = Model(
            inputs=input_layer,
            outputs=embedding,
            name='encoder'
        )

        # Decoder
        decoder_input = layers.Input(
            shape=(self.embedding_dim,),
            name='decoder_input'
        )
        decoder_output = self._build_decoder(decoder_input)
        self.decoder = Model(
            inputs=decoder_input,
            outputs=decoder_output,
            name='decoder'
        )

        # Autoencoder complet
        reconstructed = self.decoder(embedding)
        self.autoencoder = Model(
            inputs=input_layer,
            outputs=reconstructed,
            name='autoencoder'
        )

        print("✓ Architecture de l'auto-encodeur construite")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Embedding dimension: {self.embedding_dim}")

    def compile_model(
            self,
            learning_rate: float = 0.001,
            loss: str = 'mse'
    ):
        """Compile le modèle avec optimizer et loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.autoencoder.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae']
        )

        print("✓ Modèle compilé")

    def train(
            self,
            X_train: np.ndarray,
            X_val: np.ndarray = None,
            epochs: int = 100,
            batch_size: int = 32,
            callbacks: list = None
    ):
        """
        Entraîne l'auto-encodeur

        Args:
            X_train: Données d'entraînement
            X_val: Données de validation
            epochs: Nombre d'époques
            batch_size: Taille des batchs
            callbacks: Liste de callbacks Keras

        Returns:
            History de l'entraînement
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

        print(f"\n--- Entraînement de l'auto-encodeur ---")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Train samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")

        history = self.autoencoder.fit(
            X_train, X_train,  # Auto-encodeur : input = target
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("✓ Entraînement terminé")
        return history

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extrait les embeddings à partir des signaux ECG

        Args:
            X: Signaux ECG de shape (n_samples, longueur, 12)

        Returns:
            Embeddings de shape (n_samples, embedding_dim)
        """
        embeddings = self.encoder.predict(X, verbose=0)
        print(f"✓ Embeddings extraits : shape {embeddings.shape}")
        return embeddings

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruit les signaux à partir des inputs

        Args:
            X: Signaux ECG

        Returns:
            Signaux reconstruits
        """
        return self.autoencoder.predict(X, verbose=0)

    def save(self, path: str):
        """Sauvegarde les modèles"""
        self.autoencoder.save(f"{path}_autoencoder.keras")
        self.encoder.save(f"{path}_encoder.keras")
        self.decoder.save(f"{path}_decoder.keras")
        print(f"✓ Modèles sauvegardés : {path}")

    def load(self, path: str):
        """Charge les modèles"""
        self.autoencoder = keras.models.load_model(f"{path}_autoencoder.keras")
        self.encoder = keras.models.load_model(f"{path}_encoder.keras")
        self.decoder = keras.models.load_model(f"{path}_decoder.keras")
        print(f"✓ Modèles chargés : {path}")

    def summary(self):
        """Affiche le résumé des modèles"""
        print("\n=== ENCODER ===")
        self.encoder.summary()
        print("\n=== DECODER ===")
        self.decoder.summary()
        print("\n=== AUTOENCODER ===")
        self.autoencoder.summary()


if __name__ == "__main__":
    # Test de l'architecture
    print("Test de l'architecture de l'auto-encodeur\n")

    # Paramètres typiques pour PTB-XL à 100Hz
    # ECG de 10 secondes à 100Hz = 1000 points, 12 dérivations
    input_shape = (1000, 12)
    embedding_dim = 64

    # Créer le modèle
    ae = ECGAutoencoder(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        filters=[32, 64, 128, 256]
    )

    # Afficher les architectures
    ae.summary()

    # Test avec des données aléatoires
    print("\n--- Test avec données simulées ---")
    X_test = np.random.randn(10, 1000, 12).astype(np.float32)

    # Compiler
    ae.compile_model()

    # Extraire embeddings
    embeddings = ae.get_embeddings(X_test)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reconstruction
    reconstructed = ae.reconstruct(X_test)
    print(f"Reconstruction shape: {reconstructed.shape}")
    print(f"MSE de reconstruction: {np.mean((X_test - reconstructed) ** 2):.6f}")