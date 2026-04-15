"""
Construction de graphes de patients et calcul de métriques de représentativité/atypicité
"""
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List
import pandas as pd


class PatientGraphBuilder:
    """Construit un graphe de patients à partir d'embeddings"""

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame = None):
        """
        Args:
            embeddings: Embeddings des patients, shape (n_patients, embedding_dim)
            metadata: Métadonnées associées aux patients
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.n_patients = len(embeddings)

        self.graph = None
        self.similarity_matrix = None
        self.distance_matrix = None

        print(f"✓ GraphBuilder initialisé avec {self.n_patients} patients")
        print(f"  Dimension des embeddings: {embeddings.shape[1]}")

    def compute_similarity_matrix(self, metric: str = 'cosine') -> np.ndarray:
        """
        Calcule la matrice de similarité entre tous les patients

        Args:
            metric: 'cosine' ou 'euclidean'

        Returns:
            Matrice de similarité (n_patients, n_patients)
        """
        print(f"\nCalcul de la matrice de similarité ({metric})...")

        if metric == 'cosine':
            # Similarité cosinus (entre 0 et 1 pour vecteurs positifs)
            self.similarity_matrix = cosine_similarity(self.embeddings)

            # Convertir en distance pour certains calculs
            self.distance_matrix = 1 - self.similarity_matrix

        elif metric == 'euclidean':
            # Distance euclidienne
            self.distance_matrix = euclidean_distances(self.embeddings)

            # Convertir en similarité (plus la distance est petite, plus la similarité est grande)
            # Utiliser une fonction RBF-like
            sigma = np.median(self.distance_matrix)
            self.similarity_matrix = np.exp(-self.distance_matrix ** 2 / (2 * sigma ** 2))

        else:
            raise ValueError(f"Métrique non supportée: {metric}")

        print(f"✓ Matrice de similarité calculée")
        print(f"  Similarité moyenne: {self.similarity_matrix.mean():.4f}")
        print(f"  Similarité min/max: {self.similarity_matrix.min():.4f} / {self.similarity_matrix.max():.4f}")

        return self.similarity_matrix

    def build_knn_graph(self, k: int = 10, weight_threshold: float = 0.0) -> nx.Graph:
        """
        Construit un graphe K-NN (k plus proches voisins)

        Args:
            k: Nombre de plus proches voisins à connecter
            weight_threshold: Seuil minimum de similarité pour créer une arête

        Returns:
            Graphe NetworkX
        """
        print(f"\nConstruction du graphe K-NN (k={k})...")

        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Créer un graphe non dirigé
        self.graph = nx.Graph()

        # Ajouter les nœuds (patients)
        for i in range(self.n_patients):
            # Attributs du nœud
            node_attrs = {'patient_id': i}

            # Ajouter les métadonnées si disponibles
            if self.metadata is not None:
                for col in self.metadata.columns:
                    node_attrs[col] = self.metadata.iloc[i][col]

            self.graph.add_node(i, **node_attrs)

        # Trouver les k plus proches voisins pour chaque patient
        for i in range(self.n_patients):
            # Récupérer les similarités avec tous les autres patients
            similarities = self.similarity_matrix[i].copy()

            # Exclure le patient lui-même
            similarities[i] = -np.inf

            # Trouver les indices des k plus proches voisins
            k_neighbors_idx = np.argsort(similarities)[-k:]

            # Créer les arêtes
            for j in k_neighbors_idx:
                similarity = self.similarity_matrix[i, j]

                if similarity >= weight_threshold:
                    # Ajouter l'arête avec la similarité comme poids
                    self.graph.add_edge(
                        i, j,
                        weight=float(similarity),
                        distance=float(self.distance_matrix[i, j])
                    )

        print(f"✓ Graphe construit")
        print(f"  Nombre de nœuds: {self.graph.number_of_nodes()}")
        print(f"  Nombre d'arêtes: {self.graph.number_of_edges()}")
        print(f"  Degré moyen: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")

        return self.graph

    def compute_representativeness(self) -> np.ndarray:
        """
        Calcule un score de représentativité pour chaque patient

        Un patient est représentatif s'il est proche de nombreux autres patients.
        On utilise la moyenne des similarités avec les k plus proches voisins.

        Returns:
            Scores de représentativité (n_patients,)
        """
        print("\nCalcul des scores de représentativité...")

        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Méthode 1: Moyenne des similarités avec les k plus proches voisins
        k = min(20, self.n_patients - 1)
        representativeness = np.zeros(self.n_patients)

        for i in range(self.n_patients):
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = -np.inf
            top_k_similarities = np.sort(similarities)[-k:]
            representativeness[i] = top_k_similarities.mean()

        # Normaliser entre 0 et 1
        representativeness = (representativeness - representativeness.min()) / \
                             (representativeness.max() - representativeness.min() + 1e-8)

        print(f"✓ Scores de représentativité calculés")
        print(f"  Score moyen: {representativeness.mean():.4f}")
        print(f"  Score min/max: {representativeness.min():.4f} / {representativeness.max():.4f}")

        # Ajouter comme attribut des nœuds
        if self.graph is not None:
            for i in range(self.n_patients):
                self.graph.nodes[i]['representativeness'] = float(representativeness[i])

        return representativeness

    def compute_atypicality(self) -> np.ndarray:
        """
        Calcule un score d'atypicité pour chaque patient

        Un patient est atypique s'il est éloigné des autres patients.
        On utilise la distance moyenne aux k plus proches voisins.

        Returns:
            Scores d'atypicité (n_patients,)
        """
        print("\nCalcul des scores d'atypicité...")

        if self.distance_matrix is None:
            self.compute_similarity_matrix()

        # Méthode 1: Moyenne des distances aux k plus proches voisins
        k = min(20, self.n_patients - 1)
        atypicality = np.zeros(self.n_patients)

        for i in range(self.n_patients):
            distances = self.distance_matrix[i].copy()
            distances[i] = np.inf
            top_k_distances = np.sort(distances)[:k]
            atypicality[i] = top_k_distances.mean()

        # Normaliser entre 0 et 1
        atypicality = (atypicality - atypicality.min()) / \
                      (atypicality.max() - atypicality.min() + 1e-8)

        print(f"✓ Scores d'atypicité calculés")
        print(f"  Score moyen: {atypicality.mean():.4f}")
        print(f"  Score min/max: {atypicality.min():.4f} / {atypicality.max():.4f}")

        # Ajouter comme attribut des nœuds
        if self.graph is not None:
            for i in range(self.n_patients):
                self.graph.nodes[i]['atypicality'] = float(atypicality[i])

        return atypicality

    def identify_prototypes(self, n_prototypes: int = 10) -> List[int]:
        """
        Identifie les patients prototypiques (les plus représentatifs)

        Args:
            n_prototypes: Nombre de prototypes à identifier

        Returns:
            Liste des indices des patients prototypiques
        """
        representativeness = self.compute_representativeness()
        prototypes_idx = np.argsort(representativeness)[-n_prototypes:][::-1]

        print(f"\n✓ {n_prototypes} patients prototypiques identifiés:")
        for i, idx in enumerate(prototypes_idx[:5]):
            print(f"  {i + 1}. Patient {idx} - score: {representativeness[idx]:.4f}")

        # Marquer dans le graphe
        if self.graph is not None:
            for idx in prototypes_idx:
                self.graph.nodes[idx]['is_prototype'] = True

        return prototypes_idx.tolist()

    def identify_atypical(self, n_atypical: int = 10) -> List[int]:
        """
        Identifie les patients atypiques (les plus éloignés)

        Args:
            n_atypical: Nombre de patients atypiques à identifier

        Returns:
            Liste des indices des patients atypiques
        """
        atypicality = self.compute_atypicality()
        atypical_idx = np.argsort(atypicality)[-n_atypical:][::-1]

        print(f"\n✓ {n_atypical} patients atypiques identifiés:")
        for i, idx in enumerate(atypical_idx[:5]):
            print(f"  {i + 1}. Patient {idx} - score: {atypicality[idx]:.4f}")

        # Marquer dans le graphe
        if self.graph is not None:
            for idx in atypical_idx:
                self.graph.nodes[idx]['is_atypical'] = True

        return atypical_idx.tolist()

    def get_patient_neighbors(self, patient_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        Trouve les k patients les plus similaires à un patient donné

        Args:
            patient_id: ID du patient
            k: Nombre de voisins

        Returns:
            Liste de tuples (patient_id, similarité)
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        similarities = self.similarity_matrix[patient_id].copy()
        similarities[patient_id] = -np.inf

        neighbors_idx = np.argsort(similarities)[-k:][::-1]
        neighbors = [
            (int(idx), float(similarities[idx]))
            for idx in neighbors_idx
        ]

        return neighbors

    def export_graph_data(self) -> Dict:
        """
        Exporte les données du graphe pour visualisation

        Returns:
            Dictionnaire contenant nodes et edges
        """
        if self.graph is None:
            raise ValueError("Le graphe n'a pas encore été construit")

        # Convertir en format JSON-friendly
        nodes = []
        for node_id in self.graph.nodes():
            converted_attrs = {}
            for k, v in self.graph.nodes[node_id].items():
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    converted_attrs[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    converted_attrs[k] = float(v)
                elif isinstance(v, (np.bool_, bool)):
                    converted_attrs[k] = bool(v)
                else:
                    converted_attrs[k] = v

            node_data = {
                'id': int(node_id),
                **converted_attrs
            }
            nodes.append(node_data)

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edge_data = {
                'source': int(src),
                'target': int(dst),
                'weight': float(data.get('weight', 0)),
                'distance': float(data.get('distance', 0))
            }
            edges.append(edge_data)

        print(f"✓ Données exportées : {len(nodes)} nœuds, {len(edges)} arêtes")

        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'n_patients': len(nodes),
                'n_connections': len(edges),
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(nodes)
            }
        }

    def save_graph(self, filepath: str):
        """Sauvegarde le graphe au format GraphML"""
        if self.graph is None:
            raise ValueError("Le graphe n'a pas encore été construit")

        # Créer une copie du graphe et convertir tous les attributs numpy
        graph_copy = self.graph.copy()

        # Convertir les attributs des nœuds
        for node in graph_copy.nodes():
            for key, value in graph_copy.nodes[node].items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    graph_copy.nodes[node][key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    graph_copy.nodes[node][key] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    graph_copy.nodes[node][key] = bool(value)

        # Convertir les attributs des arêtes
        for u, v in graph_copy.edges():
            for key, value in graph_copy.edges[u, v].items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    graph_copy.edges[u, v][key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    graph_copy.edges[u, v][key] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    graph_copy.edges[u, v][key] = bool(value)

        nx.write_graphml(graph_copy, filepath)
        print(f"✓ Graphe sauvegardé : {filepath}")

    def load_graph(self, filepath: str):
        """Charge un graphe depuis un fichier GraphML"""
        self.graph = nx.read_graphml(filepath)
        print(f"✓ Graphe chargé : {filepath}")


if __name__ == "__main__":
    # Test avec des embeddings simulés
    print("Test du module de construction de graphe\n")

    # Créer des embeddings simulés
    n_patients = 100
    embedding_dim = 64

    np.random.seed(42)
    embeddings = np.random.randn(n_patients, embedding_dim)

    # Créer quelques clusters
    embeddings[:30] += [2, 0] + [0] * (embedding_dim - 2)  # Cluster 1
    embeddings[30:60] += [-2, 2] + [0] * (embedding_dim - 2)  # Cluster 2
    embeddings[60:90] += [0, -2] + [0] * (embedding_dim - 2)  # Cluster 3
    # 10 derniers patients isolés (atypiques)

    # Normaliser
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Construire le graphe
    builder = PatientGraphBuilder(embeddings)

    # Calculer la matrice de similarité
    builder.compute_similarity_matrix(metric='cosine')

    # Construire le graphe K-NN
    builder.build_knn_graph(k=10)

    # Calculer les métriques
    representativeness = builder.compute_representativeness()
    atypicality = builder.compute_atypicality()

    # Identifier prototypes et atypiques
    prototypes = builder.identify_prototypes(n_prototypes=5)
    atypical = builder.identify_atypical(n_atypical=5)

    # Trouver les voisins d'un patient
    patient_0_neighbors = builder.get_patient_neighbors(0, k=5)
    print(f"\nVoisins du patient 0:")
    for neighbor_id, similarity in patient_0_neighbors:
        print(f"  Patient {neighbor_id}: similarité = {similarity:.4f}")

    # Exporter les données
    graph_data = builder.export_graph_data()
    print(f"\nStatistiques du graphe:")
    print(f"  {graph_data['stats']}")