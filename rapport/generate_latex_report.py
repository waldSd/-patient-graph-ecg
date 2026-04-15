"""
Génère automatiquement un rapport LaTeX à partir des résultats
"""
import json
import os
from datetime import datetime

# Charger les statistiques
with open('statistics.json', 'r') as f:
    stats = json.load(f)

# Template LaTeX
latex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{float}
\usepackage{geometry}
\geometry{margin=2.5cm}

\title{\textbf{Graphes de Patients basés sur les Embeddings ECG} \\
\large{Démonstration pour Candidature Thèse CIFRE}}
\author{Candidat : Walid \\
Laboratoire : CReSTIC, Université de Reims}
\date{""" + datetime.now().strftime('%d %B %Y') + r"""}

\begin{document}

\maketitle

\begin{abstract}
Ce rapport présente une implémentation concrète de l'approche proposée dans le sujet de thèse "Graphes de patients et représentations latentes pour l'aide au raisonnement clinique". Nous utilisons un auto-encodeur CNN 1D entraîné sur le dataset PTB-XL (""" + str(stats['dataset']['n_patients']) + r""" ECG réels) pour extraire des embeddings compacts, puis nous construisons un graphe de patients permettant d'identifier automatiquement des cas prototypiques et atypiques. Cette approche s'inspire de la psychologie cognitive (théorie des prototypes) et facilite le raisonnement clinique par analogie.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}

\subsection{Contexte de la Thèse}

L'adoption des systèmes d'intelligence artificielle en médecine reste limitée, notamment en raison du manque de confiance qu'ils suscitent. Le raisonnement clinique repose sur de nombreux mécanismes cognitifs : comparaison de cas similaires, raisonnement par analogie, identification de profils typiques ou de situations inhabituelles.

Ce projet propose d'exploiter les \textbf{représentations vectorielles denses} (embeddings) produites par des modèles d'apprentissage profond pour construire des \textbf{graphes de patients}, où chaque nœud représente un patient et chaque arête traduit une relation de proximité apprise automatiquement.

\subsection{Objectifs de cette Démonstration}

\begin{itemize}
    \item Extraire des embeddings de signaux ECG réels via un auto-encodeur CNN 1D
    \item Construire un graphe de similarité entre patients
    \item Identifier automatiquement des patients prototypiques (représentatifs)
    \item Détecter des cas atypiques nécessitant une attention particulière
    \item Démontrer l'alignement avec le raisonnement clinique
\end{itemize}

\section{Méthodologie}

\subsection{Dataset PTB-XL}

Nous utilisons le dataset PTB-XL de PhysioNet \cite{wagner2020ptbxl}, une référence en cardiologie numérique :

\begin{table}[H]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Caractéristique} & \textbf{Valeur} \\
\midrule
Nombre d'ECG & """ + str(stats['dataset']['n_patients']) + r""" \\
Durée par ECG & """ + str(stats['dataset']['signal_duration']) + r""" secondes \\
Fréquence d'échantillonnage & """ + str(stats['dataset']['sampling_rate']) + r""" Hz \\
Nombre de dérivations & """ + str(stats['dataset']['n_leads']) + r""" (I, II, III, AVL, AVR, AVF, V1-V6) \\
Points par ECG & """ + str(stats['dataset']['signal_duration'] * stats['dataset']['sampling_rate']) + r""" × """ + str(stats['dataset']['n_leads']) + r""" = """ + str(stats['dataset']['signal_duration'] * stats['dataset']['sampling_rate'] * stats['dataset']['n_leads']) + r""" valeurs \\
\bottomrule
\end{tabular}
\caption{Caractéristiques du dataset PTB-XL}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/01_demographics.png}
\caption{Démographie des patients du dataset PTB-XL}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/02_example_ecg.png}
\caption{Exemple d'ECG 12-dérivations extrait du dataset}
\end{figure}

\subsection{Architecture de l'Auto-encodeur CNN 1D}

L'auto-encodeur est composé de :

\begin{itemize}
    \item \textbf{Encodeur} : 4 blocs convolutifs (32, 64, 128, 256 filtres) avec batch normalization et max pooling, suivis d'un global average pooling et d'une couche dense vers l'embedding
    \item \textbf{Décodeur} : Architecture symétrique avec upsampling et déconvolutions
    \item \textbf{Embedding} : Vecteur compact de """ + str(stats['model']['embedding_dim']) + r""" dimensions
    \item \textbf{Taux de compression} : """ + f"{stats['model']['compression_ratio']:.1f}" + r"""× (""" + str(stats['dataset']['signal_duration'] * stats['dataset']['sampling_rate'] * stats['dataset']['n_leads']) + r""" → """ + str(stats['model']['embedding_dim']) + r""" valeurs)
\end{itemize}

L'objectif est de forcer le modèle à capturer les \textit{patterns cardiaques essentiels} dans un espace latent réduit, de manière à ce que deux ECG similaires aient des embeddings proches.

\subsection{Entraînement}

\begin{table}[H]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Hyperparamètre} & \textbf{Valeur} \\
\midrule
Loss & MSE (Mean Squared Error) \\
Optimiseur & Adam \\
Learning rate & 0.001 \\
Époques & 50 \\
Batch size & 32 \\
\midrule
Loss finale (train) & """ + f"{stats['model']['final_train_loss']:.6f}" + r""" \\
Loss finale (validation) & """ + f"{stats['model']['final_val_loss']:.6f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Paramètres et résultats de l'entraînement}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/03_training_curves.png}
\caption{Courbes d'apprentissage de l'auto-encodeur}
\end{figure}

\section{Construction du Graphe de Patients}

\subsection{Calcul de Similarité}

Une fois les embeddings extraits, nous calculons la \textbf{similarité cosinus} entre tous les patients :

\begin{equation}
\text{similarité}(p_i, p_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}
\end{equation}

où $\mathbf{e}_i$ et $\mathbf{e}_j$ sont les embeddings des patients $i$ et $j$.

\subsection{Graphe K-NN}

Pour chaque patient, nous créons des arêtes vers ses \textbf{k plus proches voisins} (k=""" + str(stats['graph']['k_neighbors']) + r"""). Cela donne un graphe de :

\begin{itemize}
    \item \textbf{Nœuds} : """ + str(stats['graph']['n_nodes']) + r""" patients
    \item \textbf{Arêtes} : """ + str(stats['graph']['n_edges']) + r""" connexions
    \item \textbf{Degré moyen} : """ + f"{stats['graph']['avg_degree']:.2f}" + r"""
\end{itemize}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/04_similarity_distribution.png}
\caption{Distribution des similarités entre patients}
\end{figure}

\section{Métriques Cognitives}

\subsection{Représentativité (Patients Prototypiques)}

Un patient est \textbf{prototypique} s'il est proche de nombreux autres patients. Nous définissons :

\begin{equation}
\text{représentativité}(p_i) = \frac{1}{k} \sum_{j \in N_k(i)} \text{similarité}(p_i, p_j)
\end{equation}

où $N_k(i)$ désigne les k plus proches voisins du patient $i$.

\textbf{Interprétation} : Les prototypes représentent des profils "typiques" (ex: "ECG normal classique", "Infarctus typique") et peuvent servir de références pour le diagnostic par comparaison.

\subsection{Atypicité (Cas Rares)}

Un patient est \textbf{atypique} s'il est éloigné des autres patients :

\begin{equation}
\text{atypicité}(p_i) = \frac{1}{k} \sum_{j \in N_k(i)} \text{distance}(p_i, p_j)
\end{equation}

\textbf{Interprétation} : Les cas atypiques signalent des situations inhabituelles nécessitant une attention particulière du clinicien.

\subsection{Résultats}

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Métrique} & \textbf{Moyenne} & \textbf{Patients identifiés} \\
\midrule
Représentativité & """ + f"{stats['metrics']['avg_representativeness']:.4f}" + r""" & """ + str(stats['metrics']['n_prototypes']) + r""" prototypes \\
Atypicité & """ + f"{stats['metrics']['avg_atypicality']:.4f}" + r""" & """ + str(stats['metrics']['n_atypical']) + r""" atypiques \\
\bottomrule
\end{tabular}
\caption{Statistiques des métriques cognitives}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{figures/05_metrics_distribution.png}
\caption{Distribution de la représentativité et de l'atypicité}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/06_prototypes_vs_atypical.png}
\caption{Carte des patients : prototypes (étoiles vertes) vs atypiques (triangles rouges)}
\end{figure}

\section{Discussion}

\subsection{Alignement avec le Sujet de Thèse}

Cette démonstration illustre concrètement les concepts proposés dans le sujet :

\begin{enumerate}
    \item \textbf{Embeddings} : Représentations latentes extraites par apprentissage profond ✓
    \item \textbf{Graphes de patients} : Structure de similarité apprise automatiquement ✓
    \item \textbf{Prototypes} : Aligné avec la psychologie cognitive (théorie des prototypes) ✓
    \item \textbf{Atypicité} : Détection de situations rares nécessitant expertise ✓
    \item \textbf{Données réelles} : PTB-XL, référence scientifique en cardiologie ✓
\end{enumerate}

\subsection{Applications au Raisonnement Clinique}

\textbf{Diagnostic par analogie} : "Votre ECG ressemble à ces 5 patients connus"

\textbf{Détection d'anomalies} : "Ce patient a un score d'atypicité élevé (0.95), investigation recommandée"

\textbf{Apprentissage médical} : Les prototypes servent de références pédagogiques

\subsection{Extensions Possibles}

\begin{itemize}
    \item \textbf{Multimodalité} : Combiner ECG, images radiologiques, données tabulaires
    \item \textbf{Interprétabilité} : Expliquer pourquoi deux patients sont similaires
    \item \textbf{Validation clinique} : Comparer avec diagnostics réels et jugements d'experts
    \item \textbf{Robustesse} : Étudier la stabilité des graphes sous différentes conditions
\end{itemize}

\section{Conclusion}

Cette démonstration prouve la faisabilité de l'approche proposée dans la thèse. Les embeddings extraits par auto-encodeur permettent de construire des graphes de patients riches et exploitables, alignés avec le raisonnement clinique humain.

\textbf{Points forts} :
\begin{itemize}
    \item Implémentation complète et fonctionnelle
    \item Données médicales réelles (PTB-XL)
    \item Métriques cognitives innovantes
    \item Interface de visualisation interactive
\end{itemize}

Ce travail constitue une base solide pour les travaux de thèse proposés.

\begin{thebibliography}{9}
\bibitem{wagner2020ptbxl}
Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., \& Schaeffter, T. (2020).
PTB-XL, a large publicly available electrocardiography dataset.
\textit{Scientific data}, 7(1), 1-15.

\bibitem{rosch1973}
Rosch, E. (1973).
Natural categories and prototypes.
\textit{Cognitive psychology}, 4(3), 328-350.
\end{thebibliography}

\end{document}
"""

# Sauvegarder le fichier LaTeX
with open('rapport.tex', 'w') as f:
    f.write(latex_content)

print("✅ Rapport LaTeX généré : rapport/rapport.tex")
print("\nPour compiler le PDF :")
print("  cd rapport/")
print("  pdflatex rapport.tex")
print("  pdflatex rapport.tex  # 2ème fois pour la table des matières")