# Rapport de Démonstration - Graphes de Patients

## 📁 Contenu du Dossier

### Fichiers Générés Automatiquement

- **`statistics.json`** : Toutes les statistiques clés (dataset, modèle, graphe, métriques)
- **`rapport.tex`** : Rapport LaTeX complet auto-généré
- **`figures/`** : 6 figures haute résolution (300 DPI) pour publication

### Figures Disponibles

1. **`01_demographics.png`** : Distribution âge et sexe des patients
2. **`02_example_ecg.png`** : Exemple d'ECG 12-dérivations
3. **`03_training_curves.png`** : Courbes d'apprentissage (loss et MAE)
4. **`04_similarity_distribution.png`** : Distribution des similarités entre patients
5. **`05_metrics_distribution.png`** : Histogrammes représentativité et atypicité
6. **`06_prototypes_vs_atypical.png`** : Scatter plot prototypes vs atypiques

## 🚀 Génération du Rapport

### Étape 1 : Exécuter le Notebook

Le notebook **`training_notebook.ipynb`** génère automatiquement :
- Toutes les figures dans `rapport/figures/`
- Les statistiques dans `rapport/statistics.json`

```bash
# Lancer Jupyter
source venv/bin/activate
jupyter notebook training_notebook.ipynb
```

Puis : **Cell → Run All**

### Étape 2 : Générer le LaTeX

```bash
cd rapport/
python generate_latex_report.py
```

Cela crée `rapport.tex` avec toutes les statistiques insérées automatiquement.

### Étape 3 : Compiler le PDF

```bash
pdflatex rapport.tex
pdflatex rapport.tex  # 2ème fois pour table des matières
```

Résultat : **`rapport.pdf`** prêt pour la candidature !

## 📊 Utilisation dans un Document LaTeX Existant

Si vous avez déjà un rapport LaTeX, vous pouvez :

### Inclure les Figures

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{chemin/vers/rapport/figures/01_demographics.png}
\caption{Distribution démographique du dataset PTB-XL}
\label{fig:demographics}
\end{figure}
```

### Utiliser les Statistiques

Les statistiques sont dans `statistics.json` :

```json
{
  "dataset": {
    "n_patients": 1000,
    "sampling_rate": 100,
    ...
  },
  "model": {
    "embedding_dim": 64,
    "compression_ratio": 187.5,
    "final_train_loss": 0.001234,
    ...
  },
  ...
}
```

Vous pouvez les insérer manuellement dans votre LaTeX.

### Tableaux Prêts à l'Emploi

Le fichier `rapport.tex` contient plusieurs tableaux formatés :
- Caractéristiques du dataset
- Hyperparamètres d'entraînement
- Statistiques du graphe
- Résultats des métriques

Copiez-les directement !

## 📝 Structure du Rapport Auto-généré

1. **Introduction**
   - Contexte de la thèse
   - Objectifs de la démonstration

2. **Méthodologie**
   - Dataset PTB-XL (avec statistiques)
   - Architecture de l'auto-encodeur
   - Procédure d'entraînement

3. **Construction du Graphe**
   - Calcul de similarité
   - Graphe K-NN

4. **Métriques Cognitives**
   - Représentativité
   - Atypicité
   - Résultats

5. **Discussion**
   - Alignement avec la thèse
   - Applications cliniques
   - Extensions possibles

6. **Conclusion**

7. **Bibliographie** (PTB-XL + références psychologie cognitive)

## 🎨 Personnalisation

### Modifier le Template LaTeX

Éditez `generate_latex_report.py` pour :
- Changer le style (article, report, etc.)
- Ajouter des sections
- Modifier les marges, polices, etc.

### Ajouter vos Propres Figures

Sauvegardez vos figures dans `figures/` et référencez-les dans le LaTeX.

## ✅ Checklist Avant Soumission

- [ ] Toutes les figures générées (6 fichiers PNG)
- [ ] `statistics.json` créé
- [ ] `rapport.tex` généré
- [ ] PDF compilé sans erreurs
- [ ] Vérifier que toutes les valeurs sont cohérentes
- [ ] Relire le contenu généré automatiquement

## 💡 Conseils

**Pour la candidature à la thèse** :
- Le rapport auto-généré est un bon point de départ
- Personnalisez-le avec vos propres analyses
- Ajoutez une section sur votre vision de la thèse
- Expliquez pourquoi cette approche vous intéresse

**Pour gagner du temps** :
- Lancez l'entraînement pendant la nuit
- Le notebook génère tout automatiquement
- Vous n'avez qu'à compiler le LaTeX le lendemain !

## 📧 Export pour Email/Présentation

Les figures sont en haute résolution (300 DPI) :
- Parfaites pour un PDF
- Utilisables dans PowerPoint/Beamer
- Prêtes pour publication scientifique

---

**Bonne chance pour votre candidature ! 🎓**