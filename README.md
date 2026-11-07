# Détection d'images deepfake — INF6243 (UQO, automne 2026)

Projet de session du cours **INF6243 Techniques d'apprentissage** (Université du Québec en Outaouais).

**Objectif :** distinguer des visages réels de visages générés ou modifiés par IA, et comparer plusieurs classificateurs.

- **Données :** [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) (Kaggle), jeu complet (≈ 140 000 images d'entraînement, 39 000 de validation, 11 000 de test), partitions officielles conservées.
- **Caractéristiques :** embeddings ResNet-50 (2048 dimensions) + 13 caractéristiques interprétables (couleur, netteté, contours, énergie spectrale).
- **Modèles :** régression logistique, SVM RBF (Nyström), perceptron multicouche, CNN entraîné de zéro, EfficientNet-B0 affiné.
- **Méthode :** GridSearch, validation croisée k-fold, matrices de confusion, courbes d'apprentissage, importance des attributs ; le jeu de test n'est utilisé qu'une fois, après la sélection du modèle.

## Exécution

Ouvrir `Code/notebook_principal.ipynb` dans Google Colab (GPU recommandé), puis *Exécution → Tout exécuter*.
Le jeu de données est téléchargé avec `kagglehub` ; les résultats intermédiaires sont mis en cache dans Google Drive (reprise automatique).
