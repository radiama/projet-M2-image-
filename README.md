# Apprentissage de fonctions de régularisation convexes pour la résolution de problèmes inverses

## Objectifs

- Comprendre les méthodes Plug-and-Play (PnP).
- Développer une fonction de régularisation \( R(x) \) paramétrée par un réseau de neurones.
- Implémenter et comparer ces approches aux méthodes classiques, notamment la Variation Totale (TV).

## Partie I : Approches variationnelles et Plug-and-Play pour la résolution de problèmes inverses

### Travaux principaux :
1. Rédiger une introduction aux problèmes inverses.
2. Présenter les méthodes Plug-and-Play (PnP).
3. Comparer la méthode de Variation Totale (TV) et les approches PnP pour différents opérateurs linéaires.

## Partie II : Apprentissage de fonctions de régularisation convexes pour la résolution de problèmes inverses

### Travaux principaux :
- Étudier l'article d'Alexis Goujon sur l'apprentissage de fonctions de régularisation convexes appliquées aux problèmes inverses.

## Bibliothèques principales utilisées
- Python
- Pandas
- Matplotlib
- PyTorch
- DeepInverse
- et d'autres...

## Modèles de réseaux de neurones utilisés

- **DRUNet** : Modèle préentraîné par les concepteurs de DeepInverse.
- **CRRNN** : Modèle préentraîné proposé par Alexis Goujon.
- **CRRNN personnalisé** : Modèle entraîné pour des tâches spécifiques de débruitage, de défloutage et d’inpainting.

## Principaux résultats

- **DRUNet (Denoising Residual U-Net)** s'avère être la méthode la plus performante pour les tâches de débruitage.
- **CRRNN (Convex Ridge Regularization Neural Network)** et **TV (Total Variation)** sont plus adaptés aux tâches de défloutage.
- **DRUNet** présente l'avantage de gérer simultanément des doubles dégradations, comme le bruit et un masque, mais son coût computationnel reste élevé.

## Tutoriels

Un tutoriel détaillé expliquant notre démarche est disponible dans le dossier **test**.
