# Code borrowed from Alexis Goujon https://https://github.com/axgoujon/convex_ridge_regularizers

# Bibliothèques standards
import json  # Manipulation de fichiers JSON pour la configuration ou les données structurées.
import os  # Gestion des chemins de fichiers et des opérations système (ex. vérifications, création de dossiers).
import glob  # Recherche de fichiers selon des motifs (wildcards).
import sys  # Gestion des paramètres du script et des modules système.
import math  # Fonctions mathématiques de base (ex. sqrt, log).

# Bibliothèque pour les calculs numériques
import numpy as np  # Manipulation de tableaux et calculs mathématiques avancés.

# PyTorch : Framework pour le Deep Learning
import torch  # Outils pour le calcul tensoriel et l'entraînement de modèles.
from torchmetrics.functional import peak_signal_noise_ratio as psnr  # Fonction pour calculer le PSNR (rapport signal-bruit de pic).
from torchmetrics.functional import structural_similarity_index_measure as ssim  # Fonction pour calculer le SSIM (mesure de similarité structurelle).


from tqdm import tqdm  # Barre de progression pour suivre l'exécution des boucles.
from Convex_ridge_regularizer_rev import ConvexRidgeRegularizer  # Importe la classe principale pour construire le modèle.
from pathlib import Path  # Outils pour manipuler facilement les chemins de fichiers et dossiers.

def load_model(name, device='cuda:0', epoch=None):
    """
    Charge un modèle préentraîné à partir d'un checkpoint.

    Args:
        name (str): Nom du modèle (correspond au sous-dossier dans 'trained_models').
        device (str): Périphérique où charger le modèle ('cuda:0' par défaut).
        epoch (int, optional): Époque spécifique du checkpoint. Si None, charge le dernier.

    Returns:
        model: Modèle chargé en mode évaluation.
    """
    # Dossiers du modèle et des checkpoints
    directory = f'{os.path.abspath(os.path.dirname(__file__))}/../trained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    # Récupère le dernier checkpoint si aucune époque n'est spécifiée
    if epoch is None:
        files = glob.glob(f'{directory}checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)
        print(f"--- Chargement du checkpoint de l'époque {epoch} ---")

    # Chemin du checkpoint
    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'

    # Chargement du fichier de configuration
    config = json.load(open(f'{directory}config.json'))

    # Construction du modèle selon la configuration
    model, _ = build_model(config)

    # Chargement du checkpoint
    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0': device, 'cuda:1': device, 'cuda:2': device, 'cuda:3': device})

    # Transfert du modèle sur le périphérique
    model.to(device)

    # Chargement des poids sauvegardés
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Mode évaluation (désactive la propagation des gradients)

    return model

def build_model(config):
    """
    Construit un modèle basé sur les paramètres spécifiés dans le fichier de configuration.

    Args:
        config (dict): Dictionnaire contenant les paramètres du modèle.

    Returns:
        model: Modèle construit selon la configuration.
        config (dict): Configuration utilisée pour construire le modèle.
    """
    # Construction du modèle avec les paramètres spécifiés
    model = ConvexRidgeRegularizer(
        kernel_size=config['net_params']['kernel_size'],
        channels=config['net_params']['channels'],
        activation_params=config['activation_params']
    )

    return model, config

def accelerated_gd(x_noisy, model, ada_restart=False, lmbd=1, mu=1, use_strong_convexity=False, **kwargs):
    """
    Résout l'opérateur proximal en utilisant la règle accélérée FISTA.

    Args:
        x_noisy (torch.Tensor): Entrée bruitée (image ou données).
        model: Modèle utilisé pour calculer le gradient.
        ada_restart (bool): Active ou désactive le redémarrage adaptatif (par défaut False).
        lmbd (float): Paramètre de régularisation lambda.
        mu (float): Paramètre d'échelle.
        use_strong_convexity (bool): Si True, utilise une variante pour les fonctions fortement convexes.
        kwargs: Autres arguments optionnels (max_iter, tol).

    Returns:
        torch.Tensor: Solution obtenue après optimisation.
        int: Nombre d'itérations réalisées.
        int: Nombre de redémarrages adaptatifs effectués.
    """

    # Paramètres par défaut
    max_iter = kwargs.get('max_iter', 500)  # Nombre maximal d'itérations
    tol = kwargs.get('tol', 1e-4)  # Tolérance pour le critère d'arrêt

    # Initialisation
    x = torch.clone(x_noisy)  # Solution actuelle
    z = torch.clone(x_noisy)  # Variable intermédiaire accélérée
    t = 1  # Facteur d'accélération

    L = model.L  # Constante de Lipschitz
    alpha = 1 / (1 + mu * lmbd * L)  # Pas d'apprentissage
    n_restart = 0  # Compteur de redémarrages adaptatifs

    # Boucle principale
    for i in range(max_iter):
        # Sauvegarde de l'état précédent
        x_old = torch.clone(x)

        # Calcul du gradient pour la mise à jour de z
        grad_z = alpha * ((z - x_noisy) + lmbd * model(mu * z))
        x = z - grad_z  # Mise à jour de x avec la descente de gradient

        # Accélération avec la méthode de Nesterov
        t_old = t
        t = 0.5 * (1 + math.sqrt(1 + 4 * t**2))  # Mise à jour du paramètre d'accélération
        if use_strong_convexity:
            gamma = (1 - math.sqrt(alpha)) / (1 + math.sqrt(alpha))  # Variante pour convexité forte
        else:
            gamma = (t_old - 1) / t
        z = x + gamma * (x - x_old)  # Mise à jour accélérée de z

        # Redémarrage adaptatif
        if ada_restart:
            if (torch.sum(grad_z * (x - x_old)) > 0):  # Vérifie si la condition de redémarrage est remplie
                t = 1  # Réinitialise les paramètres d'accélération
                z = torch.clone(x_old)
                x = torch.clone(x_old)
                n_restart += 1  # Incrémente le compteur de redémarrages
            else:
                # Critère d'arrêt basé sur la norme relative
                res = (torch.norm(x_old - x) / torch.norm(x_old)).item()
                if res < tol:  # Arrête si le changement relatif est inférieur à tolérance
                    break

    return x, i, n_restart

def tStepDenoiser(model, x_noisy, t_steps=50):
    """
    Implémente un débruitage itératif basé sur un modèle donné.

    Args:
        model: Modèle utilisé pour calculer les étapes de débruitage.
        x_noisy (torch.Tensor): Image ou données bruitées.
        t_steps (int): Nombre d'itérations (par défaut 50).

    Returns:
        torch.Tensor: Résultat après t étapes de débruitage.
    """

    # Paramètres d'échelle et de régularisation (transférables)
    lmbd = model.lmbd_transformed
    mu = model.mu_transformed

    # Borne de Lipschitz du modèle
    if model.training:
        # Estimation différentiable de la borne de Lipschitz (L)
        L = torch.clip(model.precise_lipschitz_bound(n_iter=2, differentiable=True), 0.1, None)
        # (Le clipping de L améliore la stabilité pour les petites valeurs de L, comme L = 0 à l'initialisation)

        # Stockage pour utilisation en mode évaluation
        model.L.data = L
    else:
        # Utilise la valeur stockée pour éviter le recalcul
        L = model.L

    # Initialisation des résultats avec l'entrée bruitée
    x = torch.clone(x_noisy)

    # Boucle principale : t étapes différentiables
    for i in range(t_steps):
        opt = 1  # Choix du schéma d'étape

        # Étape 1 : Initialisation
        if i == 0:
            # Étape de descente de gradient avec un pas 2/L
            x = x_noisy - 2 / (L * mu) * model(mu * x_noisy)
        else:
            # Étapes suivantes : choix du pas
            if opt == 1:
                step_size = (2 - 1e-8) / (1 + L * lmbd * mu)  # Règle par défaut
            else:
                step_size = 2 / (2 + L * lmbd * mu)  # Alternative (moins utilisée)

            # Mise à jour de x
            x = x - step_size * ((x - x_noisy) + lmbd * model(mu * x))

    return x


def AdaGD(x_noisy, model, lmbd=1, mu=1, **kwargs):
    """
    Débruitage avec CRR-NNs en utilisant un schéma de descente de gradient adaptatif.

    Args:
        x_noisy (torch.Tensor): Données bruitées servant de point de départ.
        model: Modèle utilisé pour calculer le gradient.
        lmbd (float): Paramètre de régularisation (lambda).
        mu (float): Paramètre d'échelle.
        kwargs: Arguments supplémentaires pour configurer l'optimisation (max_iter, tol).

    Returns:
        torch.Tensor: Données débruitées après optimisation.
        int: Nombre d'itérations effectuées.
    """

    # Paramètres par défaut
    max_iter = kwargs.get('max_iter', 200)  # Nombre maximal d'itérations
    tol = kwargs.get('tol', 1e-6)  # Tolérance pour critère d'arrêt

    def grad_denoising(x):
        """
        Calcule le gradient pour le débruitage.

        Args:
            x (torch.Tensor): Entrée actuelle.

        Returns:
            torch.Tensor: Gradient du débruitage.
        """
        return (x - x_noisy) + lmbd * model(mu * x)

    # Initialisation des variables
    x_old = torch.clone(x_noisy)  # Valeur initiale
    alpha = 1e-5  # Pas initial
    grad = grad_denoising(x_old)  # Gradient initial
    x = x_old - alpha * grad  # Première mise à jour

    # Si le gradient est nul, retourne directement l'entrée
    if grad.norm() == 0:
        return x, 0

    # Variable pour l'accélération adaptative
    theta = float('inf')

    # Boucle principale pour optimiser
    for i in range(max_iter):
        grad_old = torch.clone(grad)  # Sauvegarde du gradient précédent
        grad = grad_denoising(x)  # Calcul du gradient actuel

        # Calcul du pas adaptatif
        alpha_1 = (torch.norm(x - x_old) / (1e-10 + torch.norm(grad - grad_old))).item() / 2
        alpha_2 = math.sqrt(1 + theta) * alpha  # Accélération
        alpha_old = alpha  # Sauvegarde de l'ancien pas
        alpha = min(alpha_1, alpha_2)  # Choix du pas optimal

        x_old = torch.clone(x)  # Mise à jour de x_old pour la prochaine itération
        x = x - alpha * grad  # Mise à jour de la solution

        theta = alpha / (alpha_old + 1e-10)  # Mise à jour pour l'accélération adaptative

        # Critère d'arrêt basé sur la norme relative
        res = (torch.norm(x_old - x) / torch.norm(x_old)).item()
        if res < tol:
            break

    return x, i


def AdaGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """
    Résout un problème inverse avec CRR-NNs en utilisant une descente de gradient adaptative.

    Args:
        y (torch.Tensor): Observation ou données mesurées.
        model: Modèle pour calculer le gradient.
        lmbd (float): Paramètre de régularisation lambda.
        mu (float): Paramètre d'échelle.
        H (callable): Opérateur direct (par exemple, matrice de flou ou système de mesure).
        Ht (callable): Transposé de l'opérateur direct.
        op_norm (float): Norme de l'opérateur direct.
        x_gt (torch.Tensor, optional): Données de vérité terrain pour évaluer les métriques (PSNR, SSIM).
        kwargs: Arguments supplémentaires comme `max_iter` ou `tol`.

    Returns:
        torch.Tensor: Solution reconstruite.
        float: Dernière valeur du PSNR (si `x_gt` est fourni).
        float: Dernière valeur du SSIM (si `x_gt` est fourni).
        int: Nombre d'itérations effectuées.
    """

    # Paramètres d'optimisation
    max_iter = kwargs.get('max_iter', 1000)  # Nombre maximal d'itérations
    tol = kwargs.get('tol', 1e-6)  # Tolérance pour critère d'arrêt
    enforce_positivity = kwargs.get('enforce_positivity', True)  # Impose la positivité des solutions

    def grad_func(x):
        """
        Calcule le gradient pour la mise à jour.

        Args:
            x (torch.Tensor): Estimation actuelle.

        Returns:
            torch.Tensor: Gradient calculé.
        """
        return (Ht(H(x) - y) / op_norm**2 + lmbd * model(mu * x))

    # Initialisation
    alpha = 1e-5  # Pas initial
    x_old = torch.zeros_like(Ht(y))  # Initialisation avec un tensor nul
    grad = grad_func(x_old)  # Gradient initial
    x = x_old - alpha * grad  # Première mise à jour

    pbar = tqdm(range(max_iter))  # Barre de progression

    theta = float('inf')  # Variable pour l'accélération adaptative

    # Boucle principale d'optimisation
    for i in pbar:
        grad_old = grad.clone()  # Sauvegarde du gradient précédent
        grad = grad_func(x)  # Calcul du gradient actuel

        # Mise à jour adaptative du pas
        alpha_1 = (torch.norm(x - x_old) / (torch.norm(grad - grad_old) + 1e-10)).item() / 2
        alpha_2 = math.sqrt(1 + theta) * alpha
        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        # Mise à jour de la solution
        x_old = x.clone()
        x = x - alpha * grad

        # Imposition de la positivité (si activée)
        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        # Mise à jour pour l'accélération adaptative
        theta = alpha / alpha_old

        # Critère d'arrêt basé sur la norme relative
        res = (torch.norm(x_old - x) / torch.norm(x_old)).item()

        # Calcul des métriques si la vérité terrain est fournie
        if x_gt is not None:
            psnr_ = psnr(x, x_gt, data_range=1)
            ssim_ = ssim(x, x_gt)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_, ssim_ = None, None
            pbar.set_description(f"psnr: {psnr_} | res: {res:.2e}")

        # Arrêt si la tolérance est atteinte
        if res < tol:
            break

    return x, psnr_, ssim_, i

def AdaAGD_Recon(y, model, lmbd=1, mu=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """
    Résout un problème inverse avec CRR-NNs en utilisant une descente de gradient adaptative accélérée.

    Args:
        y (torch.Tensor): Observation ou données mesurées.
        model: Modèle pour calculer le gradient.
        lmbd (float): Paramètre de régularisation lambda.
        mu (float): Paramètre d'échelle.
        H (callable): Opérateur direct (ex. flou ou mesure).
        Ht (callable): Transposé de l'opérateur direct.
        op_norm (float): Norme de l'opérateur direct.
        x_gt (torch.Tensor, optional): Vérité terrain pour évaluer les métriques (PSNR, SSIM).
        kwargs: Arguments supplémentaires comme `max_iter` ou `tol`.

    Returns:
        torch.Tensor: Solution reconstruite.
        float: Dernière valeur du PSNR (si `x_gt` est fourni).
        float: Dernière valeur du SSIM (si `x_gt` est fourni).
        int: Nombre d'itérations effectuées.
    """

    # Paramètres d'optimisation
    max_iter = kwargs.get('max_iter', 1000)  # Nombre maximal d'itérations
    tol = kwargs.get('tol', 1e-6)  # Tolérance pour critère d'arrêt
    enforce_positivity = kwargs.get('enforce_positivity', True)  # Impose la positivité des solutions

    def grad_func(x):
        """
        Calcule le gradient pour la mise à jour.

        Args:
            x (torch.Tensor): Estimation actuelle.

        Returns:
            torch.Tensor: Gradient calculé.
        """
        return (Ht(H(x) - y) / op_norm**2 + lmbd * model(mu * x))

    # Initialisation
    alpha, beta = 1e-5, 1e-5  # Pas initiaux
    x_old = torch.zeros_like(Ht(y))  # Estimation initiale (tensor nul)
    grad = grad_func(x_old)  # Gradient initial
    x = x_old - alpha * grad  # Première mise à jour
    z = x.clone()  # Variable accélérée

    pbar = tqdm(range(max_iter))  # Barre de progression

    theta, Theta = float('inf'), float('inf')  # Variables pour l'accélération adaptative

    # Boucle principale d'optimisation
    for i in pbar:
        grad_old = grad.clone()  # Sauvegarde du gradient précédent
        grad = grad_func(x)  # Calcul du gradient actuel

        # Calcul des pas adaptatifs
        alpha_1 = (torch.norm(x - x_old) / (torch.norm(grad - grad_old) + 1e-10)).item() / 2
        alpha_2 = math.sqrt(1 + theta / 2) * alpha
        alpha_old = alpha
        alpha = min(alpha_1, alpha_2)

        beta_1 = 1 / (4 * alpha_1)
        beta_2 = math.sqrt(1 + Theta / 2) * beta
        beta_old = beta
        beta = min(beta_1, beta_2)

        # Calcul de l'accélération
        gamma = (1 / math.sqrt(alpha) - math.sqrt(beta)) / (1 / math.sqrt(alpha) + math.sqrt(beta))

        # Mise à jour des variables
        z_old = z.clone()
        z = x - alpha * grad
        x_old = x.clone()
        x = z + gamma * (z - z_old)

        # Imposition de la positivité (si activée)
        if enforce_positivity:
            x = torch.clamp(x, 0, None)

        # Mise à jour des paramètres d'accélération
        theta = alpha / alpha_old
        Theta = beta / beta_old

        # Critère d'arrêt basé sur la norme relative
        res = (torch.norm(x_old - x) / torch.norm(x_old)).item()

        # Calcul des métriques si la vérité terrain est fournie
        if x_gt is not None:
            psnr_ = psnr(x, x_gt, data_range=1)
            ssim_ = ssim(x, x_gt)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_, ssim_ = None, None
            pbar.set_description(f"res: {res:.2e}")

        # Arrêt si la tolérance est atteinte
        if res < tol:
            break

    return x, psnr_, ssim_, i