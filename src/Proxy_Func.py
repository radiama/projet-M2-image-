import numpy as np
from Begin_Func import gradient, div, laplacian, norm, process_image_2
from Variational_Func import convolve
from Mix_Func import g_PM
from numpy.fft import fft2, ifft2

# prox(u) = argmin_F(x) = (1/2) * ||x-u||^2 + lambd * g(x)

# Opérateur proximal pour g(x) = |x| (soft-thresholding)
def prox_l1(u, lambd):
    """
    Applique l'opérateur proximal pour g(x) = |x| (soft-thresholding).

    Parameters:
    - u : array_like, entrée à régulariser.
    - lambd : float, paramètre de régularisation.

    Returns:
    - array_like : résultat après application de l'opérateur proximal.

    Raises:
    - ValueError : si lambd est négatif.
    """
    if lambd < 0:
        raise ValueError("Le paramètre lambd doit être positif.")
    return np.sign(u) * np.maximum(np.abs(u) - lambd, 0)


# Opérateur proximal pour g(x) = ||x||^2
def prox_l2(u, lambd):
    """
    Applique l'opérateur proximal pour g(x) = ||x||^2.

    Parameters:
    - u : array_like, entrée à régulariser.
    - lambd : float, paramètre de régularisation.

    Returns:
    - array_like : résultat après application de l'opérateur proximal.

    Raises:
    - ValueError : si lambd est négatif.
    """
    if lambd < 0:
        raise ValueError("Le paramètre lambd doit être positif.")
    return u / (1 + 2 * lambd)


# Opérateur proximal pour g(x) = ||x-f||^2
def prox_l3(u, f, lambd):
    """
    Applique l'opérateur proximal pour g(x) = ||x - f||^2.

    Parameters:
    - u : array_like, entrée à régulariser.
    - f : array_like, cible à atteindre.
    - lambd : float, paramètre de régularisation.

    Returns:
    - array_like : résultat après application de l'opérateur proximal.

    Raises:
    - ValueError : si lambd est négatif ou si u et f ont des dimensions différentes.
    """
    if lambd < 0:
        raise ValueError("Le paramètre lambd doit être positif.")
    if u.shape != f.shape:
        raise ValueError("u et f doivent avoir les mêmes dimensions.")
    return (u + lambd * f) / (1 + lambd)


# Opérateur proximal pour g(x) = ||Ax-f||^2
def prox_l4(u, A, f, lambd):
    """
    Applique l'opérateur proximal pour une régularisation quadratique.

    Résout le système linéaire : (I + lambd * A^T * A)x = u + lambd * A^T * f.

    Parameters:
    - u : array_like, vecteur d'entrée.
    - A : array_like, matrice utilisée dans la régularisation.
    - f : array_like, vecteur cible.
    - lambd : float, paramètre de régularisation.

    Returns:
    - array_like : vecteur après application de l'opérateur proximal.

    Raises:
    - ValueError : si les dimensions sont incompatibles ou si lambd est négatif.
    """
    if lambd <= 0:
        raise ValueError("Le paramètre lambd doit être strictement positif.")
    if A.shape[0] != f.shape[0] or A.shape[1] != u.shape[0]:
        raise ValueError("Les dimensions de A, u, et f doivent être compatibles.")
    
    I = np.eye(A.shape[1])
    J = I + lambd * A.T @ A
    K = u + lambd * A.T @ f
    return np.linalg.solve(J, K)


# # Opérateur proximal pour g(u) = indicatrice de C (projection sur C)
def prox_l5(u, C=np.array([0, 1])):
    """
    Applique une projection sur l'intervalle [C[0], C[1]].

    Parameters:
    - u : array_like, entrée à projeter.
    - C : array_like, intervalle défini par [C[0], C[1]].

    Returns:
    - array_like : résultat après projection.

    Raises:
    - ValueError : si C n'est pas un intervalle valide.
    """
    if C[0] > C[1]:
        raise ValueError("Les bornes de l'intervalle C doivent vérifier C[0] <= C[1].")
    return np.clip(u, C[0], C[1])


# Opérateur proximal pour g(x) = ||grad(x)||_1
def prox_l6(u, lambd, tau, K):
    """
    Applique une régularisation par variation totale (||grad(x)||_1) via une méthode itérative.

    Parameters:
    - u : array_like, image d'entrée.
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de mise à jour.
    - K : int, nombre d'itérations.

    Returns:
    - array_like : image régularisée.

    Raises:
    - ValueError : si lambd, tau, ou K sont invalides.
    """
    if lambd <= 0:
        raise ValueError("Le paramètre lambd doit être strictement positif.")
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    
    z = np.zeros((2, *u.shape))  # Initialisation de la variable duale
    
    for _ in range(K):
        grad_z = -2 * gradient(div(z) + u / lambd)
        z = z - tau * grad_z  # Mise à jour avec normalisation
        
        # Projection sur ||z||_∞ ≤ 1
        norm_z = np.linalg.norm(z, axis=0)
        # Normalisation
        mask = norm_z > 1
        z[:, mask] /= norm_z[mask]

    return u + lambd * div(z)


def prox_l7(u, lambd, tau=0.1):
    """
    Applique l'équation de la chaleur comme opérateur proximal.

    Parameters:
    - u : array_like, entrée à régulariser.
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de mise à jour (par défaut 0.1).

    Returns:
    - array_like : résultat après application de l'opérateur.

    Raises:
    - ValueError : si lambd ou tau sont négatifs.
    """
    if lambd <= 0:
        raise ValueError("Le paramètre lambd doit être strictement positif.")
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    return u + tau * lambd * laplacian(u)


def prox_l8(u, lambd, tau=0.1, alpha=1):
    """
    Applique une régularisation anisotrope basée sur le modèle de Perona-Malik.

    Parameters:
    - u : array_like, image d'entrée.
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de mise à jour (par défaut 0.1).
    - alpha : float, paramètre de contrôle pour la fonction g_PM (par défaut 1).

    Returns:
    - array_like : image régularisée.

    Raises:
    - ValueError : si lambd, tau ou alpha sont invalides.
    """
    if lambd <= 0:
        raise ValueError("Le paramètre lambd doit être strictement positif.")
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if alpha <= 0:
        raise ValueError("Le paramètre alpha doit être strictement positif.")
    
    grad_u = gradient(u)
    norm_grad_u = norm(grad_u)
    anisotropic_diffusion = div(g_PM(norm_grad_u, alpha=alpha) * grad_u)
    return u + tau * lambd * anisotropic_diffusion


def validate_inputs(u, operator_type, operator_params=None, *params):
    """
    Valide les paramètres d'entrée pour les algorithmes de traitement d'image.

    Parameters:
    - u : array_like, donnée d'entrée (image ou signal).
    - operator_type : str, type d'opérateur à utiliser ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire "Mask" (array_like) doit être fourni.
        - Pour "convolution": un noyau de convolution 2D "G" (array_like) doit être fourni.
 
    - *params : float, autres paramètres requis (ex. : lambd, tau, K).

    Raises:
    - ValueError : si un paramètre est invalide ou incompatible.

    Notes:
    - Les paramètres scalaires (dans *params) doivent être strictement positifs.
    - Les dimensions de "Mask" et de l'entrée "u" doivent correspondre.
    - Le noyau de convolution "G" doit être une matrice 2D.

    Examples:
    ---------
    >>> validate_inputs(u, "mask", operator_params={"Mask": mask}, lambd, tau, K)
    >>> validate_inputs(u, "none", operator_params=None, lambd, tau, K)
    """
    # Vérification des paramètres scalaires
    if any(param <= 0 for param in params):
        raise ValueError("Les paramètres scalaires (ex. : lambd, tau, K) doivent être strictement positifs.")

    # Vérification du type d'opérateur
    if operator_type not in ["none", "mask", "convolution"]:
        raise ValueError("Le type d'opérateur doit être 'none', 'mask', ou 'convolution'.")

    # Validation pour l'opérateur 'mask'
    if operator_type == "mask":
        if "Mask" not in operator_params:
            raise ValueError("Pour l'opérateur 'mask', un masque binaire 'Mask' doit être fourni dans 'operator_params'.")
        if operator_params["Mask"].shape != u.shape[:2]:
            raise ValueError("Le masque 'Mask' doit avoir les mêmes dimensions que l'entrée 'u'.")

    # Validation pour l'opérateur 'convolution'
    if operator_type == "convolution":
        if "G" not in operator_params:
            raise ValueError("Pour l'opérateur 'convolution', un noyau 2D 'G' doit être fourni dans 'operator_params'.")
        if operator_params["G"].ndim != 2:
            raise ValueError("Le noyau 'G' doit être une matrice 2D.")


def compute_gradient(u, y, operator_type, operator_params):
    """
    Calcule le gradient pour différents types d'opérateurs.

    Parameters:
    - u : array_like, donnée d'entrée (image ou signal).
    - y : array_like, variable intermédiaire (primal ou dual).
    - operator_type : str, type d'opérateur ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).

    Returns:
    - array_like : le gradient calculé.

    """
    # Dictionnaire des opérateurs
    operators = {
        "none": lambda u, y, _: (y - u),
        "mask": lambda u, y, params: ((y - u) * params["Mask"]),
        "convolution": lambda u, y, params: (convolve(convolve(y, params["G"]) - u, params["G"].T)),
    }

    # Calcul du gradient
    return operators[operator_type](u, y, operator_params)

# 

def forward_backward(u, operator_type, operator_params, lambd, tau, K, prox=prox_l1, prox_params=None, tol=1e-7):
    """
    Algorithme Forward-Backward modulable.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur à utiliser ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de mise à jour (step size).
    K : int, Nombre maximal d'itérations.
    prox : function, Opérateur proximal à utiliser (par défaut `prox_l1`).
    prox_params : dict, optional, Paramètres additionnels pour l'opérateur proximal.
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires à chaque itération.

    Raises:
    ------
    ValueError :
        Si les paramètres sont invalides ou si les dimensions des opérateurs sont incompatibles.

    Notes:
    -----
    Cet algorithme est une méthode d'optimisation basée sur une descente de gradient suivie d'une régularisation via un opérateur proximal.

    Examples:
    ---------
    >>> operator_params = {"Mask": mask}
    >>> x, traj = forward_backward(u, "mask", operator_params, lambd=1.0, tau=0.5, K=50)
    """
    # Validation des entrées
    validate_inputs(u, operator_type, operator_params, lambd, tau, K)

    # Initialisation par défaut pour prox_params
    if prox_params is None:
        prox_params = {}

    # Initialisation des variables
    x = np.zeros_like(u)  # Solution initiale
    trajectoires = [np.copy(x)]  # Stocker la trajectoire des solutions

    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, x, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Mise à jour par descente de gradient
        x_half = x - tau * grad_f

        # Application de l'opérateur proximal
        x = process_image_2(x_half, operator=prox, lambd=lambd, **prox_params)
        trajectoires.append(np.copy(x))  # Sauvegarder la solution intermédiaire

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires


def fista(u, operator_type, operator_params, lambd, tau, K, prox=prox_l6, prox_params=None, tol=1e-7):
    """
    Algorithme FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) modulable.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur à utiliser ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de mise à jour (step size).
    K : int, Nombre maximal d'itérations.
    prox : function, Opérateur proximal à utiliser (par défaut `prox_l6`).
    prox_params : dict, optional, Paramètres additionnels pour l'opérateur proximal.
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires à chaque itération.

    Raises:
    ------
    ValueError :
        Si les paramètres sont invalides ou si les dimensions des opérateurs sont incompatibles.

    Notes:
    -----
    L'algorithme suit la structure de FISTA avec mise à jour accélérée via le paramètre `t`.

    Examples:
    ---------
    >>> operator_params = {"Mask": mask}
    >>> x, traj = fista(u, "mask", operator_params, lambd=1.0, tau=0.5, K=50)
    """
    # Initialisation par défaut pour prox_params
    if prox_params is None:
        prox_params = {}

    # Validation des entrées
    validate_inputs(u, operator_type, operator_params, lambd, tau, K)

    # Initialisation des variables
    y = np.copy(u)  # Variable intermédiaire
    x_old = np.copy(u)  # Solution précédente
    t = 1  # Paramètre d'accélération de FISTA
    trajectoires = [np.copy(u)]  # Stocke les solutions intermédiaires

    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Mise à jour par descente de gradient
        x_half = y - tau * grad_f

        # Application de l'opérateur proximal
        x = process_image_2(x_half, operator=prox, lambd=lambd, **prox_params)

        trajectoires.append(np.copy(x))  # Sauvegarder la solution intermédiaire

        # Mise à jour de FISTA
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        x_old = np.copy(x)

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires

def PGM(u, operator_type, operator_params, lambd, tau, K, tol=1e-7):
    """
    Méthode de gradient projeté modulable.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de gradient.
    K : int, Nombre maximal d'itérations.
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires.

    Raises:
    ------
    ValueError :
        Si les paramètres sont invalides ou si les dimensions des opérateurs sont incompatibles.

    Notes:
    -----
    Cet algorithme applique une descente de gradient suivie d'une projection (opérateur proximal).
    """

    # Validation des paramètres
    validate_inputs(u, operator_type, operator_params, lambd, tau, K)

    # Initialisation
    x = np.zeros_like(u)  # Solution initiale
    trajectoires = [np.copy(x)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, x, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params) / lambd

        # Mise à jour par descente de gradient
        x_half = x - tau * grad_f

        # Application de l'opérateur proximal (projection sur [0, 1])
        x = process_image_2(x_half, operator=prox_l5, C=np.array([0, 1]))
        trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires

def APGM(u, operator_type, operator_params, lambd, tau, K, tol=1e-7):
    """
    Méthode de gradient projeté accéléré modulable (APGM).

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de gradient.
    K : int, Nombre maximal d'itérations.
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires.

    Raises:
    ------
    ValueError :
        Si les paramètres sont invalides ou si les dimensions des opérateurs sont incompatibles.

    Notes:
    -----
    APGM applique l'accélération de Nesterov pour améliorer la convergence par rapport à PGM.
    """

    # Validation des paramètres
    validate_inputs(u, operator_type, operator_params, lambd, tau, K)

    # Initialisation
    x_old = np.copy(u)  # Solution précédente
    y = np.copy(u)  # Variable intermédiaire
    t = 1  # Paramètre d'accélération
    trajectoires = [np.copy(x_old)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params) / lambd

        # Mise à jour par descente de gradient
        x_half = y - tau * grad_f

        # Application de l'opérateur proximal (projection sur [-1, 1])
        x = process_image_2(x_half, operator=prox_l5, C=np.array([-1, 1]))
        trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Mise à jour avec l'accélération de Nesterov
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)

        # Mise à jour des variables
        t = t_new
        x_old = np.copy(x)

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires



def primal_dual_algorithm(u, lambd, tau, sigma, K, prox_primal=prox_l6, prox_primal_params=None, prox_dual=prox_l1, prox_dual_params=None, theta=1.0, tol=1e-7):
    """
    Algorithme primal-dual modulable avec relaxation.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée.
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de gradient pour la régularisation.
    sigma : float, Pas de gradient pour le dual.
    K : int, Nombre total d'itérations.
    prox_primal : function, Opérateur proximal pour la mise à jour primale.
    prox_primal_params : dict, optional, Paramètres pour l'opérateur primal.
    prox_dual : function, Opérateur proximal pour la mise à jour duale.
    prox_dual_params : dict, optional, Paramètres pour l'opérateur dual.
    theta : float, Paramètre de relaxation (par défaut 1.0).
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires.

    Raises:
    ------
    ValueError :
        Si les paramètres sont invalides.

    Notes:
    -----
    Cet algorithme implémente une approche primal-dual avec relaxation pour une convergence accélérée.
    """

    # Validation des paramètres
    if lambd <= 0 or tau <= 0 or sigma <= 0 or K <= 0 or theta <= 0:
        raise ValueError("Tous les paramètres doivent être strictement positifs.")
    prox_primal_params = prox_primal_params or {}
    prox_dual_params = prox_dual_params or {}

    # Initialisation des variables
    x = np.copy(u)  # Variable primale
    y = np.zeros_like(u)  # Variable duale
    x_bar = np.copy(x)  # Variable relaxée
    trajectoires = [np.copy(x)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Mise à jour de y (dual)
        term = y + sigma * x_bar
        y = term - sigma * process_image_2(term / sigma, operator=prox_dual, **prox_dual_params)

        # Mise à jour de x (primal)
        x_new = process_image_2(x - tau * y, operator=prox_primal, lambd=lambd, **prox_primal_params)

        # Relaxation
        x_bar = x_new + theta * (x_new - x)

        # Mise à jour des variables
        x = np.copy(x_new)
        trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Critère de convergence
        if norm(x - x_bar) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires



def ADMM(u, operator_type, operator_params, lambd, tau, rho=1.0, K=100, prox=prox_l6, prox_params=None, tol=1e-7):
    """
    Méthode ADMM (Alternating Direction Method of Multipliers) modulable.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée.
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    lambd : float, Paramètre de régularisation.
    tau : float, Pas de gradient pour la régularisation.
    rho : float, Paramètre du multiplicateur.
    prox : function, Opérateur proximal à utiliser.
    prox_params : dict, optional, Paramètres additionnels pour l'opérateur proximal.
    K : int, Nombre maximal d'itérations.
    tol : float, optional, Tolérance pour le critère de convergence (par défaut 1e-7).

    Returns:
    -------
    tuple :
        - x : array_like, Solution optimisée.
        - trajectoires : list of array_like, Liste des solutions intermédiaires.

    Raises:
    ------
    ValueError :
        Si les paramètres ou les dimensions sont invalides.

    Notes:
    -----
    ADMM est une méthode itérative pour résoudre des problèmes d'optimisation impliquant des régularisations et des contraintes.

    Examples:
    ---------
    >>> operator_params = {"Mask": mask}
    >>> x, traj = ADMM(u, "mask", operator_params, lambd=0.1, tau=0.01, rho=1.0, K=50)
    """

    # Validation des paramètres
    validate_inputs(u, operator_type, operator_params, lambd, tau, rho, K)

    # Initialisation par défaut pour prox_params
    if prox_params is None:
        prox_params = {}

    # Initialisation des variables
    x = np.zeros_like(u)  # Variable primale
    y = np.zeros_like(u)  # Variable intermédiaire
    z = np.zeros_like(u)  # Variable duale
    trajectoires = [np.copy(x)]  # Stocker les trajectoires

    # Fonction pour calculer y (variable intermédiaire)
    def compute_gradient_2(u, x, z, operator_type, operator_params, rho):
        operators = {
            "none": lambda u, x, z, _: (u + rho * x + z) / (1 + rho),
            "mask": lambda u, x, z, params: (params["Mask"] * u + rho * x + z) / (params["Mask"] + rho),
            "convolution": lambda u, x, z, params: np.linalg.solve(convolve(params["G"], params["G"].T) + rho * np.eye(u.shape[0]), convolve(u, params["G"]) + rho * x - z),
        }
        if operator_type not in operators:
            raise ValueError(f"Type d'opérateur '{operator_type}' non supporté.")
        return operators[operator_type](u, x, z, operator_params)

    # Boucle principale d'optimisation
    for k in range(K):
        # Mise à jour de y selon l'opérateur
        y = process_image_2(u, x, z, operator=compute_gradient_2, operator_type=operator_type, operator_params=operator_params, rho=rho)

        # Mise à jour de x avec l'opérateur proximal
        x = process_image_2(y - z / rho, operator=prox, lambd=lambd / rho, **prox_params)
        trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Mise à jour de la variable duale z
        z += rho * (x - y)

        # Critère de convergence
        if norm(x - y) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires
