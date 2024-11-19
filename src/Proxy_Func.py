import numpy as np
from Begin_Func import gradient, div, laplacian, norm, process_image_2
from Variational_Func import convolve
from Mix_Func import g_PM

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
        z = np.clip(z, -1, 1)
    
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


def forward_backward(u, lambd, tau, K, G=None, Mask=None, tol=1e-7, prox=prox_l1, prox_params=None):
    """
    Algorithme Forward-Backward modulable

    Parameters:
    - u : array_like, donnée d'entrée (image à traiter, niveau de gris ou couleur).
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de mise à jour.
    - K : int, nombre d'itérations.
    - G : array_like, noyau de convolution (optionnel, pour convolutions).
    - Mask : array_like, masque binaire (optionnel).
    - tol : float, tolérance pour le critère de convergence.
    - prox : function, opérateur proximal à utiliser.
    - prox_params : dict, paramètres additionnels pour l'opérateur proximal.

    Returns:
    - array_like : solution optimisée (image traitée, niveau de gris ou couleur).

    Raises:
    - ValueError : si les paramètres ou les dimensions sont invalides.
    """
    if lambd <= 0 or tau <= 0 or K <= 0:
        raise ValueError("Les paramètres lambd, tau et K doivent être strictement positifs.")

    if Mask is not None and Mask.shape != u.shape:
        raise ValueError("Le masque Mask doit avoir les mêmes dimensions que u.")
    
    if G is None or G.ndim != 2:
                raise ValueError("Pour l'opérateur 'convolution', G doit être fourni et être une matrice 2D.")

    if prox_params is None:
        prox_params = {}

    # Initialisation
    x = np.zeros_like(u)

    for _ in range(K):
        if G is not None:
            # Gestion de la convolution via process_image_2
            grad_f = process_image_2(x, u, operator= lambda img, img2: convolve(convolve(img, G) - img2, G.T))
        elif Mask is not None:
            # Gradient pondéré par Mask
            grad_f = (x - u) * Mask
        else:
            # Gradient simple
            grad_f = x - u

        # Descente de gradient
        x_half = x - tau * grad_f

        # Application de l'opérateur proximal via process_image_2
        x = process_image_2(x_half, operator= prox, lambd=lambd, **prox_params)

        # Critère de convergence
        if norm(x - x_half) < tol:
            break

    return x



def fista(u, operator_type, operator_params, lambd, tau, K, prox=prox_l6, prox_params={"tau": 0.01, "K": 50}, tol=1e-7):
    """
    Algorithme FISTA modulable.

    Parameters:
    - u : array_like, donnée d'entrée (image ou signal).
    - operator_type : str, type d'opérateur ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": Mask (array_like, masque de pondération).
        - Pour "convolution": G (array_like, noyau de convolution).
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de mise à jour.
    - prox : function, opérateur proximal à utiliser (par défaut `prox_l6`).
    - prox_params : dict, paramètres additionnels pour l'opérateur proximal.
    - K : int, nombre d'itérations globales.
    - tol : float, tolérance pour le critère de convergence.

    Returns:
    - array_like : solution optimisée.

    Raises:
    - ValueError : si les paramètres sont invalides ou si les dimensions des opérateurs sont incompatibles.
    """
    # Validations générales
    if lambd <= 0 or tau <= 0 or K <= 0:
        raise ValueError("Les paramètres lambd, tau et K doivent être strictement positifs.")
    if operator_type not in ["none", "mask", "convolution"]:
        raise ValueError("Le type d'opérateur doit être 'none', 'mask', ou 'convolution'.")
    if prox_params is None:
        prox_params = {}

    # Initialisation
    y = process_image_2(u, operator= np.copy)
    x_old = process_image_2(u, operator= np.copy)
    t = 1  # Paramètre d'accélération

    # Fonction pour gérer les différents types d'opérateurs
    def compute_gradient(u, y, operator_type, operator_params):
        if operator_type == "none":
            return y - u
        elif operator_type == "mask":
            Mask = operator_params.get("Mask", None)
            if Mask is None or Mask.shape != u.shape:
                raise ValueError("Pour l'opérateur 'mask', Mask doit être fourni et avoir les mêmes dimensions que u.")
            return (y - u) * Mask
        elif operator_type == "convolution":
            G = operator_params.get("G", None)
            if G is None or G.ndim != 2 or y.ndim != 2:
                raise ValueError("Pour l'opérateur 'convolution', G doit être fourni et être une matrice 2D.")
            return convolve(convolve(y, G) - u, G.T)

    for _ in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator= compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Descente de gradient
        x_half = y - tau * grad_f

        # Application de l'opérateur proximal
        x = process_image_2(x_half, operator= prox, lambd=lambd, **prox_params)

        # Mise à jour FISTA
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        x_old = process_image_2(x, operator= np.copy)

        # Critère de convergence
        if norm(x - x_half) < tol:
            break

    return x


def PGM(u, operator_type, operator_params, lambd, tau, K, tol=1e-7):
    """
    Méthode de gradient projeté modulable.

    Parameters:
    - u : array_like, donnée d'entrée (image ou signal).
    - operator_type : str, type d'opérateur ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": Mask (array_like, masque de pondération).
        - Pour "convolution": G (array_like, noyau de convolution).
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de gradient.
    - K : int, nombre d'itérations.
    - tol : float, tolérance pour le critère de convergence.

    Returns:
    - array_like : solution optimisée.

    Raises:
    - ValueError : si les paramètres ou les dimensions sont invalides.
    """
    # Validation des paramètres
    if lambd <= 0 or tau <= 0 or K <= 0:
        raise ValueError("Les paramètres lambd, tau et K doivent être strictement positifs.")
    if operator_type not in ["none", "mask", "convolution"]:
        raise ValueError("Le type d'opérateur doit être 'none', 'mask', ou 'convolution'.")

    # Initialisation
    x = process_image_2(u, operator= np.zeros_like)

    # Fonction pour gérer les gradients en fonction de l'opérateur
    def compute_gradient(u, x, operator_type, operator_params, lambd):
        if operator_type == "none":
            return (x - u) / lambd
        elif operator_type == "mask":
            Mask = operator_params.get("Mask", None)
            if Mask is None or Mask.shape != u.shape:
                raise ValueError("Pour l'opérateur 'mask', Mask doit être fourni et avoir les mêmes dimensions que u.")
            return (x - u) * Mask / lambd
        elif operator_type == "convolution":
            G = operator_params.get("G", None)
            if G is None or u.ndim != 2 or G.ndim != 2:
                raise ValueError("Pour l'opérateur 'convolution', G doit être fourni et être une matrice 2D.")
            return convolve(convolve(x, G) - u, G.T) / lambd

    # Boucle d'optimisation
    for _ in range(K):
        grad_f = process_image_2(u, x, operator= compute_gradient, operator_type=operator_type, operator_params=operator_params, lambd=lambd)
        
        # Descente de gradient
        x_half = x - tau * grad_f

        # Application de l'opérateur proximal (projection sur un intervalle)
        x = process_image_2(x_half, operator= prox_l5, C=np.array([0, 1]))

        # Critère de convergence
        if norm(x - x_half) < tol:
            break

    return x

def APGM(u, operator_type, operator_params, lambd, tau, K, tol=1e-7):
    """
    Méthode de gradient projeté accéléré modulable (APGM).

    Parameters:
    - u : array_like, donnée d'entrée (image ou signal).
    - operator_type : str, type d'opérateur ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": Mask (array_like, masque de pondération).
        - Pour "convolution": G (array_like, noyau de convolution).
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de gradient.
    - K : int, nombre d'itérations.
    - tol : float, tolérance pour le critère de convergence.

    Returns:
    - array_like : solution optimisée.

    Raises:
    - ValueError : si les paramètres ou les dimensions sont invalides.
    """
    # Validation des paramètres
    if lambd <= 0 or tau <= 0 or K <= 0:
        raise ValueError("Les paramètres lambd, tau et K doivent être strictement positifs.")
    if operator_type not in ["none", "mask", "convolution"]:
        raise ValueError("Le type d'opérateur doit être 'none', 'mask', ou 'convolution'.")

    # Initialisation
    x_old = process_image_2(u, operator= np.copy)
    y = process_image_2(u, operator= np.copy)
    t = 1  # Paramètre d'accélération

    # Fonction pour calculer le gradient en fonction de l'opérateur
    def compute_gradient(u, y, operator_type, operator_params, lambd):
        if operator_type == "none":
            return (y - u) / lambd
        elif operator_type == "mask":
            Mask = operator_params.get("Mask", None)
            if Mask is None or Mask.shape != u.shape:
                raise ValueError("Pour l'opérateur 'mask', Mask doit être fourni et avoir les mêmes dimensions que u.")
            return ((y - u) * Mask) / lambd
        elif operator_type == "convolution":
            G = operator_params.get("G", None)
            if G is None or u.ndim != 2 or G.ndim != 2:
                raise ValueError("Pour l'opérateur 'convolution', G doit être fourni et être une matrice 2D.")
            return convolve(convolve(y, G) - u, G.T) / lambd

    # Boucle d'optimisation
    for _ in range(K):
        grad_f = process_image_2(u, y, operator= compute_gradient, operator_type=operator_type, operator_params=operator_params, lambd=lambd)

        # Descente de gradient
        x_half = y - tau * grad_f

        # Application de l'opérateur proximal
        x = process_image_2(x_half, operator= prox_l5, C=np.array([-1, 1]))

        # Accélération de Nesterov
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
        x_old = process_image_2(x, operator= np.copy)

        # Critère de convergence
        if norm(x - x_half) < tol:
            break

    return x


def primal_dual_algorithm(u, lambd, tau, sigma, K, 
                          prox_primal=prox_l6, prox_primal_params={"tau": 0.01, "K": 50}, 
                          prox_dual=prox_l1, prox_dual_params={"lambd": 1.0}, 
                          theta=1.0, tol=1e-7):
    """
    Algorithme primal-dual modulable avec relaxation.

    Parameters:
    - u : array_like, donnée d'entrée.
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de gradient pour la régularisation.
    - sigma : float, pas de gradient pour le dual.
    - K : int, nombre total d'itérations.
    - prox_primal : function, opérateur proximal pour la mise à jour primale.
    - prox_primal_params : dict, paramètres pour l'opérateur primal.
    - prox_dual : function, opérateur proximal pour la mise à jour duale.
    - prox_dual_params : dict, paramètres pour l'opérateur dual.
    - theta : float, paramètre de relaxation (par défaut 1.0).
    - tol : float, tolérance pour le critère de convergence.

    Returns:
    - array_like : solution optimisée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    # Validation des paramètres
    if lambd <= 0 or tau <= 0 or sigma <= 0 or K <= 0 or theta <= 0:
        raise ValueError("Tous les paramètres doivent être strictement positifs.")
    if prox_primal_params is None:
        prox_primal_params = {}
    if prox_dual_params is None:
        prox_dual_params = {}

    # Initialisation des variables
    x = process_image_2(u, operator= np.copy)  # Variable primale
    y = process_image_2(u, operator= lambda img: np.zeros_like(img))  # Variable duale
    x_bar = process_image_2(x, operator= np.copy)  # Variable relaxée

    for _ in range(K):
        # Mise à jour de y (dual)
        term = y + sigma * x_bar
        y = term - sigma * process_image_2(term / sigma, operator= prox_dual, **prox_dual_params)

        # Mise à jour de x (primal)
        x_new = process_image_2(x, operator= prox_primal, lambd=lambd, **prox_primal_params)

        # Relaxation
        x_bar = x_new + theta * (x_new - x)

        # Mise à jour des variables
        x = process_image_2(x_new, operator= np.copy)
    
        # Critère de convergence
        if norm(x - y) < tol:
            break

    return x


def ADMM(u, operator_type, operator_params, lambd, tau, rho=1.0, K=100, 
         prox=prox_l6, prox_params={"tau": 0.01, "K": 50}, tol=1e-7):
    """
    Méthode ADMM modulable.

    Parameters:
    - u : array_like, donnée d'entrée.
    - operator_type : str, type d'opérateur ("none", "mask", "convolution").
    - operator_params : dict, paramètres spécifiques à l'opérateur.
        - Pour "mask": M (array_like, masque binaire).
        - Pour "convolution": G (array_like, noyau de convolution).
    - lambd : float, paramètre de régularisation.
    - tau : float, pas de gradient pour la régularisation.
    - rho : float, paramètre du multiplicateur.
    - prox : function, opérateur proximal à utiliser.
    - prox_params : dict, paramètres additionnels pour l'opérateur proximal.
    - K : int, nombre d'itérations globales.
    - tol : float, tolérance pour le critère de convergence.

    Returns:
    - array_like : solution optimisée.

    Raises:
    - ValueError : si les paramètres ou les dimensions sont invalides.
    """
    # Validation des paramètres
    if lambd <= 0 or tau <= 0 or rho <= 0 or K <= 0:
        raise ValueError("Tous les paramètres doivent être strictement positifs.")
    if operator_type not in ["none", "mask", "convolution"]:
        raise ValueError("Le type d'opérateur doit être 'none', 'mask', ou 'convolution'.")
    if prox_params is None:
        prox_params = {}

    # Initialisation des variables
    x = process_image_2(u, operator= np.zeros_like)
    y = process_image_2(u, operator= np.zeros_like)
    z = process_image_2(u, operator= np.zeros_like)

    # Fonction pour gérer les différents types d'opérateurs
    def compute_y_update(u, x, z, operator_type, operator_params, rho):
        if operator_type == "none":
            return (u + rho * x + z) / (1 + rho)
        elif operator_type == "mask":
            M = operator_params.get("M", None)
            if M is None or M.shape != u.shape:
                raise ValueError("Pour l'opérateur 'mask', M doit être fourni et avoir les mêmes dimensions que u.")
            return (M * u + rho * x + z) / (M + rho)
        elif operator_type == "convolution":
            G = operator_params.get("G", None)
            if G is None or u.ndim != 2 or G.ndim != 2:
                raise ValueError("Pour l'opérateur 'convolution', G doit être fourni et être une matrice 2D.")
            return (convolve(u, G) + rho * x + z) / (convolve(G, G.T) + rho)

    # Boucle principale
    for _ in range(K):
        # Mise à jour de y selon l'opérateur
        y = process_image_2(u, x, z, operator= compute_y_update, operator_type=operator_type, operator_params=operator_params, rho=rho)
        
        # Mise à jour de x avec le proximal
        x = process_image_2(y - z / rho, operator= prox, lambd=lambd / rho, **prox_params)
        
        # Mise à jour de la variable duale z
        z = z + rho * (x - y)
        
        # Critère de convergence
        if norm(x - y) < tol:
            break

    return x