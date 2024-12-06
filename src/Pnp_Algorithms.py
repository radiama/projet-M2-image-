import numpy as np
from Proxy_Func import compute_gradient, validate_inputs, prox_l6
from Begin_Func import numpy_to_tensor, tensor_to_numpy, norm, process_image_2
from Variational_Func import convolve
import torch


def pnp_admm(u, operator_type, operator_params, tau, denoiser, rho=1.0, sigma=0.1, K=100, tol=1e-7):
    """
    Méthode Plug-and-Play ADMM (PNP-ADMM) modulable.

    Parameters:
    ----------
    u : array_like, Donnée d'entrée.
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    tau : float, Pas de gradient pour la régularisation.
    denoiser : callable, Modèle ou fonction de débruitage.
    rho : float, Paramètre du multiplicateur (par défaut 1.0).
    sigma : float, Niveau de bruit pour le débruiteur (par défaut 0.1).
    K : int, Nombre maximal d'itérations (par défaut 100).
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
    PNP-ADMM est une extension de ADMM, où un modèle de débruitage est utilisé comme régularisation implicite.
    """

    # Validation des paramètres
    if sigma is None:
        validate_inputs(u, operator_type, operator_params, tau, rho, K)
    else:
        validate_inputs(u, operator_type, operator_params, tau, rho, sigma, K)

    # Initialisation des variables
    x = np.zeros_like(u)  # Variable primale
    y = np.zeros_like(u)  # Variable intermédiaire
    z = np.zeros_like(u)  # Variable duale
    trajectoires = [np.copy(x)]  # Stocker les trajectoires

    # Fonction pour gérer les différents types d'opérateurs
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
        # Mise à jour de y (variable intermédiaire)
        y = process_image_2(u, x, z, operator=compute_gradient_2, operator_type=operator_type, operator_params=operator_params, rho=rho)

        # Mise à jour de x (débruiteur)
        with torch.no_grad():
            y_tensor = numpy_to_tensor(y)
            x_tensor = denoiser(y_tensor, sigma=sigma)
            x = tensor_to_numpy(x_tensor)
            trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Mise à jour de la variable duale z
        z += rho * (x - y)

        # Critère de convergence
        if norm(x - y) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires

def pnp_pgm(u, operator_type, operator_params, tau, denoiser, sigma=0.1, K=100, tol=1e-7):
    """
    Méthode de gradient projeté Plug-and-Play (PNP-APGM).

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    tau : float, Pas de gradient.
    denoiser : callable, Modèle ou fonction de débruitage.
    sigma : float, Niveau de bruit pour le débruiteur (par défaut 0.1).
    K : int, Nombre maximal d'itérations (par défaut 100).
    tol : float, Tolérance pour le critère de convergence (par défaut 1e-7).

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
    Cette méthode applique une accélération de Nesterov combinée avec un débruiteur Plug-and-Play.
    """

    # Validation des paramètres
    if sigma is None:
        validate_inputs(u, operator_type, operator_params, tau, K)
    else:
        validate_inputs(u, operator_type, operator_params, tau, sigma, K)

    # Initialisation
    y = np.copy(u)  # Variable intermédiaire
    trajectoires = [np.copy(u)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Mise à jour par descente de gradient
        x_half = y - tau * grad_f

        # # Application du débruiteur
        with torch.no_grad():
            x_half_tensor = numpy_to_tensor(x_half)
            x_tensor = denoiser(x_half_tensor, sigma=sigma)
            x = tensor_to_numpy(x_tensor)
            trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires

def pnp_apgm(u, operator_type, operator_params, tau, denoiser, sigma=0.1, K=100, tol=1e-7):
    """
    Méthode de gradient projeté accéléré Plug-and-Play (PNP-APGM).

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    tau : float, Pas de gradient.
    denoiser : callable, Modèle ou fonction de débruitage.
    sigma : float, Niveau de bruit pour le débruiteur (par défaut 0.1).
    K : int, Nombre maximal d'itérations (par défaut 100).
    tol : float, Tolérance pour le critère de convergence (par défaut 1e-7).

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
    Cette méthode applique une accélération de Nesterov combinée avec un débruiteur Plug-and-Play.
    """

    # Validation des paramètres
    if sigma is None:
        validate_inputs(u, operator_type, operator_params, tau, K)
    else:
        validate_inputs(u, operator_type, operator_params, tau, sigma, K)

    # Initialisation
    x_old = np.copy(u)  # Solution précédente
    y = np.copy(u)  # Variable intermédiaire
    trajectoires = [np.copy(x_old)]  # Stocker les trajectoires
    theta_n = lambda n: 1 / (n + 100)  # Séquence décroissante

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Mise à jour par descente de gradient
        x_half = y - tau * grad_f

        # # Application du débruiteur
        with torch.no_grad():
            x_half_tensor = numpy_to_tensor(x_half)
            x_tensor = denoiser(x_half_tensor, sigma=sigma)
            x = tensor_to_numpy(x_tensor)
            trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Accélération
        y = x * (1 - theta_n(k)) + theta_n(k) *(x_old)

        # Mise à jour des variables
        x_old = np.copy(x)

        # Critère de convergence
        if norm(x - x_half) < tol:
            print(f"Convergence atteinte à l'itération {k+1}.")
            break

    return x, trajectoires


def pnp_apgm2(u, operator_type, operator_params, tau, denoiser, sigma=0.1, K=100, tol=1e-7):
    """
    Méthode de gradient projeté accéléré Plug-and-Play (PNP-APGM).

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    tau : float, Pas de gradient.
    denoiser : callable, Modèle ou fonction de débruitage.
    sigma : float, Niveau de bruit pour le débruiteur (par défaut 0.1).
    K : int, Nombre maximal d'itérations (par défaut 100).
    tol : float, Tolérance pour le critère de convergence (par défaut 1e-7).

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
    Cette méthode applique une accélération de Nesterov combinée avec un débruiteur Plug-and-Play.
    """

    # Validation des paramètres
    if sigma is None:
        validate_inputs(u, operator_type, operator_params, tau, K)
    else:
        validate_inputs(u, operator_type, operator_params, tau, sigma, K)

    # Initialisation
    x_old = np.copy(u)  # Solution précédente
    y = np.copy(u)  # Variable intermédiaire
    t = 1  # Paramètre d'accélération
    trajectoires = [np.copy(x_old)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, y, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Mise à jour par descente de gradient
        x_half = y - tau * grad_f

        # # Application du débruiteur
        with torch.no_grad():
            x_half_tensor = numpy_to_tensor(x_half)
            x_tensor = denoiser(x_half_tensor, sigma=sigma)
            x = tensor_to_numpy(x_tensor)
            trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # Accélération de Nesterov
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


def pnp_red(u, operator_type, operator_params, tau, lambd, denoiser, sigma=0.1, K=100, tol=1e-7):
    """
    Méthode Plug-and-Play Regularization by Denoising (PnP-RED).

    Parameters:
    ----------
    u : array_like, Donnée d'entrée (image ou signal).
    operator_type : str, Type d'opérateur ("none", "mask", "convolution").
    operator_params : dict, Paramètres spécifiques à l'opérateur.
        - Pour "mask": un masque binaire ou pondéré "Mask" (array_like).
        - Pour "convolution": un noyau de convolution 2D "G" (array_like).
    tau : float, Pas de gradient (step size).
    lambd : float, Paramètre de régularisation.
    denoiser : callable, Modèle ou fonction de débruitage.
    sigma : float, Niveau de bruit pour le débruiteur (par défaut 0.1).
    K : int, Nombre maximal d'itérations (par défaut 100).
    tol : float, Tolérance pour le critère de convergence (par défaut 1e-7).

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
    Cette méthode combine la descente de gradient et un débruiteur comme régularisation implicite.
    """

    # Validation des paramètres
        # Validation des paramètres
    if sigma is None:
        validate_inputs(u, operator_type, operator_params, tau, lambd, K)
    else:
        validate_inputs(u, operator_type, operator_params, tau, lambd, sigma, K)

    # Initialisation
    x = np.copy(u)  # Initialisation de la solution
    trajectoires = [np.copy(x)]  # Stocker les trajectoires

    # Boucle d'optimisation
    for k in range(K):
        # Calcul du gradient
        grad_f = process_image_2(u, x, operator=compute_gradient, operator_type=operator_type, operator_params=operator_params)

        # Application du débruiteur
        with torch.no_grad():
            x_tensor = numpy_to_tensor(x)
            Rx_tensor = denoiser(x_tensor, sigma=sigma)
            Rx = tensor_to_numpy(Rx_tensor)

        # Mise à jour de la solution
        x = x - tau * (grad_f + lambd * (x - Rx))
        trajectoires.append(np.copy(x))  # Sauvegarder la trajectoire

        # # Critère de convergence
        # if norm(grad_f) < tol:
        #     print(f"Convergence atteinte à l'itération {k+1}.")
        #     break

    return x, trajectoires