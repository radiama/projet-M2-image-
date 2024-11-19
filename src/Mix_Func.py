from Begin_Func import gradient, div, norm, laplacian, process_image_2
from Variational_Func import gaussian_kernel, convolve
import numpy as np

# Equation de la chaleur

def heat_equation(f, dt, K):
    """
    Résout l'équation de la chaleur sur une image.

    Parameters:
    - f : array_like, image d'entrée (niveaux de gris ou couleur).
    - dt : float, pas de temps (doit être positif).
    - K : int, nombre d'itérations (doit être positif).

    Returns:
    - u : array_like, image après diffusion.

    Raises:
    - ValueError : si dt est négatif ou K non positif.
    """
    if dt <= 0:
        raise ValueError("Le pas de temps dt doit être positive.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")

    def diffusion_channel(channel):
        u = channel.copy()
        for _ in range(K):
            lap_u = laplacian(u)
            u += dt * lap_u
        return u

    # Appliquer la diffusion à tous les canaux si nécessaire
    return process_image_2(f, operator= diffusion_channel)

# Contours

def grad_edge(u, eta):
    """
    Détecte les contours dans une image en utilisant le gradient et un seuil.

    Parameters:
    - u : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).
    - eta : float, seuil de sensibilité pour détecter les contours.

    Returns:
    - contours : array_like, carte binaire des contours.

    Raises:
    - ValueError : si eta est négatif ou si u n'est pas une image valide.
    """
    if eta <= 0:
        raise ValueError("Le seuil eta doit être positif.")
    if u.ndim not in [2, 3]:
        raise ValueError("L'image u doit être un tableau 2D ou 3D.")

    def edge_detection_channel(channel):
        grad_u = gradient(channel)
        grad_magnitude = np.sqrt(grad_u[0] ** 2 + grad_u[1] ** 2)
        return grad_magnitude > eta

    # Appliquer la détection aux images en couleur ou en niveaux de gris
    return process_image_2(u, edge_detection_channel)

def change_sign(J):
    """
    Détecte les changements de signe dans une image.

    Parameters:
    - J : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).

    Returns:
    - bool_map : array_like, carte booléenne des changements de signe.

    Raises:
    - ValueError : si J n'est pas un tableau 2D ou 3D.
    """
    if J.ndim not in [2, 3]:
        raise ValueError("L'image J doit être un tableau 2D ou 3D.")

    def change_sign_channel(channel):
        bool_map = np.full_like(channel, False, dtype=bool)

        # Changements de signe en vertical
        prod_ver = (channel[:-1, :] * channel[1:, :]) <= 0
        diff_abs_ver = abs(channel[:-1, :]) - abs(channel[1:, :])
        bool_map[:-1, :] |= prod_ver & (diff_abs_ver >= 0)

        # Changements de signe en horizontal
        prod_hor = (channel[:, :-1] * channel[:, 1:]) <= 0
        diff_abs_hor = abs(channel[:, :-1]) - abs(channel[:, 1:])
        bool_map[:, :-1] |= prod_hor & (diff_abs_hor >= 0)

        return bool_map

    # Appliquer la détection des changements de signe aux canaux
    return process_image_2(J, operator= change_sign_channel)


def lap_edge(u):
    """
    Détecte les contours dans une image en utilisant le Laplacien.

    Parameters:
    - u : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).

    Returns:
    - contours : array_like, carte binaire des contours.

    Raises:
    - ValueError : si u n'est pas un tableau 2D ou 3D.
    """
    if u.ndim not in [2, 3]:
        raise ValueError("L'image u doit être un tableau 2D ou 3D.")

    def laplacian_edge_channel(channel):
        laplacian_u = laplacian(channel)
        bool_map = change_sign(laplacian_u)
        contours = np.zeros_like(channel, dtype=int)
        contours[bool_map] = 1
        return contours

    # Appliquer la détection aux canaux si nécessaire
    return process_image_2(u, operator= laplacian_edge_channel)

def combined_edge(u, eta):
    """
    Combine deux critères de détection pour détecter des contours dans une image.

    Parameters:
    - u : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).
    - eta : float, seuil de sensibilité pour le gradient.

    Returns:
    - contours : array_like, carte binaire des contours combinés.

    Raises:
    - ValueError : si eta est négatif ou si u n'est pas un tableau valide.
    """
    if eta <= 0:
        raise ValueError("Le seuil eta doit être positif.")
    if u.ndim not in [2, 3]:
        raise ValueError("L'image u doit être un tableau 2D ou 3D.")
    
    def combined_edge_channel(channel):
        cond1 = grad_edge(channel, eta)
        cond2 = lap_edge(channel)
        return cond1 & cond2

    # Appliquer la détection aux images en couleur ou en niveaux de gris
    return process_image_2(u, operator= combined_edge_channel)


def Marr_Hildreth(u, eta):
    """
    Implémente l'algorithme de détection des contours de Marr-Hildreth.

    Parameters:
    - u : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).
    - eta : float, seuil de sensibilité pour le gradient.

    Returns:
    - contours : array_like, carte binaire des contours.

    Raises:
    - ValueError : si eta est négatif ou si u n'est pas un tableau valide.
    """
    if eta <= 0:
        raise ValueError("Le seuil eta doit être positif.")
    if u.ndim not in [2, 3]:
        raise ValueError("L'image u doit être un tableau 2D ou 3D.")

    def marr_hildreth_channel(channel):
        laplacian_u = laplacian(channel)
        norm_grad_u = norm(gradient(channel))
        contours = np.zeros_like(channel, dtype=int)
        bool_map = change_sign(laplacian_u) & (norm_grad_u > eta)
        contours[bool_map] = 1
        return contours

    # Appliquer la détection aux images en couleur ou en niveaux de gris
    return process_image_2(u, operator= marr_hildreth_channel)



def g_exp(xi, alpha=1):
    """
    Applique un filtrage passe-bas en lissant les variations rapides.

    Parameters:
    - xi : array_like, données à filtrer (2D, 3D ou 1D).
    - alpha : float, paramètre contrôlant l'intensité du filtrage (par défaut 1).

    Returns:
    - array_like : données filtrées.

    Raises:
    - ValueError : si alpha est négatif ou nul.
    """
    if alpha <= 0:
        raise ValueError("Le paramètre alpha doit être strictement positif.")
    
    def exp_filter(channel):
        return np.exp(- (channel / alpha) ** 2)
    
    return process_image_2(xi, operator= exp_filter)



def g_PM(xi, alpha=1):
    """
    Applique un filtrage passe-haut pour préserver les détails fins.

    Parameters:
    - xi : array_like, données à filtrer (2D, 3D ou 1D).
    - alpha : float, paramètre contrôlant l'anisotropie (par défaut 1).

    Returns:
    - array_like : données filtrées.

    Raises:
    - ValueError : si alpha est négatif ou nul.
    """
    if alpha <= 0:
        raise ValueError("Le paramètre alpha doit être strictement positif.")
    
    def pm_filter(channel):
        return 1 / np.sqrt((channel / alpha) ** 2 + 1)
    
    return process_image_2(xi, operator= pm_filter)


def Perona_Malik(f, dt, K, alpha, g=g_PM):
    """
    Applique la diffusion anisotrope de Perona-Malik pour restaurer une image.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - dt : float, pas de temps (doit être positive).
    - K : int, nombre d'itérations (doit être positif).
    - alpha : float, paramètre de diffusion (doit être positif).
    - g : function, fonction de filtrage (par défaut g_PM).

    Returns:
    - u : array_like, image restaurée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if dt <= 0:
        raise ValueError("Le pas de temps dt doit être positive.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if alpha <= 0:
        raise ValueError("Le paramètre alpha doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    def perona_malik_channel(channel):
        u = np.copy(channel)
        for k in range(K):
            grad_u = gradient(u)
            norm_grad_u = norm(grad_u)
            anisotropic_diffusion = div(g(norm_grad_u, alpha=alpha) * grad_u)
            u = u + dt * anisotropic_diffusion
        return u

    # Appliquer la diffusion anisotrope aux canaux si nécessaire
    return process_image_2(f, operator= perona_malik_channel)


def Perona_Malik_enhanced(f, dt, K, alpha, s, g=g_PM):
    """
    Amélioration de la diffusion anisotrope de Perona-Malik avec un lissage gaussien.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - dt : float, pas de temps (doit être positive).
    - K : int, nombre d'itérations (doit être positif).
    - alpha : float, paramètre de diffusion (doit être positif).
    - s : float, écart-type pour le noyau gaussien (doit être positif).
    - g : function, fonction de filtrage (par défaut g_PM).

    Returns:
    - u : array_like, image restaurée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if dt <= 0:
        raise ValueError("Pas de temps dt doit être positive.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if alpha <= 0:
        raise ValueError("Le paramètre alpha doit être strictement positif.")
    if s <= 0:
        raise ValueError("Le paramètre s doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    def perona_malik_channel(channel):
        u = np.copy(channel)
        G = gaussian_kernel(dt, s)
        
        for k in range(K):        
            grad_u = gradient(u)
            conv = convolve(u, G)
            grad_conv = gradient(conv)
            norm_grad_conv = norm(grad_conv)
            
            anisotropic_diffusion = div(g(norm_grad_conv, alpha=alpha) * grad_u)
            u = u + dt * anisotropic_diffusion
        
        return u

    # Appliquer la diffusion anisotrope aux canaux si nécessaire
    return process_image_2(f, operator= perona_malik_channel)
