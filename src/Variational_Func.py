import numpy as np
from scipy.signal import convolve2d
from Begin_Func import laplacian, div, gradient, norm, process_image_2

# Gaussian noise

def add_gaussian_noise(I, s):
    """
    Ajoute un bruit gaussien à une image.

    Parameters:
    - I : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).
    - s : float, écart-type du bruit gaussien.

    Returns:
    - I_out : array_like, image bruitée.

    Raises:
    - ValueError : si I n'est pas 2D ou 3D, ou si s est négatif.
    """
    if I.ndim not in [2, 3]:
        raise ValueError("L'image I doit être un tableau 2D ou 3D.")
    if s < 0:
        raise ValueError("Le paramètre s doit être positif.")

    def gaussian_noise_channel(channel):
        m, n = channel.shape
        return channel + s * np.random.randn(m, n)

    # Ajouter du bruit pour les images en niveaux de gris ou en couleur
    return process_image_2(I, operator= gaussian_noise_channel)


# Gaussian kernel 

def gaussian_kernel(size, sigma):
    """
    Génère un noyau gaussien.

    Parameters:
    - size : int, taille du noyau (doit être impair).
    - sigma : float, écart-type du noyau gaussien.

    Returns:
    - kernel : array_like, noyau gaussien normalisé.

    Raises:
    - ValueError : si size n'est pas impair ou si sigma est négatif.
    """
    if not isinstance(size, int) or size <= 0 or size % 2 == 0:
        raise ValueError("La taille du noyau size doit être un entier impair et positif.")
    if sigma <= 0:
        raise ValueError("Le paramètre sigma doit être strictement positif.")
    
    variance = sigma**2
    half_size = size // 2
    range_matrix = np.arange(-half_size, half_size + 1)
    X, Y = np.meshgrid(range_matrix, range_matrix)
    kernel = np.exp(- (X**2 + Y**2) / (2 * variance)) / (2 * np.pi * variance)

    # Normalisation
    return kernel / kernel.sum()


# Convolution

def convolve(f, G):
    """
    Applique une convolution bidimensionnelle.

    Parameters:
    - f : array_like, tableau 2D (niveaux de gris) ou 3D (couleur).
    - G : array_like, noyau de convolution (2D).

    Returns:
    - result : array_like, résultat de la convolution.

    Raises:
    - ValueError : si f ou G n'ont pas les dimensions attendues.
    """
    if G.ndim != 2:
        raise ValueError("Le noyau G doit être un tableau 2D.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    def convolve_channel(channel):
        return convolve2d(channel, G, mode='same', boundary='wrap')

    # Appliquer la convolution pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= convolve_channel)



# Denoise_Tikhonov

def Denoise_Tikhonov(f, K, lamb, tau=None):
    """
    Débruite une image en utilisant la régularisation de Tikhonov.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - K : int, nombre d'itérations (doit être positif).
    - lamb : float, paramètre de régularisation (doit être strictement positif).
    - tau : float, pas de mise à jour (calculé automatiquement si None).

    Returns:
    - u : array_like, image débruitée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if tau is not None and tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    # Calculer le pas de mise à jour si non spécifié
    if tau is None:
        tau = 1 / (lamb + 4)

    def tikhonov_channel(channel):
        u = np.copy(channel)
        for _ in range(1, K + 1):
            u = u + tau * (lamb * (channel - u) + laplacian(u))
        return u

    # Appliquer la régularisation pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= tikhonov_channel)



# Denoise_TV

def Denoise_TV(f, K, lamb, eps, tau):
    """
    Débruite une image en utilisant la régularisation par variation totale (TV).

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - K : int, nombre d'itérations (doit être positif).
    - lamb : float, paramètre de régularisation (doit être strictement positif).
    - eps : float, paramètre pour éviter les divisions par zéro (doit être strictement positif).
    - tau : float, pas de mise à jour (doit être strictement positif).

    Returns:
    - u : array_like, image débruitée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if eps <= 0:
        raise ValueError("Le paramètre eps doit être strictement positif.")
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    def tv_channel(channel):
        u = np.copy(channel)
        for _ in range(1, K + 1):
            grad_u = gradient(u)
            norm_grad_u = norm(grad_u)
            regularized_grad = grad_u / np.sqrt(norm_grad_u**2 + eps)
            u = u + tau * (lamb * (channel - u) + div(regularized_grad))
        return u

    # Appliquer la régularisation TV pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= tv_channel)


# Denoise_Tikhonov_Fourier

def Denoise_Tikhonov_Fourier(f, lamb):
    """
    Débruite une image en utilisant la régularisation de Tikhonov et la transformée de Fourier.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - lamb : float, paramètre de régularisation (doit être strictement positif).

    Returns:
    - u : array_like, image débruitée.

    Raises:
    - ValueError : si lamb est négatif ou si f n'est pas un tableau valide.
    """
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")

    def tikhonov_fourier_channel(channel):
        m, n = channel.shape
        x = np.fft.fft2(channel)
        p = np.arange(m).reshape(-1, 1)  # Indices pour les lignes
        q = np.arange(n).reshape(1, -1)  # Indices pour les colonnes

        denominator = lamb + 4 * (np.sin(np.pi * p / m)**2 + np.sin(np.pi * q / n)**2)
        y = lamb * x / denominator
        u = np.fft.ifft2(y)
        return np.real(u)

    # Appliquer la régularisation pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= tikhonov_fourier_channel)



# Deconvolution_TV

def Deconvolution_TV(f, G, tau, eps, K, lamb):
    """
    Effectue une déconvolution basée sur la variation totale (TV).

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - G : array_like, noyau de convolution.
    - tau : float, pas de mise à jour (doit être strictement positif).
    - eps : float, paramètre pour éviter les divisions par zéro (doit être strictement positif).
    - K : int, nombre d'itérations (doit être supérieur à zéro).
    - lamb : float, paramètre de régularisation (doit être strictement positif).

    Returns:
    - u : array_like, image déconvoluée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if eps <= 0:
        raise ValueError("Le paramètre eps doit être strictement positif.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if f.ndim not in [2, 3]:
        raise ValueError("L'image f doit être un tableau 2D ou 3D.")
    if G.ndim != 2:
        raise ValueError("Le noyau de convolution G doit être un tableau 2D.")

    def tv_deconvolution_channel(channel):
        u = np.copy(channel)
        for _ in range(1, K):
            grad_u = gradient(u)
            fidelity_term = convolve(channel - convolve(u, G), G)
            regularization_term = div(grad_u / np.sqrt(norm(grad_u)**2 + eps**2))
            u = u + tau * (lamb * fidelity_term + regularization_term)
        return u

    # Appliquer la déconvolution pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= tv_deconvolution_channel)



# Impainting_TV

def Inpainting_TV(f, M, tau, eps, K, lamb):
    """
    Effectue un inpainting basé sur la variation totale (TV).

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - M : array_like, masque binaire indiquant les zones à inpaint (mêmes dimensions que f).
    - tau : float, pas de mise à jour (doit être strictement positif).
    - eps : float, paramètre pour éviter les divisions par zéro (doit être strictement positif).
    - K : int, nombre d'itérations (doit être supérieur à zéro).
    - lamb : float, paramètre de régularisation (doit être strictement positif).

    Returns:
    - u : array_like, image restaurée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if eps <= 0:
        raise ValueError("Le paramètre eps doit être strictement positif.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if f.shape != M.shape:
        raise ValueError("Le masque M doit avoir les mêmes dimensions que l'image f.")
    if M.dtype not in [np.bool_, np.int32, np.uint8]:
        raise ValueError("Le masque M doit être binaire (valeurs 0 ou 1).")

    def tv_inpainting_channel(channel, mask):
        u = np.copy(channel)
        for _ in range(K):
            grad_u = gradient(u)
            fidelity_term = (channel - u) * mask
            regularization_term = div(grad_u / np.sqrt(norm(grad_u)**2 + eps**2))
            u = u + tau * (lamb * fidelity_term + regularization_term)
        return u

    # Appliquer l'inpainting pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= lambda channel: tv_inpainting_channel(channel, M))


# Impainting_Tikhonov

def Inpainting_Tikhonov(f, M, tau, K, lamb):
    """
    Effectue un inpainting basé sur la régularisation de Tikhonov.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - M : array_like, masque binaire indiquant les zones à inpaint (mêmes dimensions que f).
    - tau : float, pas de mise à jour (doit être strictement positif).
    - K : int, nombre d'itérations (doit être supérieur à zéro).
    - lamb : float, paramètre de régularisation (doit être strictement positif).

    Returns:
    - u : array_like, image restaurée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if f.shape != M.shape:
        raise ValueError("Le masque M doit avoir les mêmes dimensions que l'image f.")
    if M.dtype not in [np.bool_, np.int32, np.uint8]:
        raise ValueError("Le masque M doit être binaire (valeurs 0 ou 1).")

    def tikhonov_inpainting_channel(channel, mask):
        u = np.copy(channel)
        for _ in range(K):
            fidelity_term = (channel - u) * mask
            regularization_term = laplacian(u)
            u = u + tau * (lamb * fidelity_term + regularization_term)
        return u

    # Appliquer l'inpainting pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= lambda channel: tikhonov_inpainting_channel(channel, M))


# Denoise_g1

def Denoise_g(f, K, lamb, eps, tau, g_type="g1"):
    """
    Débruite une image en utilisant une régularisation basée sur les fonctions g_1 ou g_2.

    Parameters:
    - f : array_like, image d'entrée (2D ou 3D).
    - K : int, nombre d'itérations.
    - lamb : float, paramètre de régularisation.
    - eps : float, paramètre pour éviter les divisions par zéro.
    - tau : float, pas de mise à jour.
    - g_type : str, type de fonction de régularisation ("g1" ou "g2").

    Returns:
    - u : array_like, image débruitée.

    Raises:
    - ValueError : si les paramètres sont invalides.
    """
    if tau is None:
        tau = 1 / (lamb + 4)
    if tau <= 0:
        raise ValueError("Le paramètre tau doit être strictement positif.")
    if lamb <= 0:
        raise ValueError("Le paramètre lamb doit être strictement positif.")
    if eps <= 0:
        raise ValueError("Le paramètre eps doit être strictement positif.")
    if K <= 0:
        raise ValueError("Le nombre d'itérations K doit être supérieur à zéro.")
    if g_type not in ["g1", "g2"]:
        raise ValueError("Le paramètre g_type doit être 'g1' ou 'g2'.")

    def denoise_channel(channel):
        u = np.copy(channel)
        for _ in range(K):
            grad_u = gradient(u)
            norm_grad = norm(grad_u)
            if g_type == "g1":
                div_grad = div(2 * grad_u * norm_grad / ((eps + norm_grad**2)**2))
            elif g_type == "g2":
                div_grad = div(2 * grad_u * norm_grad / (eps + norm_grad**2))
            u = u + tau * (lamb * (channel - u) + div_grad)
        return u

    # Appliquer le débruitage pour les images en niveaux de gris ou en couleur
    return process_image_2(f, operator= denoise_channel)
