from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import deepinv as dinv
import torch, os, itertools, cv2
import matplotlib.pyplot as plt

# Chargement des images

def load_img(fname, subfolder='data/set3c', resize=None, normalize=True):
    """
    Charge une image depuis un sous-dossier et la normalise si nécessaire.

    Parameters:
    - fname : str, nom du fichier image.
    - subfolder : str, chemin relatif vers le dossier contenant l'image.
    - resize : tuple, taille de redimensionnement (par défaut None).
    - normalize : bool, normalise l'image dans l'intervalle [0, 1] (par défaut True).

    Returns:
    - img : array_like, image chargée et prétraitée.
    """
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    image_directory = os.path.join(parent_dir, subfolder)
    file_path = os.path.join(image_directory, fname)
    
    try:
        img = Image.open(file_path)
        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        img = np.array(img)
        if normalize:
            img = img / 255
        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"Image '{fname}' introuvable dans le dossier '{subfolder}'.")


# Sauvegarder les images

def treat_image_and_add_psnr(image, reference_image, psnr=True):

    # Vérifiez la mutabilité de l'image
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    # Normalisation de l'image
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    if psnr and reference_image is not None:

        # Calcul du PSNR
        psnr_value = cv2.PSNR(reference_image, image)

        # Dimensions de l'image
        H, _ = image.shape[:2]

        # Texte à afficher
        text = f"PSNR: {psnr_value:.2f}"

        # Paramètres du texte
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.8
        thickness = 2

        # Obtention des dimensions du texte pour ajuster le fond
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        # Position du texte (coin inférieur gauche)
        x, y = 3, H - 7

        # Définir le rectangle de fond blanc
        rect_top_left = (0, y - text_height - 5)  # Coin supérieur gauche du rectangle
        rect_bottom_right = (text_width + 7, H)  # Coin inférieur droit du rectangle
        color_white = (255, 255, 255) if image.ndim == 3 else 255  # Blanc

        # Dessiner le rectangle blanc
        image = cv2.rectangle(image, rect_top_left, rect_bottom_right, color_white, -1)

        # Dessiner le texte noir sur le fond blanc
        color_black = (0, 0, 0) if image.ndim == 3 else 0  # Noir
        image = cv2.putText(image, text, (x, y), font, font_scale, color_black, thickness)

    return image

def save_path(folder, file_name, names_list, images_list, reference_image=None, psnr=True, trajectories=None):
    """
    Sauvegarde les images côte à côte et trace les trajectoires de PSNR si applicables.

    Parameters:
    ----------
    folder : str, Dossier où sauvegarder les résultats.
    file_name : str, Nom de base pour les fichiers sauvegardés.
    names_list : list of str, Noms associés aux images pour les titres.
    images_list : list of array_like, Liste des images à sauvegarder (niveaux de gris ou couleur).
    reference_image : array_like, optional, Image de référence pour calculer le PSNR.
    psnr : bool, optional, Si True, ajoute le PSNR sur les images sauvegardées (par défaut True).
    trajectories : list of list of array_like, optional, Trajectoires associées aux images réparées.

    Returns:
    -------
    None
    """
    # Définir le chemin de sauvegarde
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    path = os.path.join(parent_dir, folder)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{file_name}_comparaison.png")

    # Normalisation de l'image de référence
    if reference_image is not None:
        if reference_image.dtype != np.uint8:
            reference_image = (reference_image * 255).clip(0, 255).astype(np.uint8)

    # Traitement des images
    treated_images = []
    for im in images_list:
        treated_im = treat_image_and_add_psnr(im, reference_image, psnr)
        treated_images.append(treated_im)

    # Gestion des subplots
    n = len(treated_images) + (1 if reference_image is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:  # Cas où il y a une seule image
        axes = [axes]

    # Affichage de l'image originale si présente
    start = 0
    if reference_image is not None:
        axes[0].imshow(reference_image)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        start += 1

    # Affichage des images traitées
    for idx, (name, img) in enumerate(zip(names_list, treated_images), start=start):
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(name)
        axes[idx].axis("off")

    # Sauvegarde du fichier comparatif
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Gestion des trajectoires de PSNR
    if trajectories is not None and reference_image is not None:
        n_trajet = len(trajectories)
        plt.figure(figsize=(10, 6))
        for name, trajectory in zip(names_list[-n_trajet:], trajectories):
            psnr_values = [cv2.PSNR(reference_image, (traj * 255).clip(0, 255).astype(np.uint8)) for traj in trajectory]
            plt.plot(range(len(psnr_values)), psnr_values, marker="o", linestyle="-", label=name)

        plt.xlabel("Itérations")
        plt.ylabel("PSNR (dB)")
        plt.title("Évolution du PSNR au fil des itérations")
        plt.legend()
        plt.grid(True)
        psnr_plot_path = os.path.join(path, f"{file_name}_psnr_plot.png")
        plt.savefig(psnr_plot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()

def process_image(image, processing_function, **kwargs):
    """
    Traite une image en couleur ou en niveaux de gris.

    Parameters:
    - image : array_like, image en couleur (H, W, 3) ou en niveaux de gris (H, W).
    - processing_function : callable, fonction à appliquer à chaque canal ou à l'image.
    - kwargs : dict, paramètres pour la fonction de traitement.

    Returns:
    - array_like : image traitée (H, W, 3) ou (H, W).
    """

    if image.ndim == 2:  # Image en niveaux de gris
        return processing_function(image, **kwargs)
    elif image.ndim == 3 and image.shape[2] == 3:  # Image en couleur
        # Séparer les canaux
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Appliquer la fonction à chaque canal
        R_processed = processing_function(R, **kwargs)
        G_processed = processing_function(G, **kwargs)
        B_processed = processing_function(B, **kwargs)
        
        # Recomposer l'image
        return np.stack([R_processed, G_processed, B_processed], axis=2)
    else:
        raise ValueError("L'image doit être de dimension (H, W) ou (H, W, 3).")

def process_image_2(*images, operator, **kwargs):
    """
    Traite un ou plusieurs tableaux représentant des images (en couleur ou en niveaux de gris)
    en appliquant un opérateur canal par canal.

    Parameters:
    - *images : list of array_like, un ou plusieurs tableaux représentant les images.
    - operator : callable, opérateur ou fonction à appliquer.
    - kwargs : dict, paramètres supplémentaires pour la fonction de traitement.

    Returns:
    - array_like : image ou tableau traité (H, W, 3) ou (H, W).

    Raises:
    - ValueError : si les dimensions des images ne correspondent pas ou si elles ne sont pas valides.
    """
    # Vérifier que toutes les images ont les mêmes dimensions
    if len(images) < 1:
        raise ValueError("Au moins une image doit être fournie.")
    base_shape = images[0].shape
    for img in images:
        if img.shape != base_shape:
            raise ValueError("Toutes les images doivent avoir les mêmes dimensions.")
    
    # Cas des images en niveaux de gris
    if len(base_shape) == 2 or (len(base_shape) == 3 and base_shape[2] == 1):  # Image en niveaux de gris
        return operator(*images, **kwargs)
    
    # Cas des images en couleur
    elif len(base_shape) == 3 and base_shape[2] == 3:  # Images avec 3 canaux
        # Appliquer l'opérateur canal par canal
        return np.stack([operator(*(img[:, :, c] for img in images), **kwargs) for c in range(3)], axis=2)
    else:
        raise ValueError("Les images doivent être de dimension (H, W) ou (H, W, 3).")

# Produit scalaire u, v

def scalar_product(u, v):
    """
    Calcule le produit scalaire entre deux tableaux.

    Parameters:
    - u : array_like, premier tableau.
    - v : array_like, deuxième tableau.

    Returns:
    - float : produit scalaire de u et v.
    """
    if u.shape != v.shape:
        raise ValueError("Les dimensions de u et v doivent être identiques.")
    return np.sum(u * v)


# Norme de u

def norm(u):
    """
    Calcule la norme Euclidienne d'un tableau.

    Parameters:
    - u : array_like, tableau d'entrée.

    Returns:
    - float : norme Euclidienne de u.
    """
    if not isinstance(u, np.ndarray):
        raise TypeError("L'entrée u doit être un tableau numpy.")
    return np.sqrt(scalar_product(u, u))


# gradient de u

def gradient(u):
    """
    Calcule le gradient d'un tableau en utilisant les différences finies.

    Parameters:
    - u : array_like, tableau 2D.

    Returns:
    - grad_u : array_like, gradient de u de taille (2, m, n).
    """
    if u.ndim != 2:
        raise ValueError("L'entrée u doit être un tableau 2D.")
    
    m, n = u.shape
    grad_u = np.zeros((2, m, n))
    
    grad_u[0, :-1, :] = u[1:] - u[:-1]  # Gradient vertical
    grad_u[1, :, :-1] = u[:, 1:] - u[:, :-1]  # Gradient horizontal
    
    return grad_u

# divergence de u

def div(p):
    """
    Calcule la divergence d'un champ vectoriel.

    Parameters:
    - p : array_like, champ vectoriel de taille (2, m, n).

    Returns:
    - div_p : array_like, divergence de p de taille (m, n).
    """
    if p.ndim != 3 or p.shape[0] != 2:
        raise ValueError("L'entrée p doit être un tableau de taille (2, m, n).")
    
    m, n = p.shape[1:]
    div_1 = np.zeros((m, n))
    div_1[:-1, :] = p[0, :-1, :]
    div_1[1:, :] -= p[0, :-1, :]
    
    div_2 = np.zeros((m, n))
    div_2[:, :-1] = p[1, :, :-1]
    div_2[:, 1:] -= p[1, :, :-1]
    
    return div_1 + div_2


# laplacian de u

def laplacian(u):
    """
    Calcule le laplacien d'un tableau.

    Parameters:
    - u : array_like, tableau 2D.

    Returns:
    - laplace_u : array_like, laplacien de u.
    """
    if u.ndim != 2:
        raise ValueError("L'entrée u doit être un tableau 2D.")
    return div(gradient(u))

# critère de convergence

def convergence_criteria(u0, u1, conv_crit):
    """
    Vérifie la convergence entre deux tableaux.

    Parameters:
    - u0 : array_like, tableau à l'itération précédente.
    - u1 : array_like, tableau à l'itération actuelle.
    - conv_crit : float, critère de convergence.

    Returns:
    - bool : True si la convergence est atteinte, sinon False.
    """
    return norm(u0 - u1) / norm(u0) < conv_crit

    
# Mean Square Error
    
def MSE(u_truth, u_estim):
    """
    Calcule l'erreur quadratique moyenne (MSE) entre deux tableaux.

    Parameters:
    - u_truth : array_like, tableau de référence.
    - u_estim : array_like, tableau estimé.

    Returns:
    - float : erreur quadratique moyenne entre u_truth et u_estim.
    """
    if u_truth.shape != u_estim.shape:
        raise ValueError("Les tableaux u_truth et u_estim doivent avoir les mêmes dimensions.")
    return np.mean((u_truth - u_estim)**2)


# Peak Signal to Noise Ratio
 
def PSNR(u_truth, u_estim, max_intensity= 1.0):
    """
    Calcule le rapport signal-bruit (PSNR) entre deux images (niveaux de gris ou couleur).

    Parameters:
    - u_truth : array_like, image de référence (H, W) ou (H, W, 3).
    - u_estim : array_like, image estimée (H, W) ou (H, W, 3).
    - max_intensity : float, intensité maximale de l'image (par défaut : 1.0 pour cause de normalisation). # Possible 255 pour des images 8-bit

    Returns:
    - float : PSNR entre u_truth et u_estim.
    """
    if u_truth.shape != u_estim.shape:
        raise ValueError("Les dimensions des images doivent être identiques.")
    
    mse = np.mean((u_truth - u_estim) ** 2)
    if mse == 0:
        return float("inf")  # Les deux images sont identiques
    
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


# Numpy to tensor

def numpy_to_tensor(im_col):
    """
    Convertit une image NumPy en tenseur PyTorch.

    Parameters:
    - im_col : array_like, tableau NumPy représentant une image.

    Returns:
    - tensor : tenseur PyTorch de l'image.
    """

    if not isinstance(im_col, np.ndarray):
        raise TypeError("L'entrée doit être un tableau NumPy.")
    
    # Normaliser les types (float32)
    if im_col.dtype != np.float32:
        im_col = im_col.astype(np.float32)

    if im_col.ndim == 2:  # Image en niveaux de gris
        return torch.from_numpy(im_col).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif im_col.ndim == 3 and im_col.shape[2] in [3, 4]:  # Couleur ou RGBA
        return torch.from_numpy(im_col).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    else:
        raise ValueError("L'entrée doit être une image 2D (niveaux de gris) ou 3D (couleur).")
    
def tensor_to_numpy(im_tsr):
    """
    Convertit un tenseur PyTorch en tableau NumPy.

    Parameters:
    - im_tsr : tensor, tenseur PyTorch représentant une image.

    Returns:
    - array_like : tableau NumPy de l'image.
    """

    if not isinstance(im_tsr, torch.Tensor):
        raise TypeError("L'entrée doit être un tenseur PyTorch.")

    if im_tsr.dim() != 4 or im_tsr.size(0) != 1:
        raise ValueError("Le tenseur doit avoir une dimension (1, C, H, W).")
    
    # Si le tenseur a requires_grad=True, on le détache
    if im_tsr.requires_grad:
        im_tsr = im_tsr.detach()

    # Cas niveaux de gris
    if im_tsr.size(1) == 1:
        return im_tsr.squeeze(0).squeeze(0).numpy()  # (H, W)
    # Cas couleur ou RGBA
    elif im_tsr.size(1) in [3, 4]:
        return im_tsr.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
    else:
        raise ValueError("Le tenseur doit avoir 1, 3 ou 4 canaux (C).")

    
# Operateur

class operateur:
    def __init__(self, image):
        """
        Initialise l'image comme tenseur PyTorch.

        Parameters:
        - image : array_like, image d'entrée sous forme de tableau NumPy.

        Raises:
        - TypeError : si l'entrée n'est pas un tableau NumPy.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("L'image d'entrée doit être un tableau NumPy.")
        
        self.image = image

    def noise(self, sigma=0.2):
        """
        Bruite l'image en appliquant un modèle de bruit gaussien.

        Parameters:
        - sigma : float, écart-type du bruit gaussien (par défaut 0.2).

        Returns:
        - array_like : image bruitée sous forme de tableau NumPy.
        """
        if sigma <= 0:
            raise ValueError("Le paramètre sigma doit être positif.")
        
        noise = dinv.physics.Denoising()
        noise.noise_model = dinv.physics.GaussianNoise(sigma=sigma)
        return process_image_2(self.image, operator=lambda x: tensor_to_numpy(noise(numpy_to_tensor(x))))

    def blur(self, sigma=(2, 2), angle=0):
        """
        Applique un flou gaussien à l'image.

        Parameters:
        - sigma : tuple, écart-type du flou pour les axes x et y (par défaut (2, 2)).
        - angle : float, angle du flou en degrés (par défaut 0).

        Returns:
        - tuple : (image floutée sous forme de tableau NumPy, filtre de flou).
        """
        if not (isinstance(sigma, tuple) and len(sigma) == 2 and all(s > 0 for s in sigma)):
            raise ValueError("sigma doit être un tuple de deux valeurs positives.")
        
        Filt = dinv.physics.blur.gaussian_blur(sigma=sigma, angle=angle)
        Flou_oper = dinv.physics.Blur(Filt, padding = 'circular')
        blurred = process_image_2(self.image, operator= lambda x: tensor_to_numpy(Flou_oper(numpy_to_tensor(x).float())))
        return blurred, tensor_to_numpy(Filt)

    def inpaint(self, mask=torch.rand(1, 1, 256, 256) > 0.4, sigma=0.05):
        """
        Applique un masque et du bruit gaussien à l'image.

        Parameters:
        - mask : tensor, masque binaire indiquant les pixels à désactiver.
        - sigma : float, écart-type du bruit gaussien (par défaut 0.05).

        Returns:
        - tuple : (image masquée sous forme de tableau NumPy, masque utilisé).
        """
        if not isinstance(mask, torch.Tensor) or mask.dtype not in [torch.bool, torch.uint8]:
            raise ValueError("Le masque doit être un tenseur binaire.")
        if mask.shape[2:] != self.image.shape[:2]:
            raise ValueError("Les dimensions du masque doivent correspondre à celles de l'image.")
        if sigma <= 0:
            raise ValueError("Le paramètre sigma doit être positif.")
        
        Inpaint = dinv.physics.Inpainting(
            mask=mask,
            tensor_size=self.image.shape[:2],
            noise_model=dinv.physics.GaussianNoise(sigma=sigma)
        )
        inpainted = process_image_2(self.image, operator= lambda x: tensor_to_numpy(Inpaint(numpy_to_tensor(x))))
        return inpainted, tensor_to_numpy(mask)
    

def search_opt(func, u_truth, param_ranges, metric, func_params=None, prox_params_ranges=None):
    """
    Recherche exhaustive pour optimiser une fonction donnée en fonction de plusieurs paramètres,
    avec prise en charge optionnelle des paramètres spécifiques au prox.

    Parameters:
    ----------
    func : callable
        Fonction à optimiser (doit retourner une image ou un tableau).
    u_truth : array_like
        Donnée de référence pour évaluer la performance.
    param_ranges : dict
        Dictionnaire contenant les paramètres globaux à optimiser et leurs plages de valeurs.
    metric : callable
        Fonction pour évaluer la performance (doit retourner un score, ex. PSNR).
    func_params : dict, optional
        Dictionnaire des paramètres fixes pour `func` (par défaut None).
    prox_params_ranges : dict, optional
        Dictionnaire des plages pour les paramètres spécifiques au prox (par défaut None).

    Returns:
    -------
    tuple :
        - best_params : dict, combinaison des paramètres globaux qui maximise la métrique.
        - best_score : float, score maximal atteint.
        - score_map_df : DataFrame, carte des scores pour chaque combinaison de paramètres.
    """
    if not param_ranges:
        raise ValueError("Les plages de paramètres globaux ne peuvent pas être vides.")
    if func_params is None:
        func_params = {}

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    if any(len(vals) == 0 for vals in param_values):
        raise ValueError("Toutes les plages de paramètres globaux doivent contenir au moins une valeur.")

    # Gestion des paramètres spécifiques au prox
    use_prox_params = prox_params_ranges is not None and bool(prox_params_ranges)
    if use_prox_params:
        prox_param_names = list(prox_params_ranges.keys())
        prox_param_values = list(prox_params_ranges.values())
        if any(len(vals) == 0 for vals in prox_param_values):
            raise ValueError("Toutes les plages de paramètres du prox doivent contenir au moins une valeur.")

    # Initialisation
    best_params = None
    best_score = -np.inf
    score_map = []

    # Boucle principale sur les paramètres globaux
    with tqdm(total=len(list(itertools.product(*param_values))), desc="Recherche globale") as global_bar:
        for params in itertools.product(*param_values):
            current_params = dict(zip(param_names, params))

            # Si prox_params_ranges n'est pas fourni, exécutez uniquement pour les paramètres globaux
            prox_param_combinations = itertools.product(*prox_param_values) if use_prox_params else [None]

            for prox_params in prox_param_combinations:
                if use_prox_params:
                    current_prox_params = dict(zip(prox_param_names, prox_params))
                else:
                    current_prox_params = {}

                # Préparation des paramètres pour la fonction
                func_params_local = func_params.copy()
                func_params_local.update(current_params)
                if use_prox_params:
                    func_params_local["prox_params"] = current_prox_params

                try:
                    # Calculer la sortie de la fonction
                    result, _ = func(**func_params_local)

                    # Vérifier les dimensions avant de calculer la métrique
                    if result.shape != u_truth.shape:
                        raise ValueError("Les dimensions du résultat et de la référence ne correspondent pas.")

                    # Évaluer la performance
                    score = metric(u_truth, result)
                    score_entry = (current_params, current_prox_params, score) if use_prox_params else (current_params, score)
                    score_map.append(score_entry)

                    # Mise à jour du meilleur score
                    if score > best_score:
                        best_score = score
                        best_params = (current_params, current_prox_params) if use_prox_params else current_params

                except Exception as e:
                    print(f"Erreur avec paramètres {current_params}, prox {current_prox_params}: {e}")

            global_bar.update(1)  # Mise à jour de la barre externe

    # Conversion des scores en DataFrame
    if use_prox_params:
        score_map_df = pd.DataFrame([(dict(p), dict(pp), s) for p, pp, s in score_map], columns=["Params", "Prox_Params", "Score"])
    else:
        score_map_df = pd.DataFrame([(dict(p), s) for p, s in score_map], columns=["Params", "Score"])

    # Trier par score décroissant
    score_map_df = score_map_df.sort_values(by="Score", ascending=False)

    if use_prox_params:
        return best_params, best_score, score_map_df
    else:
        return best_params, best_score, score_map_df
