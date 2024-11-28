import torch  # Bibliothèque PyTorch pour les calculs sur les tenseurs
import numpy as np  # Bibliothèque Numpy pour les manipulations de tableaux
import torch.nn as nn  # Module pour les réseaux de neurones dans PyTorch

def weights_init_drunet(m):
    """
    Initialise les poids des couches de convolution avec une matrice orthogonale.
    :param m: Couche du modèle. Si la couche est une convolution, les poids sont
              initialisés de façon orthogonale pour plus de stabilité.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)  # Gain contrôle l'échelle des poids

def tensor2array(img):
    """
    Convertit un tenseur PyTorch en un tableau NumPy au format (H, W, C).
    
    :param img: Tenseur PyTorch, supposé au format (C, H, W).
    :return: Tableau NumPy au format (H, W, C), pour visualisation d'image.
    """
    img = img.cpu().detach().numpy()  # Détache le tenseur et le convertit en NumPy
    img = np.transpose(img, (1, 2, 0))  # Réorganise en format (H, W, C)
    return img

def array2tensor(img):
    """
    Convertit un tableau NumPy en un tenseur PyTorch au format (C, H, W).
    
    :param img: Tableau NumPy au format (H, W, C).
    :return: Tenseur PyTorch au format (C, H, W), adapté à l'entrée du modèle.
    """
    return torch.from_numpy(img).permute(2, 0, 1)  # Convertit en tenseur et réorganise en (C, H, W)

def get_weights_url(model_name, file_name):
    """
    Génère l'URL pour télécharger les poids d'un modèle depuis Hugging Face.
    
    :param model_name: Nom du dépôt du modèle sur Hugging Face.
    :param file_name: Nom du fichier de poids à télécharger.
    :return: Chaîne de caractères de l'URL pour télécharger les poids.
    """
    return ("https://huggingface.co/deepinv/" + model_name + "/resolve/main/" + file_name + "?download=true")

def test_pad(model, L, modulo=16):
    """
    Applique un padding (remplissage) pour adapter l'image aux dimensions requises par le modèle.
    
    :param model: Modèle utilisé pour le traitement de l'image.
    :param L: Tenseur de l'image d'entrée de basse qualité.
    :param modulo: Modulo pour le padding, pour s'assurer que les dimensions soient multiples de 16.
    :return: Tenseur d'image traité après suppression du padding supplémentaire.
    """
    h, w = L.size()[-2:]  # Hauteur et largeur de l'image d'entrée
    padding_bottom = int(np.ceil(h / modulo) * modulo - h)  # Calcule le padding en hauteur
    padding_right = int(np.ceil(w / modulo) * modulo - w)  # Calcule le padding en largeur
    L = torch.nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))(L)  # Applique le padding
    E = model(L)  # Traite l'image paddée avec le modèle
    E = E[..., :h, :w]  # Découpe pour revenir aux dimensions originales
    return E

def test_onesplit(model, L, refield=32, sf=1):
    """
    Divise l'image d'entrée en quatre parties, les traite et recompose les sorties.
    
    :param model: Modèle utilisé pour traiter chaque partie de l'image.
    :param L: Tenseur de l'image d'entrée de basse qualité.
    :param refield: Taille du champ réceptif, influence la division.
    :param sf: Facteur de mise à l'échelle pour la super-résolution.
    :return: Tenseur d'image traité, reconstitué à partir des sorties du modèle.
    """
    h, w = L.size()[-2:]  # Hauteur et largeur de l'image d'entrée
    top = slice(0, (h // 2 // refield + 1) * refield)  # Définir la partie supérieure
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)  # Définir la partie inférieure
    left = slice(0, (w // 2 // refield + 1) * refield)  # Définir la partie gauche
    right = slice(w - (w // 2 // refield + 1) * refield, w)  # Définir la partie droite
    Ls = [
        L[..., top, left],  # Partie haut-gauche
        L[..., top, right],  # Partie haut-droite
        L[..., bottom, left],  # Partie bas-gauche
        L[..., bottom, right],  # Partie bas-droite
    ]
    Es = [model(Ls[i]) for i in range(4)]  # Applique le modèle à chaque partie
    b, c = Es[0].size()[:2]  # Dimensions de batch et de canal
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)  # Tenseur vide pour le résultat final
    # Place chaque partie traitée dans la position correspondante dans E
    E[..., : h // 2 * sf, : w // 2 * sf] = Es[0][..., : h // 2 * sf, : w // 2 * sf]
    E[..., : h // 2 * sf, w // 2 * sf : w * sf] = Es[1][..., : h // 2 * sf, (-w + w // 2) * sf :]
    E[..., h // 2 * sf : h * sf, : w // 2 * sf] = Es[2][..., (-h + h // 2) * sf :, : w // 2 * sf]
    E[..., h // 2 * sf : h * sf, w // 2 * sf : w * sf] = Es[3][..., (-h + h // 2) * sf :, (-w + w // 2) * sf :]
    return E