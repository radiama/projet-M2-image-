# Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
# code borrowed from Deepinv https://github.com/deepinv/deepinv/blob/main/deepinv/model/DRUNet

# Importation des modules essentiels
import torch
import torch.nn as nn
from collections import OrderedDict
from Utils import get_weights_url, test_onesplit, test_pad

# Configuration du type de tenseur en fonction de la disponibilité du GPU
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Classe DRUNet (Denoising Residual U-Net): Réseau de débruitage basé sur une architecture U-Net
class DRUNet(nn.Module):
    r"""
    Réseau de débruitage DRUNet.

    L'architecture est basée sur le papier "Plug-and-Play Image Restoration with Deep Denoiser Prior",
    et a une structure similaire à U-Net avec des blocs convolutifs dans les parties encodage et décodage.
    
    Le réseau prend en compte le niveau de bruit de l'image d'entrée, qui est codé comme un canal d'entrée supplémentaire.
    """

    def __init__(
        self, in_channels=3, out_channels=3, nc=[64, 128, 256, 512], nb=4, act_mode="R",
        downsample_mode="strideconv", upsample_mode="convtranspose", pretrained="download", device=None,
    ):
        super(DRUNet, self).__init__()
        
        # Ajout d'un canal d'entrée pour le bruit
        in_channels = in_channels + 1
        
        # Partie d'encodage : couche initiale
        self.m_head = conv(in_channels, nc[0], bias=False, mode="C")

        # Choix du bloc de souséchantillonnage (downsample)
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(f"downsample mode [{downsample_mode}] is not found")

        # Blocs convolutifs de l'encodeur avec souséchantillonnage 
        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode="2"),)
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode="2"),)
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode="2"),)

        # Partie intermédiaire : blocs convolutifs sans changement de résolution
        self.m_body = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C") for _ in range(nb)])

        # Choix du bloc de suréchantillonnage (upsample)
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode}] is not found")

        # Blocs convolutifs du décodeur avec suréchantillonnage
        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode="2"), *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],)
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode="2"), *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],)
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode="2"), *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],)

        # Couche finale de sortie
        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C")
        
        # Chargement des poids pré-entraînés si nécessaire
        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 4:
                    name = "drunet_deepinv_color_finetune_22k.pth"
                elif in_channels == 2:
                    name = "drunet_deepinv_gray_finetune_26k.pth"
                url = get_weights_url(model_name="drunet", file_name=name)
                ckpt_drunet = torch.hub.load_state_dict_from_url(url, map_location=lambda storage, loc: storage, file_name=name)
            else:
                ckpt_drunet = torch.load(pretrained, map_location=lambda storage, loc: storage)
            # Charger les poids dans le modèle
            self.load_state_dict(ckpt_drunet, strict=True)
            self.eval()
        else:
            # Initialisation des poids
            self.apply(weights_init_drunet)

        # Définir le dispositif d'exécution (GPU ou CPU)
        if device is not None:
            self.to(device)

    # Définition de la passe avant du réseau U-Net
    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        # Remonter les résolutions successivement
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x

    # Définition de la passe avant générale du réseau
    def forward(self, x, sigma):
        r"""
        Exécution du débruiteur sur une image avec un niveau de bruit donné (sigma).

        :param torch.Tensor x: Image bruitée.
        :param float, torch.Tensor sigma: Niveau de bruit.
        """
        # Créer une carte de niveau de bruit en fonction de sigma
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device) * sigma[None, None, None, None].to(x.device)
        else:
            noise_level_map = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device) * sigma
        
        # Ajouter la carte de niveau de bruit comme un canal d'entrée
        x = torch.cat((x, noise_level_map), 1)

        # Gestion de différentes tailles d'image
        if (x.size(2) % 8 == 0 and x.size(3) % 8 == 0 and x.size(2) > 31 and x.size(3) > 31):
            x = self.forward_unet(x)
        elif self.training or (x.size(2) < 32 or x.size(3) < 32):
            x = test_pad(self.forward_unet, x, modulo=16)
        else:
            x = test_onesplit(self.forward_unet, x, refield=64)
        return x

# Blocs fonctionnels supplémentaires

# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR

def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # Pas besoin de nn.Sequential.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# Retourne un nn.Sequential de (Conv + BN + ReLU)
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CBR", negative_slope=0.2,):
    """
    Crée une séquence d'opérations convolutionnelles, de normalisation et d'activation.
    
    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    :param kernel_size: Taille du noyau de la convolution.
    :param stride: Pas de la convolution.
    :param padding: Remplissage appliqué à l'entrée.
    :param bias: Utilisation d'un biais dans la couche convolutionnelle.
    :param mode: Combinaison des opérations à réaliser (ex: "CBR" pour Conv -> BatchNorm -> ReLU).
    :param negative_slope: Pente négative pour LeakyReLU.
    :return: Un nn.Sequential composé des opérations spécifiées.
    """
    L = []
    for t in mode:
        if t == "C":
            # Convolution
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,))
        elif t == "T":
            # Convolution transposée (utilisée pour suréchantillonner)
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,))
        elif t == "B":
            # Normalisation par lots
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            # Normalisation par instance
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            # Activation ReLU
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            # Activation ReLU sans modification en place
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            # Activation LeakyReLU
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            # Activation LeakyReLU sans modification en place
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "E":
            # Activation ELU
            L.append(nn.ELU(inplace=False))
        elif t == "s":
            # Activation Softplus
            L.append(nn.Softplus())
        elif t == "2":
            # Pixel Shuffle avec un facteur de suréchantillonnage de 2
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            # Pixel Shuffle avec un facteur de suréchantillonnage de 3
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            # Pixel Shuffle avec un facteur de suréchantillonnage de 4
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            # Suréchantillonnage par facteur de 2 avec interpolation au plus proche voisin
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            # Suréchantillonnage par facteur de 3 avec interpolation au plus proche voisin
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            # Suréchantillonnage par facteur de 4 avec interpolation au plus proche voisin
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            # MaxPooling
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            # AveragePooling
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)

# --------------------------------------------
# Bloc Résiduel: x + conv(ReLU(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    """
    Définition d'un bloc résiduel. 
    Il aide à l'apprentissage des différences d'une image par rapport à son état précédent, facilitant ainsi l'entraînement des réseaux profonds.
    
    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie (doit être égal à in_channels).
    :param kernel_size: Taille du noyau de la convolution.
    :param stride: Pas de la convolution.
    :param padding: Remplissage appliqué à l'entrée.
    :param bias: Utilisation d'un biais dans la couche convolutionnelle.
    :param mode: Combinaison des opérations à réaliser (ex: "CRC" pour Conv -> ReLU -> Conv).
    :param negative_slope: Pente négative pour LeakyReLU.
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CRC", negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        # Construction du bloc résiduel (Conv -> ReLU -> Conv)
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        """
        Passe avant du bloc résiduel.
        
        :param x: Tenseur d'entrée.
        :return: Tenseur avec la connexion résiduelle ajoutée.
        """
        res = self.res(x)
        return x + res

# --------------------------------------------
# Upsampler
# --------------------------------------------
# Définitions des différentes méthodes de suréchantillonnage.

# Pixel Shuffle (sous-échantillonnage en utilisant un facteur de pixel)
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    """
    Upsampling via Pixel Shuffle, Réorganise les pixels d'une image pour augmenter sa résolution.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    :param mode: Modes avec un facteur d'agrandissement ("2" pour doubler la taille, par exemple).
    """
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode="C" + mode, negative_slope=negative_slope)
    return up1

# Suréchantillonnage via interpolation au plus proche voisin suivi d'une convolution
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    """
    Upsampling via interpolation au plus proche voisin, puis application d'une convolution.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    :param mode: Modes avec un facteur d'agrandissement ("2", "3", ou "4").
    """
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"

    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1

# Convolution transposée, une méthode efficace pour suréchantillonner une image
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode="2R", negative_slope=0.2):
    """
    Upsampling via convolution transposée.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    :param kernel_size: Taille du noyau de la convolution (dépend du mode).
    :param stride: Pas de la convolution (dépend du mode).
    """
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."

    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1

# --------------------------------------------
# Downsampler
# --------------------------------------------
# Définitions des différentes méthodes de sous-échantillonnage.

# Sous-échantillonnage via une convolution avec stride supérieur à 1
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode="2R", negative_slope=0.2):
    """
    Sous-échantillonnage via une convolution avec un stride donné.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    :param stride: Pas de la convolution (doit être supérieur à 1).
    """
    assert len(mode) < 4 and mode[0] in ["2", "3", "4"], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1

# MaxPooling suivi d'une convolution pour sous-échantillonner
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode="2R", negative_slope=0.2):
    """
    Sous-échantillonnage via MaxPooling suivi d'une convolution.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    """
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)

# AveragePooling suivi d'une convolution pour sous-échantillonner
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    """
    Sous-échantillonnage via AveragePooling suivi d'une convolution.

    :param in_channels: Nombre de canaux d'entrée.
    :param out_channels: Nombre de canaux de sortie.
    """
    assert len(mode) < 4 and mode[0] in ["2", "3"], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)

# Fonction d'initialisation des poids
def weights_init_drunet(m):
    """
    Initialisation des poids des couches convolutionnelles du réseau DRUNet en utilisant l'initialisation orthogonale.

    :param m: Module dont les poids doivent être initialisés.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)


# code borrowed from Deepinv https://github.com/deepinv/deepinv/blob/main/deepinv/model/DnCNN

class DnCNN(nn.Module):
    r"""
    DnCNN convolutional denoiser.

    L'architecture a été introduite dans https://arxiv.org/abs/1608.03981 et est composée d'une série de couches
    convolutionnelles avec des fonctions d'activation ReLU. Le nombre de couches peut être spécifié par l'utilisateur.
    Contrairement à l'article original, cette implémentation n'inclut pas de couches de normalisation par lots.

    Le réseau peut être initialisé avec des poids préentraînés, qui peuvent être téléchargés depuis un dépôt en ligne.
    Les poids préentraînés sont entraînés avec les paramètres par défaut du réseau : 20 couches, 64 canaux et biais.

    :param int in_channels: Nombre de canaux de l'image d'entrée.
    :param int out_channels: Nombre de canaux de l'image de sortie.
    :param int depth: Nombre de couches convolutionnelles dans le réseau.
    :param bool bias: Utilisation de biais dans les couches convolutionnelles.
    :param int nf: Nombre de canaux par couche convolutionnelle.
    :param str, None pretrained: Utilisation d'un réseau préentraîné. Si "None", les poids sont initialisés de manière aléatoire. Si "download", les poids sont téléchargés en ligne.
    :param str device: Dispositif d'exécution, GPU ou CPU.
    """

    def __init__(self, in_channels=3, out_channels=3, depth=20, bias=True, nf=64, pretrained="download", device="cpu"):
        super(DnCNN, self).__init__()

        # Profondeur du réseau (nombre de couches convolutionnelles)
        self.depth = depth

        # Première couche de convolution
        self.in_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias)
        
        # Liste des couches convolutionnelles intermédiaires
        self.conv_list = nn.ModuleList([nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(self.depth - 2)])
       
        # Dernière couche de convolution pour la sortie
        self.out_conv = nn.Conv2d(nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # Liste des couches d'activation ReLU
        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        # Chargement des poids préentraînés si nécessaire
        if pretrained is not None:
            if pretrained.startswith("download"):
                name = ""
                if bias and depth == 20:
                    if pretrained == "download_lipschitz":
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_lipschitz_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_lipschitz_gray.pth"
                    else:
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_gray.pth"

                if name == "":
                    raise Exception("No pretrained weights were found online that match the chosen architecture")
                url = get_weights_url(model_name="dncnn", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(url, map_location=lambda storage, loc: storage, file_name=name)
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt, strict=True)
            self.eval()
        else:
            self.apply(weights_init_kaiming)

        # Définir le dispositif d'exécution (GPU ou CPU)
        if device is not None:
            self.to(device)

    def forward(self, x, sigma=None):
        r"""
        Exécution du débruiteur sur une image bruitée. Le niveau de bruit n'est pas utilisé dans ce débruiteur.

        :param torch.Tensor x: Image bruitée.
        :param float sigma: Niveau de bruit (non utilisé).
        """
        # Passe avant : application des couches convolutionnelles et des fonctions d'activation
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)

        # Application des couches intermédiaires
        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)

        # Couche de sortie et ajout de l'image bruitée à la sortie (skip connection)
        return self.out_conv(x1) + x

# Fonction d'initialisation des poids utilisant la méthode de Kaiming
def weights_init_kaiming(m):
    """
    Initialisation des poids des modules d'un réseau en utilisant l'initialisation de Kaiming.
    
    Parameters:
    ----------
    m : nn.Module
        Module dont les poids doivent être initialisés.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0.0)


# drunet_deepinv_color_finetune_22k.pth or drunet_deepinv_color.pth or drunet_color.pth"

