# Code borrowed from Alexis Goujon https://https://github.com/axgoujon/convex_ridge_regularizers

import torch, math
from torch import nn
import torch.nn.utils.parametrize as P
from math import sqrt
import numpy as np

class MultiConv2d(nn.Module):
    """
    Module pour réaliser des multi-convolutions, c'est-à-dire une composition
    de plusieurs couches convolutives. 

    Ce module est utile pour les CRR-NNs et améliore l'entraînement, 
    notamment pour les couches avec des grands noyaux.

    Args:
        channels (list[int]): Liste des dimensions des canaux (entrée, intermédiaires, sortie).
            Exemple : [1, 8, 32] pour 2 couches convolutives (1 canal d'entrée, 8 canaux intermédiaires, 32 canaux de sortie).
        kernel_size (int): Taille des noyaux convolutifs.
        padding (int): Padding à appliquer (généralement `kernel_size // 2` pour préserver la taille spatiale).
    """

    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()

        # Paramètres
        self.padding = padding
        self.kernel_size = kernel_size
        self.channels = channels

        # Liste des couches convolutives
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1],
                    kernel_size=kernel_size, padding=self.padding, bias=False))

            # Applique une paramétrisation pour garantir une moyenne nulle
            P.register_parametrization(self.conv_layers[-1], "weight", ZeroMean())

        # Normalise les filtres pour garantir une norme spectrale initiale de 1
        self.initSN()

    def forward(self, x):
        """
        Forward pass : Applique les couches convolutives sur les entrées.

        Args:
            x (torch.Tensor): Entrées (4D : batch_size x channels x height x width).
        
        Returns:
            torch.Tensor: Sorties après application des convolutions.
        """
        return self.convolution(x)

    def convolution(self, x):
        """
        Applique séquentiellement toutes les couches convolutives.

        Args:
            x (torch.Tensor): Entrées.

        Returns:
            torch.Tensor: Sorties après convolutions.
        """
        for conv in self.conv_layers:
            x = nn.functional.conv2d(x, conv.weight, padding=self.padding, dilation=conv.dilation)
        return x

    def transpose(self, x):
        """
        Applique la transposée de l'opération convolutive.

        Cela garantit que la transposée est correctement calculée, ce qui est
        essentiel pour préserver la convexité des CRR-NNs.

        Args:
            x (torch.Tensor): Entrées.

        Returns:
            torch.Tensor: Sorties après convolutions transposées.
        """
        for conv in reversed(self.conv_layers):
            weight = conv.weight
            x = nn.functional.conv_transpose2d(x, weight, padding=conv.padding, groups=conv.groups, dilation=conv.dilation)
        return x

    def spectral_norm(self, n_power_iterations=10, size=40):
        """
        Calcule la norme spectrale des filtres via la méthode des puissances.

        Args:
            n_power_iterations (int): Nombre d'itérations pour approximer la norme spectrale.
            size (int): Taille spatiale des tensors utilisés pour l'approximation.

        Returns:
            float: Norme spectrale estimée.
        """
        # Initialise un vecteur aléatoire
        u = torch.empty((1, self.conv_layers[0].weight.shape[1], size, size),
            device=self.conv_layers[0].weight.device).normal_()

        with torch.no_grad():
            for _ in range(n_power_iterations):
                v = normalize(self.convolution(u))  # Applique les convolutions
                u = normalize(self.transpose(v))  # Applique la transposée

            # Produit scalaire pour estimer la norme spectrale
            cur_sigma = torch.sum(u * self.transpose(v))
            return cur_sigma

    def initSN(self):
        """
        Normalise les filtres pour garantir une norme spectrale initiale de 1.
        """
        with torch.no_grad():
            cur_sn = self.spectral_norm()
            for conv in self.conv_layers:
                conv.weight.data /= cur_sn ** (1 / len(self.conv_layers))

    def checkTranpose(self):
        """
        Vérifie si la transposée est correctement implémentée.

        Effectue un test en comparant la relation entre une opération convolutive
        et sa transposée.
        """
        x = torch.randn((1, 1, 40, 40), device=self.conv_layers[0].weight.device)
        Hx = self.forward(x)
        y = torch.randn((1, Hx.shape[1], 40, 40), device=self.conv_layers[0].weight.device)

        Hty = self.transpose(y)

        # Vérifie que les produits scalaires sont égaux (Hx·y ≈ x·Hty)
        v1 = torch.sum(Hx * y)
        v2 = torch.sum(x * Hty)

        print("Vérification de la transposée :", torch.max(torch.abs(v1 - v2)))

    
def normalize(tensor, eps=1e-12):
    """
    Normalise un tensor par sa norme \( L_2 \).

    Args:
        tensor (torch.Tensor): Tensor à normaliser.
        eps (float): Évite les divisions par zéro.

    Returns:
        torch.Tensor: Tensor normalisé.
    """
    norm = float(torch.sqrt(torch.sum(tensor**2)))
    norm = max(norm, eps)
    return tensor / norm

class ZeroMean(nn.Module):
    """
    Paramétrisation garantissant une moyenne nulle pour les noyaux convolutifs.

    Cela est essentiel pour éviter des biais systématiques dans les résultats.
    """

    def forward(self, X):
        """
        Soustrait la moyenne de chaque noyau convolutif.

        Args:
            X (torch.Tensor): Poids des noyaux convolutifs.

        Returns:
            torch.Tensor: Noyaux ajustés pour avoir une moyenne nulle.
        """
        return X - torch.mean(X, dim=(1, 2, 3), keepdim=True)

