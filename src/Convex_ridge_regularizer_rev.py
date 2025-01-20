# Code borrowed from Alexis Goujon https://https://github.com/axgoujon/convex_ridge_regularizers

import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from Multi_conv_rev import MultiConv2d
from Linear_spline_rev import LinearSpline


class ConvexRidgeRegularizer(nn.Module):
    """
    Module pour paramétrer un modèle CRR-NN (Convex Ridge Regularizer Neural Network).
    Ce modèle est principalement axé sur les gradients pour résoudre des problèmes inverses.
    """

    def __init__(self, channels=[1, 8, 32], kernel_size=3, activation_params={"knots_range": 0.1, "n_knots": 21}):
        """
        Initialise le modèle CRR-NN.

        Args:
            channels (list[int]): Liste des canaux (entrée, intermédiaires, sortie).
                Par exemple : [1, 8, 32] pour une entrée avec 1 canal, 32 canaux en sortie, et 8 canaux intermédiaires.
            kernel_size (int): Taille du noyau des convolutions.
            activation_params (dict): Paramètres pour les fonctions d'activation (ex. plage des nœuds).
        """
        super().__init__()

        # Calcul du padding pour préserver la taille spatiale des convolutions
        padding = kernel_size // 2
        self.padding = padding
        self.channels = channels

        # Régularisation avec des paramètres apprenants
        self.lmbd = nn.parameter.Parameter(data=torch.tensor(5.), requires_grad=True)  # Poids de régularisation
        self.mu = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=True)  # Échelle de régularisation

        # Couche linéaire composée de convolutions multiples
        self.conv_layer = MultiConv2d(channels=channels, kernel_size=kernel_size, padding=padding)

        # Paramètres des fonctions d'activation
        self.activation_params = activation_params
        activation_params["n_channels"] = channels[-1]

        # Détermine la fonction d'activation à utiliser
        if "name" not in activation_params:
            activation_params["name"] = "spline"

        if activation_params["name"] == "ReLU":
            # Utilise ReLU comme fonction d'activation
            self.activation = nn.ReLU()
            self.bias = nn.parameter.Parameter(data=torch.zeros((1, channels[-1], 1, 1)), requires_grad=True)
            self.use_splines = False
            self.lmbd.data *= 1e-3  # Ajuste le poids de régularisation pour ReLU
        else:
            # Utilise des splines linéaires comme fonction d'activation
            self.activation = LinearSpline(mode="conv", num_activations=activation_params["n_channels"],
                size=activation_params["n_knots"], range_=activation_params["knots_range"])
            self.use_splines = True

        # Nombre total de paramètres dans le modèle
        self.num_params = sum(p.numel() for p in self.parameters())

        # Initialise un vecteur pour estimer la borne de Lipschitz
        self.initializeEigen(size=20)

        # Estimation initiale de la borne de Lipschitz
        self.L = nn.parameter.Parameter(data=torch.tensor(1.), requires_grad=False)

        # Affiche des informations sur le modèle
        print("---------------------")
        print(f"Construction d'un modèle CRR-NN avec \n - {channels} canaux \n Paramètres des splines :")
        print(f"  ({self.activation})")
        print("---------------------")

    def initializeEigen(self, size=100):
        """
        Initialise un vecteur propre pour estimer la borne de Lipschitz.
        """
        self.u = torch.empty((1, 1, size, size)).uniform_()

    @property
    def lmbd_transformed(self):
        """
        Garantit que lambda (\( \lambda \)) est strictement positif.
        """
        return torch.clip(self.lmbd, 0.0001, None)

    @property
    def mu_transformed(self):
        """
        Garantit que mu (\( \mu \)) est strictement positif.
        """
        return torch.clip(self.mu, 0.01, None)

    def forward(self, x):
        """
        Implémente le passage avant du modèle.

        Args:
            x (torch.Tensor): Entrée (batch_size x channels x height x width).

        Returns:
            torch.Tensor: Sortie après traitement convolutif et activation.
        """
        # Couche convolutive
        y = self.conv_layer(x)

        # Applique la fonction d'activation
        if not self.use_splines:
            y = y + self.bias
        y = self.activation(y)

        # Couche convolutive transposée
        y = self.conv_layer.transpose(y)

        return y

    def grad(self, x):
        """
        Calcule le gradient pour \( x \).

        Args:
            x (torch.Tensor): Entrée.

        Returns:
            torch.Tensor: Gradient calculé.
        """
        return self.forward(x)

    def update_integrated_params(self):
        """
        Met à jour les paramètres intégrés des splines.
        """
        for ac in self.activation:
            ac.update_integrated_coeff()

    def cost(self, x):
        """
        Calcule le coût associé à une entrée \( x \).

        Args:
            x (torch.Tensor): Entrée.

        Returns:
            torch.Tensor: Coût calculé.
        """
        s = x.shape

        # Couche convolutive initiale
        y = self.conv_layer(x)

        # Intègre les splines
        y = self.activation.integrate(y)

        # Somme des coûts sur les dimensions restantes
        return torch.sum(y, dim=tuple(range(1, len(s))))

    # regularization
    def TV2(self, include_weights=False):
        """
        Calcule la régularisation de variation totale de second ordre (TV2).

        Args:
            include_weights (bool): Si True, inclut les poids dans le calcul.

        Returns:
            torch.Tensor: Valeur de la régularisation.
        """
        if self.use_splines:
            return self.activation.TV2(include_weights=include_weights)
        else:
            return 0

    
    def precise_lipschitz_bound(self, n_iter=50, differentiable=False):
        """
        Calcule une borne précise pour la constante de Lipschitz via des itérations de puissance.

        Args:
            n_iter (int): Nombre d'itérations pour estimer la borne.
            differentiable (bool): Si True, rend la borne différentiable.

        Returns:
            torch.Tensor: Borne de Lipschitz estimée.
        """
        with torch.no_grad():
            # Vérifie et utilise les pentes maximales des splines
            if self.use_splines:
                slope_max = self.activation.slope_max
                if slope_max.max().item() == 0:
                    return torch.tensor([0.], device=slope_max.device)

            # Itérations de puissance pour estimer la valeur propre dominante
            self.u = self.u.to(self.conv_layer.conv_layers[0].weight.device)
            u = self.u
            for _ in range(n_iter - 1):
                u = normalize(u)  # Normalisation
                u = self.conv_layer.forward(u)  # Applique la convolution
                if self.use_splines:
                    u *= slope_max.view(1, -1, 1, 1)  # Applique les pentes maximales
                u = self.conv_layer.transpose(u)  # Convolution transposée

            sigma_estimate = norm(u)  # Norme finale

        self.u = u  # Met à jour le vecteur propre
        return sigma_estimate


    @property
    def device(self):
        """
        Retourne le périphérique (GPU ou CPU) où se trouvent les poids du modèle.
        """
        return self.conv_layer.conv_layers[0].weight.device

    def prune(self, tol=1e-4, prune_filters=True, collapse_filters=False, change_splines_to_clip=False):
        """
        Réduit le modèle pour améliorer son efficacité (uniquement pour les tests).

        Args:
            tol (float): Seuil pour supprimer les filtres non significatifs.
            prune_filters (bool): Si True, supprime les filtres faibles.
            collapse_filters (bool): Si True, combine toutes les convolutions en une seule.
            change_splines_to_clip (bool): Si True, remplace les splines par des fonctions clip.

        Returns:
            None
        """
        device = self.conv_layer.conv_layers[0].weight.device

        if collapse_filters:
            # Combine toutes les convolutions en une seule
            new_padding = sum([conv.kernel_size[0] // 2 for conv in self.conv_layer.conv_layers])
            new_kernel_size = 2 * new_padding + 1
            impulse = torch.zeros((1, 1, new_kernel_size, new_kernel_size), device=device, requires_grad=False)
            impulse[0, 0, new_kernel_size // 2, new_kernel_size // 2] = 1

            new_kernel = self.conv_layer.convolution(impulse)
            new_conv_layer = MultiConv2d(channels=[1, new_kernel.shape[1]], kernel_size=self.channels[-1], padding=new_padding)
            new_conv_layer.conv_layers[0].parametrizations.weight.original.data = new_kernel.permute(1, 0, 2, 3)

            self.conv_layer = new_conv_layer
            self.channels = [1, new_kernel.shape[1]]
            self.padding = new_padding

        if prune_filters:
            # Supprime les filtres non significatifs
            new_padding = sum([conv.kernel_size[0] // 2 for conv in self.conv_layer.conv_layers])
            new_kernel_size = 2 * new_padding + 1
            impulse = torch.zeros((1, 1, new_kernel_size, new_kernel_size), device=device, requires_grad=False)
            impulse[0, 0, new_kernel_size // 2, new_kernel_size // 2] = 1

            new_kernel = self.conv_layer.convolution(impulse)
            kernel_norm = torch.sum(new_kernel**2, dim=(0, 2, 3))

            coeff = self.activation.projected_coefficients
            slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.activation.grid.item()
            tv2 = torch.sum(torch.abs(slopes[:, 1:-1]), dim=1)

            weight = tv2 * kernel_norm
            l_keep = torch.where(weight > tol)[0]

            print("---------------------")
            print(f"PRUNNING : {len(l_keep)} filtres conservés.")
            print("---------------------")

            # Mise à jour des coefficients des splines
            new_spline_coeff = torch.clone(
                self.activation.coefficients_vect.view(self.activation.num_activations, self.activation.size)[l_keep, :]
            ).contiguous().view(-1)
            self.activation.coefficients_vect.data = new_spline_coeff
            self.activation.num_activations = len(l_keep)
            self.activation.grid_tensor = torch.linspace(-self.activation.range_, self.activation.range_, self.activation.size).expand((self.activation.num_activations, self.activation.size))
            self.activation.init_zero_knot_indexes()

            # Mise à jour des convolutions
            self.conv_layer.conv_layers[-1].parametrizations.weight.original.data = self.conv_layer.conv_layers[-1].parametrizations.weight.original.data[l_keep, :, :, :]
            self.channels[-1] = len(l_keep)

        if change_splines_to_clip:
            self.activation = self.activation.get_clip_equivalent()

        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Nombre de paramètres après pruning : {self.num_params}")


def norm(u):
    """
    Calcule la norme \( L_2 \) d'un tensor.

    Args:
        u (torch.Tensor): Tensor d'entrée.

    Returns:
        torch.Tensor: Norme calculée.
    """
    return torch.sqrt(torch.sum(u**2))

def normalize(u):
    """
    Normalise un tensor par sa norme \( L_2 \).

    Args:
        u (torch.Tensor): Tensor à normaliser.

    Returns:
        torch.Tensor: Tensor normalisé.
    """
    return u / norm(u)
