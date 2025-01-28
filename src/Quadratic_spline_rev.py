# Code borrowed from Alexis Goujon https://github.com/axgoujon/convex_ridge_regularizers

import torch
from torch import nn

class Quadratic_Spline_Func(torch.autograd.Function):
    """
    Fonction autograd pour calculer des splines quadratiques.
    Propagation des gradients uniquement à travers les coefficients pertinents.
    """

    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):
        """
        Forward pass : Calcule les splines quadratiques pour les entrées x.

        Args:
            ctx: Contexte pour sauvegarder les variables pour le backward pass.
            x (torch.Tensor): Entrées.
            coefficients_vect (torch.Tensor): Coefficients des splines.
            grid (torch.Tensor): Espacement entre les nœuds.
            zero_knot_indexes (torch.Tensor): Index de référence pour le nœud zéro.
            size (int): Nombre total de nœuds.

        Returns:
            torch.Tensor: Sorties après application des splines quadratiques.
        """
        # Clamp des valeurs d'entrée pour rester dans la plage des nœuds
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                            max=(grid.item() * (size // 2 - 2)))

        # Indices des coefficients pertinents
        floored_x = torch.floor(x_clamped / grid)
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Évaluation des bases B-spline pour l'interpolation
        shift1 = (x - floored_x * grid) / grid
        frac1 = ((shift1 - 1)**2) / 2
        frac2 = (-2 * (shift1)**2 + 2 * shift1 + 1) / 2
        frac3 = (shift1)**2 / 2

        # Combinaison pondérée des coefficients pour calculer la sortie
        activation_output = (
            coefficients_vect[indexes + 2] * frac3 +
            coefficients_vect[indexes + 1] * frac2 +
            coefficients_vect[indexes] * frac1
        )

        # Calcul du gradient par rapport à x
        grad_x = (
            coefficients_vect[indexes + 2] * shift1 +
            coefficients_vect[indexes + 1] * (1 - 2 * shift1) +
            coefficients_vect[indexes] * (shift1 - 1)
        ) / grid

        # Sauvegarde des variables nécessaires pour le backward pass
        ctx.save_for_backward(grad_x, frac1, frac2, frac3, coefficients_vect, indexes, grid)

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass : Calcule les gradients pour les entrées et les coefficients.

        Args:
            ctx: Contexte contenant les variables sauvegardées.
            grad_out (torch.Tensor): Gradients venant des couches suivantes.

        Returns:
            Tuple: Gradients pour les entrées et coefficients.
        """
        # Récupération des variables sauvegardées
        grad_x, frac1, frac2, frac3, coefficients_vect, indexes, grid = ctx.saved_tensors

        # Mise à jour des gradients par rapport à x
        grad_x *= grad_out

        # Initialisation des gradients pour les coefficients
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1) + 2, (frac3 * grad_out).view(-1))
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1) + 1, (frac2 * grad_out).view(-1))
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1), (frac1 * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None


class quadratic_spline(nn.Module):
    """
    Module PyTorch pour les splines quadratiques.

    Args:
        n_channels (int): Nombre de canaux ou d'activations.
        n_knots (int): Nombre de nœuds de la spline.
        T (float): Intervalle entre les nœuds.
    """

    def __init__(self, n_channels, n_knots, T):
        super().__init__()
        self.n_knots = n_knots
        self.n_channels = n_channels
        self.T = self.spline_grid_from_range(n_knots, T)
        print("Espacement des nœuds (grid step) :", self.T.item())

        # Initialisation des coefficients
        self.coefficients = torch.zeros((n_channels, n_knots), requires_grad=False)

        # Index du nœud zéro
        activation_arange = torch.arange(0, self.n_channels)
        self.zero_knot_indexes = activation_arange * self.n_knots + (self.n_knots // 2)

    def forward(self, input):
        """
        Applique la spline quadratique aux entrées.

        Args:
            input (torch.Tensor): Entrées 2D ou 4D.

        Returns:
            torch.Tensor: Sorties après application des splines.
        """
        assert input.size(1) == self.n_channels, f'{input.size(1)} != {self.n_channels}.'

        coefficients_vect = self.coefficients.view(-1)
        grid = self.T.to(coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        return Quadratic_Spline_Func.apply(input, coefficients_vect, grid, zero_knot_indexes, self.n_knots)

    def spline_grid_from_range(self, spline_size, spline_range, round_to=1e-6):
        """
        Calcule l'espacement entre les nœuds en fonction de la plage et du nombre de nœuds.

        Args:
            spline_size (int): Nombre total de nœuds (doit être impair).
            spline_range (float): Plage de la spline.
            round_to (float): Arrondi de l'espacement.

        Returns:
            torch.Tensor: Espacement entre les nœuds.
        """
        if spline_size % 2 == 0:
            raise ValueError('spline_size doit être un entier impair.')
        if spline_range <= 0:
            raise ValueError('spline_range doit être un nombre positif.')

        spline_grid = ((spline_range / (spline_size // 2)) // round_to) * round_to
        return torch.tensor(spline_grid)