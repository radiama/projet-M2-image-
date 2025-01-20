# Code borrowed from Alexis Goujon https://https://github.com/axgoujon/convex_ridge_regularizers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod
from Quadratic_spline_rev import Quadratic_Spline_Func


def monotonic_clipping(cs):
    """
    Projeter les coefficients de spline en garantissant la monotonie.

    Args:
        cs (torch.Tensor): Coefficients des splines (shape: batch_size, n).

    Returns:
        torch.Tensor: Coefficients ajustés garantissant la monotonie.
    """
    # Calcul des pentes projetées en garantissant leur positivité
    slopes = torch.clamp(cs[:, 1:] - cs[:, :-1], min=0)

    # Reconstruction cumulative des coefficients avec des pentes ajustées
    new_cs = torch.cat([torch.zeros(cs.size(0), 1, device=cs.device), torch.cumsum(slopes, dim=1)], dim=1)

    # Ajustement global pour centrer la spline (valeur au milieu égale à 0)
    new_cs -= new_cs[:, new_cs.size(1) // 2].unsqueeze(1)

    return new_cs


def initialize_coeffs(init, grid_tensor, grid):
    """
    Initialise les coefficients des splines avec validation simplifiée.

    Args:
        init (str): Méthode d'initialisation ('identity', 'zero', 'relu').
        grid_tensor (torch.Tensor): Tensor des nœuds, taille (batch_size, n).

    Returns:
        torch.Tensor: Coefficients initialisés.

    Raises:
        ValueError: Si la méthode d'initialisation n'est pas valide.
    """
    # Dictionnaire pour mapper les méthodes d'initialisation
    init_methods = {
        'identity': lambda x: x,
        'zero': lambda x: torch.zeros_like(x),
        'relu': lambda x: F.relu(x),
    }

    # Vérification de la validité de l'initialisation
    if init not in init_methods:
        raise ValueError(f"Invalid init method '{init}'. Choose from {list(init_methods.keys())}.")

    # Appliquer l'initialisation choisie
    return init_methods[init](grid_tensor)


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function pour le calcul des splines linéaires.

    """

    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even):
        """
        Forward pass : Calcul de la spline linéaire pour les entrées x.

        Args:
            ctx: Contexte pour sauvegarder les variables pour le backward pass.
            x (torch.Tensor): Entrées.
            coefficients_vect (torch.Tensor): Coefficients des splines.
            grid (float): Espacement entre les nœuds.
            zero_knot_indexes (int): Index de référence des nœuds.
            size (int): Nombre total de nœuds.
            even (bool): Si la grille est paire.

        Returns:
            torch.Tensor: Sortie après application des splines linéaires.
        """
        # Ajuster pour grilles paires si nécessaire
        if even:
            x = x - grid / 2
            max_range = grid * (size // 2 - 2)
        else:
            max_range = grid * (size // 2 - 1)

        # Clamp des valeurs pour rester dans les limites
        x_clamped = x.clamp(min=-grid * (size // 2), max=max_range)

        # Calcul des indices des coefficients pertinents
        floored_x = torch.floor(x_clamped / grid)
        fracs = x / grid - floored_x  # Distance fractionnaire
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Interpolation linéaire
        activation_output = (coefficients_vect[indexes + 1] * fracs + coefficients_vect[indexes] * (1 - fracs))

        # Ajustement pour grille paire
        if even:
            activation_output += grid / 2

        # Sauvegarde pour le backward pass
        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass : Calcul des gradients par rapport aux entrées et coefficients.

        Args:
            ctx: Contexte contenant les variables sauvegardées.
            grad_out (torch.Tensor): Gradients provenant des couches suivantes.

        Returns:
            Tuple: Gradients par rapport aux entrées et coefficients.
        """
        # Récupération des variables sauvegardées
        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors

        # Gradient par rapport à l'entrée x
        grad_x = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid * grad_out

        # Gradient par rapport aux coefficients
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1) + 1, (fracs * grad_out).view(-1))
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None


class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Fonction autograd pour calculer la dérivée des splines linéaires.

    """

    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even):
        """
        Forward pass : Calcul des dérivées des splines pour les entrées x.

        Args:
            ctx: Contexte pour sauvegarder les variables pour le backward pass.
            x (torch.Tensor): Entrées.
            coefficients_vect (torch.Tensor): Coefficients des splines.
            grid (float): Espacement entre les nœuds.
            zero_knot_indexes (torch.Tensor): Index de référence des nœuds.
            size (int): Nombre total de nœuds.
            even (bool): Indique si la grille est paire.

        Returns:
            torch.Tensor: Dérivée des splines.
        """
        # Ajustement pour les grilles paires
        if even:
            x -= grid / 2
            max_range = grid * (size // 2 - 2)
        else:
            max_range = grid * (size // 2 - 1)

        # Clamp des valeurs dans la plage admissible
        x_clamped = x.clamp(min=-grid * (size // 2), max=max_range)

        # Calcul des indices des coefficients
        floored_x = torch.floor(x_clamped / grid)
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Calcul de la dérivée via la différence des coefficients
        activation_output = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid.item()

        # Sauvegarde des variables pour backward pass
        ctx.save_for_backward(coefficients_vect, indexes, grid)

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass : Calcul des gradients.

        Args:
            ctx: Contexte contenant les variables sauvegardées.
            grad_out (torch.Tensor): Gradients de la couche suivante.

        Returns:
            Tuple: Gradients des entrées et coefficients.
        """
        # Récupération des variables sauvegardées
        coefficients_vect, indexes, grid = ctx.saved_tensors

        # Gradients par rapport aux entrées (ici, toujours 0)
        grad_x = torch.zeros_like(grad_out)

        # Initialisation des gradients des coefficients
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)

        # Retourne les gradients
        return grad_x, grad_coefficients_vect, None, None, None, None


class LinearSpline(ABC, nn.Module):
    """
    Classe pour les fonctions d'activation basées sur des splines linéaires.

    Args:
        mode (str): 'conv' pour convolution ou 'fc' pour fully-connected.
        num_activations (int): Nombre de fonctions d'activation.
        size (int): Nombre de coefficients des splines, avec K = size - 2 nœuds.
        range_ (float): Intervalle positif pour l'expansion des splines B, [−range_, range_].
        init (str): Méthode d'initialisation ('relu', 'identity', 'zero').
        monotonic_constraint (bool): Contrainte garantissant une augmentation monotone.
    """

    def __init__(self, mode, num_activations, size, range_, init="zero", monotonic_constraint=True, **kwargs):
        # Vérification des paramètres
        if mode not in ['conv', 'fc']:
            raise ValueError('Mode doit être "conv" ou "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations doit être un entier positif.')

        super().__init__()

        # Initialisation des attributs
        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0  # Détecte si la taille est paire
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)

        # Espacement entre les nœuds (grid)
        grid = 2 * self.range_ / float(self.size - 1)
        self.grid = torch.Tensor([grid])

        # Initialisation des index pour le nœud zéro
        self.init_zero_knot_indexes()

        # Filtre pour les différences finies de second ordre
        self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)

        # Contrainte de monotonie
        self.monotonic_constraint = monotonic_constraint

        # Coefficients intégrés
        self.integrated_coeff = None

        # Grille des coefficients
        self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))

        # Initialisation des coefficients des splines
        coefficients = initialize_coeffs(init, self.grid_tensor, self.grid)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))  # Coefficients vectorisés

    def init_zero_knot_indexes(self):
        """Initialise les index des nœuds zéro pour chaque activation."""
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = activation_arange * self.size + (self.size // 2)

    @property
    def coefficients(self):
        """Retourne les coefficients des splines sous forme matricielle."""
        return self.coefficients_vect.view(self.num_activations, self.size)

    @property
    def projected_coefficients(self):
        """Projette les coefficients pour satisfaire les contraintes."""
        if self.monotonic_constraint:
            return self.monotonic_coefficients
        else:
            return self.coefficients
    
    @property
    def projected_coefficients_vect(self):
        """ B-spline coefficients projected to meet the constraint. """
        return self.projected_coefficients.contiguous().view(-1)

    @property
    def monotonic_coefficients(self):
        """Projette les coefficients pour garantir une monotonie."""
        return monotonic_clipping(self.coefficients)

    def cache_constraint(self):
        """Applique la contrainte de monotonie aux coefficients après l'entraînement."""
        if self.monotonic_constraint:
            with torch.no_grad():
                self.coefficients_vect.data = self.monotonic_coefficients_vect.data
                self.monotonic_constraint = False

    def forward(self, x):
        """
        Calcul direct (forward pass).

        Args:
            x (torch.Tensor): Entrée (2D ou 4D selon 'fc' ou 'conv').

        Returns:
            torch.Tensor: Sortie après application de la spline.
        """
        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)
        coeff_vect = self.projected_coefficients_vect
        return LinearSpline_Func.apply(x, coeff_vect, grid, zero_knot_indexes, self.size, self.even)

    def derivative(self, x):
        """
        Calcul de la dérivée de la spline.

        Args:
            x (torch.Tensor): Entrée (2D ou 4D selon 'fc' ou 'conv').

        Returns:
            torch.Tensor: Dérivée des splines pour chaque point d'entrée.
        """
        assert x.size(1) == self.num_activations, "Erreur : mauvaise taille d'entrée."
        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)
        coeff_vect = self.projected_coefficients_vect
        return LinearSplineDerivative_Func.apply(x, coeff_vect, grid, zero_knot_indexes, self.size, self.even)

    def update_integrated_coeff(self):
        """
        Met à jour les coefficients intégrés pour le calcul de la régularisation.
        """
        coeff = self.projected_coefficients

        # Ajout des coefficients extrapolés
        coeff_int = torch.cat((coeff[:, 0:1], coeff, coeff[:, -1:]), dim=1)

        # Intégration cumulative pour les coefficients quadratiques
        self.integrated_coeff = torch.cumsum(coeff_int, dim=1) * self.grid.to(coeff.device)

        # Réglage pour garantir 0 au centre
        self.integrated_coeff = (self.integrated_coeff - self.integrated_coeff[:, (self.size + 2) // 2].view(-1, 1)).view(-1)

        # Mise à jour des index des nœuds zéro pour l'intégration
        self.zero_knot_indexes_integrated = (torch.arange(0, self.num_activations) * (self.size + 2) + ((self.size + 2) // 2))

    def integrate(self, x):
        """
        Intègre les splines.

        Args:
            x (torch.Tensor): Entrée pour intégration.

        Returns:
            torch.Tensor: Résultat de l'intégration.
        """
        if self.integrated_coeff is None:
            self.update_integrated_coeff()

        if x.device != self.integrated_coeff.device:
            self.integrated_coeff = self.integrated_coeff.to(x.device)

        return Quadratic_Spline_Func.apply(
            x - self.grid.to(x.device),
            self.integrated_coeff,
            self.grid.to(x.device),
            self.zero_knot_indexes_integrated.to(x.device),
            (self.size + 2)
        )

    def extra_repr(self):
        """Retourne une représentation pour print(model)."""
        return (f'mode={self.mode}, num_activations={self.num_activations}, '
                f'init={self.init}, size={self.size}, grid={self.grid[0]:.3f}, '
                f'monotonic_constraint={self.monotonic_constraint}.')


    def TV2(self, ignore_tails=False, **kwargs):
        """
        Calcule la régularisation de variation totale de second ordre (TV(2)).

        La régularisation est définie comme la somme des normes des pentes
        associées aux segments de la spline linéaire.

        Args:
            ignore_tails (bool): Si True, ignore les pentes des segments aux extrémités.
        
        Returns:
            torch.Tensor: La somme des normes des pentes (||a||_1).
        """
        if ignore_tails:
            # Ignore les pentes des segments aux extrémités
            return torch.sum(self.relu_slopes[:, 1:-1].norm(1, dim=1))
        else:
            # Prend en compte toutes les pentes
            sl = self.relu_slopes
            return torch.sum(sl.norm(1, dim=1))


    def TV2_vec(self, ignore_tails=False, p=1, **kwargs):
        """
        Calcule la régularisation de variation totale de second ordre (TV(2)) avec une norme personnalisable.

        Args:
            ignore_tails (bool): Si True, ignore les pentes des segments aux extrémités.
            p (int): La norme p à utiliser (par défaut, norme 1).

        Returns:
            torch.Tensor: La régularisation TV(2) calculée avec la norme p.
        """
        if ignore_tails:
            # Ignore les pentes des segments aux extrémités
            return torch.sum(self.relu_slopes[:, 1:-1].norm(p, dim=1))
        else:
            # Prend en compte toutes les pentes
            return self.relu_slopes.norm(p, dim=1)


    @property
    def slope_max(self):
        """
        Retourne la pente maximale parmi tous les segments de la spline.

        Calcule la différence entre les coefficients consécutifs pour chaque segment,
        divisée par l'espacement entre les nœuds.

        Returns:
            torch.Tensor: Valeurs maximales des pentes pour chaque fonction d'activation.
        """
        # Calcul des pentes entre les coefficients adjacents
        coeff = self.projected_coefficients
        slope = (coeff[:, 1:] - coeff[:, :-1]) / self.grid.item()

        # Retourne la pente maximale pour chaque activation
        slope_max = torch.max(slope, dim=1)[0]
        return slope_max


    def get_clip_equivalent(self):
        """
        Représente les splines comme une somme de deux ReLU.

        Cette méthode est pertinente uniquement pour les splines qui ressemblent
        à une fonction clip (fonction saturée avec des pentes constantes).

        Returns:
            clip_activation: Fonction clip équivalente représentée par deux ReLU.
        """
        # Copie des coefficients projetés pour éviter les modifications accidentelles
        coeff_proj = self.projected_coefficients.clone().detach()

        # Calcul des pentes entre les coefficients adjacents
        slopes = (coeff_proj[:, 1:] - coeff_proj[:, :-1])

        # Changements dans les pentes pour identifier les points de clip
        slopes_change = slopes[:, 1:] - slopes[:, :-1]

        # Trouve les indices des points où les pentes changent (deux maximums)
        i1 = torch.max(slopes_change, dim=1)
        i2 = torch.min(slopes_change, dim=1)

        # Indices pour sélectionner les points pertinents
        i0 = torch.arange(0, coeff_proj.shape[0]).to(coeff_proj.device)

        # Extraction des points de clip (x1, y1) et (x2, y2)
        self.grid_tensor = self.grid_tensor.to(coeff_proj.device)
        x1 = self.grid_tensor[i0, i1[1] + 1].view(1, -1, 1, 1)
        y1 = coeff_proj[i0, i1[1] + 1].view(1, -1, 1, 1)

        x2 = self.grid_tensor[i0, i2[1] + 1].view(1, -1, 1, 1)
        y2 = coeff_proj[i0, i2[1] + 1].view(1, -1, 1, 1)

        # Calcul des pentes entre les points de clip
        slopes = ((y2 - y1) / (x2 - x1)).view(1, -1, 1, 1)

        # Création de la fonction clip équivalente
        cl = clip_activation(x1, x2, y1, slopes)

        return cl

class clip_activation(nn.Module):
    """
    Fonction d'activation de type "clip" définie par deux limites x1 et x2.

    Args:
        x1 (torch.Tensor): Limite inférieure où la pente commence.
        x2 (torch.Tensor): Limite supérieure où la pente s'arrête.
        y1 (torch.Tensor): Décalage initial.
        slopes (torch.Tensor): Pente entre x1 et x2.
    """

    def __init__(self, x1, x2, y1, slopes):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.slopes = torch.nan_to_num(slopes)  # Gérer les NaN pour éviter les erreurs

    def forward(self, x):
        """
        Applique la fonction clip aux entrées x.

        Args:
            x (torch.Tensor): Entrées.

        Returns:
            torch.Tensor: Sorties après application de la fonction clip.
        """
        # Combinaison des segments ReLU pour créer la fonction clip
        return self.slopes * (torch.relu(x - self.x1) - torch.relu(x - self.x2)) + self.y1

    def integrate(self, x):
        """
        Intègre la fonction clip pour produire une spline quadratique.

        Args:
            x (torch.Tensor): Entrées.

        Returns:
            torch.Tensor: Sorties intégrées.
        """
        # Intégration des segments ReLU au carré
        return (self.slopes / 2) * (torch.relu(x - self.x1)**2 - torch.relu(x - self.x2)**2) + self.y1 * x

    def slope_max(self):
        """
        Retourne la pente maximale parmi tous les segments actifs.

        Returns:
            torch.Tensor: Pente maximale.
        """
        return torch.max(self.slopes)