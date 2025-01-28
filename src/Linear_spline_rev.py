import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod
from Quadratic_spline_rev import Quadratic_Spline_Func


def monotonic_clipping(cs):
    """Projection simple des coefficients du spline pour obtenir un spline linéaire monotone"""
    device = cs.device
    n = cs.shape[1]
    # obtenir les pentes projetées
    new_slopes = torch.clamp(cs[:,1:] - cs[:,:-1], 0, None)
    # extension du clampage
    new_slopes[:,0] = 0
    new_slopes[:,-1] = 0
    
    # construire les nouveaux coefficients
    new_cs = torch.zeros(cs.shape, device=device)
    new_cs[:,1:] = torch.cumsum(new_slopes, dim=1)

    # fixer zéro au point central
    new_cs = new_cs + (-new_cs[:,new_cs.shape[1]//2]).unsqueeze(1)

    return new_cs

def initialize_coeffs(init, grid_tensor, grid):
    """Les coefficients sont initialisés avec la valeur de l'activation
    # à chaque nœud (c[k] = f[k], puisque les splines B1 sont des interpolateurs)."""
        
    if init == 'identity':
        coefficients = grid_tensor
    elif init == 'zero':
        coefficients = grid_tensor * 0
    elif init == 'relu':
        coefficients = F.relu(grid_tensor)       
    else:
        raise ValueError("init doit être dans ['identity', 'relu', 'absolute_value', 'maxmin', 'max_tv'].")
    
    return coefficients


class LinearSpline_Func(torch.autograd.Function):
    """
    Fonction Autograd permettant de rétropropager uniquement à travers les B-splines
    qui ont été utilisées pour calculer la sortie = activation(entrée),
    pour chaque élément de l'entrée.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even, train=True):

        # La valeur du spline en n'importe quel x est une combinaison 
        # d'au plus deux coefficients
        max_range = (grid.item() * (size // 2 - 1))
        if even:
            x = x - grid / 2
            max_range = (grid.item() * (size // 2 - 2))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  # coefficient gauche
        # fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance au coefficient gauche

        # Cela donne les indices (dans coefficients_vect) des coefficients de gauche
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Seulement deux fonctions de base B-spline sont nécessaires pour calculer la sortie
        # (via l'interpolation linéaire) pour chaque entrée dans la plage du B-spline.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)
        if even:
            activation_output = activation_output + grid / 2

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Ensuite, ajouter les gradients par rapport à chaque coefficient, de sorte que,
        # pour chaque point de données, seuls les gradients par rapport aux deux
        # coefficients les plus proches sont ajoutés (puisque seuls ceux-ci peuvent être non nuls).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # gradients des coefficients de droite
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # gradients des coefficients de gauche
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None


class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Fonction Autograd permettant de rétropropager uniquement à travers les B-splines
    qui ont été utilisées pour calculer la sortie = activation(entrée),
    pour chaque élément de l'entrée.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even):

        # La valeur du spline en n'importe quel x est une combinaison 
        # d'au plus deux coefficients
        max_range = (grid.item() * (size // 2 - 1))
        if even:
            x = x - grid / 2
            max_range = (grid.item() * (size // 2 - 2))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  # coefficient gauche
        # fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance au coefficient gauche

        # Cela donne les indices (dans coefficients_vect) des coefficients de gauche
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Seulement deux fonctions de base B-spline sont nécessaires pour calculer la sortie
        # (via l'interpolation linéaire) pour chaque entrée dans la plage du B-spline.
        activation_output = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid.item()
        if even:
            activation_output = activation_output + grid / 2

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = 0 * grad_out

        # Ensuite, ajouter les gradients par rapport à chaque coefficient, de sorte que,
        # pour chaque point de données, seuls les gradients par rapport aux deux
        # coefficients les plus proches sont ajoutés (puisque seuls ceux-ci peuvent être non nuls).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # gradients des coefficients de droite
        # grad_coefficients_vect.scatter_add_(0,
        #                                     indexes.view(-1) + 1,
        #                                     (fracs * grad_out).view(-1))
        # gradients des coefficients de gauche
        # grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
        #                                     ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None


class LinearSpline(ABC, nn.Module):
    """
    Classe pour les fonctions d'activation LinearSpline

    Args:
        mode (str): 'conv' (convolutionnel) ou 'fc' (fully-connected / entièrement connecté).
        num_activations (int) : nombre de fonctions d'activation.
        size (int): nombre de coefficients de la grille du spline ; le nombre de nœuds K = size - 2.
        range_ (float) : plage positive de l'expansion du B-spline. Plage des B-splines = [-range_, range_].
        init (str): Fonction pour initialiser les activations ('relu', 'identity', 'zero').
        monotonic_constraint (bool): Contraindre l'activation à être monotone croissante.
    """

    def __init__(self, mode, num_activations, size, range_, init="zero", monotonic_constraint=True, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode doit être soit "conv" soit "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations doit être un entier positif...')

        super().__init__()
        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        grid = 2 * self.range_ / float(self.size-1)
        self.grid = torch.Tensor([grid])

        self.init_zero_knot_indexes()
        self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)
        self.monotonic_constraint = monotonic_constraint

        self.integrated_coeff = None

        # tenseur contenant les positions des coefficients du spline
        self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))
        coefficients = initialize_coeffs(init, self.grid_tensor, self.grid)  # coefficients du spline
        # Nécessite de vectoriser les coefficients pour effectuer certaines opérations spécifiques
        # taille : (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))


    def init_zero_knot_indexes(self):
        """ Initialise les indices des nœuds zéro de chaque activation.
        """
        # self.zero_knot_indexes[i] donne l'indice du nœud 0 pour le filtre/neuron_i.
        # taille : (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size +
                                  (self.size // 2))

    @property
    def coefficients(self):
        """ Coefficients des B-splines. """
        return self.coefficients_vect.view(self.num_activations, self.size)

    @property
    def projected_coefficients(self):
        """ Coefficients des B-splines projetés pour respecter la contrainte. """
        if self.monotonic_constraint:
            return self.monotonic_coefficients
        else:
            return self.coefficients

    @property
    def projected_coefficients_vect(self):
        """ Coefficients des B-splines projetés pour respecter la contrainte. """
        return self.projected_coefficients.contiguous().view(-1)

    @property
    def monotonic_coefficients(self):
        """Projection des coefficients des B-splines de sorte que le spline soit croissant."""
        return monotonic_clipping(self.coefficients)

    @property
    def relu_slopes(self):
        """ Obtenir les pentes d'activation ReLU {a_k},
        en effectuant une convolution valide des coefficients {c_k}
        avec le filtre de différences finies du second ordre [1,-2,1].
        """
        D2_filter = self.D2_filter.to(device=self.coefficients.device)

        coeff = self.projected_coefficients

        slopes = F.conv1d(coeff.unsqueeze(1), D2_filter).squeeze(1)
        return slopes
    
    @property
    def monotonic_coefficients_vect(self):
        """Projection des coefficients des B-splines de sorte qu'ils soient croissants."""
        return self.monotonic_coefficients.contiguous().view(-1)


    def cache_constraint(self):
        """ Mettre à jour les coefficients avec ceux contraints, après l'entraînement. """
        if self.monotonic_constraint:
            with torch.no_grad():
                self.coefficients_vect.data = self.monotonic_coefficients_vect.data
                self.monotonic_constraint = False


    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D ou 4D, selon que la couche est
                convolutionnelle ('conv') ou entièrement connectée ('fc').

        Returns:
            output (torch.Tensor)
        """

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        coeff_vect = self.projected_coefficients_vect
        
        x = LinearSpline_Func.apply(x, coeff_vect, grid, zero_knot_indexes, \
                                        self.size, self.even)

        return x

    def derivative(self, x):
        """
        Args:
            input (torch.Tensor):
                2D ou 4D, selon que la couche est
                convolutionnelle ('conv') ou entièrement connectée ('fc').

        Returns:
            output (torch.Tensor)
        """
        assert x.size(1) == self.num_activations, \
            'Forme incorrecte de l’entrée : {} != {}.'.format(input.size(1), self.num_activations)

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        coeff_vect = self.projected_coefficients_vect

        x = LinearSplineDerivative_Func.apply(x, coeff_vect, grid, zero_knot_indexes, \
                                        self.size, self.even)

        return x

    def update_integrated_coeff(self):
        print("-----------------------")
        print("Mise à jour des coefficients du spline pour le coût de régularisation\n"
              "(le modèle d'étape de gradient est entraîné et une intégration est requise pour calculer le coût de régularisation)")
        print("-----------------------")
        coeff = self.projected_coefficients
        
        # extrapoler en supposant des pentes nulles du spline linéaire aux deux extrémités
        coeff_int = torch.cat((coeff[:, 0:1], coeff, coeff[:, -1:]), dim=1)

        # intégrer pour obtenir
        # les coefficients de l'expansion quadratique BSpline correspondante
        self.integrated_coeff = torch.cumsum(coeff_int, dim=1) * self.grid.to(coeff.device)
        
        # imposer zéro en zéro et remodeler
        # ceci est arbitraire, car l'intégration est définie à une constante près
        self.integrated_coeff = (self.integrated_coeff - self.integrated_coeff[:, (self.size + 2)//2].view(-1, 1)).view(-1)

        # stocker une seule fois pour tous les indices des nœuds
        # pas le même que pour le spline linéaire car nous avons 2 nœuds supplémentaires
        self.zero_knot_indexes_integrated = (torch.arange(0, self.num_activations) * (self.size + 2) +
                                             ((self.size + 2) // 2))

    def integrate(self, x):
        if self.integrated_coeff is None:
            self.update_integrated_coeff()

        if x.device != self.integrated_coeff.device:
            self.integrated_coeff = self.integrated_coeff.to(x.device)

        x = Quadratic_Spline_Func.apply(x - self.grid.to(x.device), self.integrated_coeff, 
                                        self.grid.to(x.device), self.zero_knot_indexes_integrated.to(x.device), 
                                        (self.size + 2))

        return x

    def extra_repr(self):
        """ Représentation pour print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]:.3f}, '
             'monotonic_constraint={monotonic_constraint}.')

        return s.format(**self.__dict__)

    def TV2(self, ignore_tails=False, **kwargs):
        """
        Calcule la régularisation de variation totale du second ordre.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        Le terme de régularisation appliqué à cette fonction est :
        TV(2)(deepspline) = ||a||_1.
        """
        if ignore_tails:
            return torch.sum(self.relu_slopes[:, 1:-1].norm(1, dim=1))
        else:
            sl = self.relu_slopes
            return torch.sum(sl.norm(1, dim=1))

    def TV2_vec(self, ignore_tails=False, p=1, **kwargs):
        """
        Calcule la régularisation de variation totale du second ordre.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        Le terme de régularisation appliqué à cette fonction est :
        TV(2)(deepspline) = ||a||_1.
        """
        if ignore_tails:
            return torch.sum(self.relu_slopes[:, 1:-1].norm(p, dim=1))
        else:
            return self.relu_slopes.norm(1, dim=1)

    @property
    def slope_max(self):
        """ Calcul du maximum des pentes du spline """
        coeff = self.projected_coefficients
        slope = (coeff[:, 1:] - coeff[:, :-1]) / self.grid.item()
        slope_max = torch.max(slope, dim=1)[0]
        return slope_max

    # Transformer les splines en fonctions de type clip
    def get_clip_equivalent(self):
        """ Exprimer les splines comme somme de deux ReLU
        Ne concerne que les splines qui ressemblent à la fonction clip """
        coeff_proj = self.projected_coefficients.clone().detach()
        slopes = (coeff_proj[:, 1:] - coeff_proj[:, :-1])
        slopes_change = slopes[:, 1:] - slopes[:, :-1]

        i1 = torch.max(slopes_change, dim=1)
        i2 = torch.min(slopes_change, dim=1)

        i0 = torch.arange(0, coeff_proj.shape[0]).to(coeff_proj.device)

        self.grid_tensor = self.grid_tensor.to(coeff_proj.device)
        x1 = self.grid_tensor[i0, i1[1] + 1].view(1, -1, 1, 1)
        y1 = coeff_proj[i0, i1[1] + 1].view(1, -1, 1, 1)

        x2 = self.grid_tensor[i0, i2[1] + 1].view(1, -1, 1, 1)
        y2 = coeff_proj[i0, i2[1] + 1].view(1, -1, 1, 1)

        slopes = ((y2 - y1) / (x2 - x1)).view(1, -1, 1, 1)

        cl = clip_activation(x1, x2, y1, slopes)

        return cl


class clip_activation(nn.Module):
    """ Classe représentant l'activation en tant que fonction de type clip """
    def __init__(self, x1, x2, y1, slopes):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.slopes = torch.nan_to_num(slopes)
        self.y1 = y1

    def forward(self, x):
        """ Évaluation de la fonction clip """
        return self.slopes * (torch.relu(x - self.x1) - torch.relu(x - self.x2)) + self.y1

    def integrate(self, x):
        """ Intégration de la fonction clip """
        return self.slopes / 2 * ((torch.relu(x - self.x1) ** 2 - torch.relu(x - self.x2) ** 2) + self.y1 * x)
    
    @property
    def slope_max(self):
        """ Calcul du maximum des pentes """
        slope_max = torch.max(self.slopes, dim=1)[0]
        return slope_max



