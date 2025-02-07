import os
import torch, matplotlib.pyplot as plt, numpy as np
from tqdm import tqdm
from Utils_rev import tStepDenoiser
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from Convex_ridge_regularizer_rev import ConvexRidgeRegularizer, CRRNN
from Begin_Func import OperatorDataset, EarlyStopping, process_image_2, operateur, load_img, PSNR
from torchmetrics.functional import peak_signal_noise_ratio as psnr


def main():

    # Chemin du dossier contenant les images propres
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    images_dir = os.path.join(parent_dir, 'data/BSD/train')
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    print("I have found {} images in the directory".format(len(image_files)))

    # Transformation des images
    H, W = 40, 40 # Taille des images
    transform = transforms.Compose([
    transforms.Resize((H, W)),               # Redimensionnement
    # transforms.RandomHorizontalFlip(p=0.5),     # Flip horizontal aléatoire
    # transforms.RandomRotation(degrees=90),      # Rotation aléatoire
    # transforms.RandomVerticalFlip(p=0.5),        # Flip vertical
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Luminosité, contraste, saturation
    transforms.ToTensor(),                      # Convertir en tenseur
    ])

    # Chargement des images

    noise=25 # Niveau de bruit
    blur=(3,3) # Niveau de flou

    simple_mask = torch.ones(1, 1, H, W)

    simple_mask[:, :, 0::int(np.sqrt(H)), :], simple_mask[:, :, :, 0::int(np.sqrt(W))] = 0, 0

    simple_mask = simple_mask.to(torch.bool)

    train_dataset = OperatorDataset(images_dir=images_dir, image_files=image_files,
                                    operator="noising", transform=transform, noise_level=round(noise/255, 1 if noise >= 25 else 2),
                                    blur_level=blur, mask=simple_mask, random=False) # Créer le dataset noisy (noising), blurred (blurring), masked (painting)

    num_train = int(len(train_dataset) * 0.95)

    train_0, test = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    num_train_0 = int(len(train_0) * 0.9)

    train, valid = \
        random_split(train_0, [num_train_0, len(train_0) - num_train_0])

    # DataLoader (Batch size = 1) # Batch size = 128
    batch_size = 64
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    data_iter = iter(train_loader)
    images_noised, images_truth = next(data_iter)
    # images_noised, images_truth, _ = next(data_iter)

    # Afficher 2*5 images transformées
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(5):
        axes[0, i].imshow(images_noised[i].permute(1, 2, 0), cmap="gray")  # Permuter les dimensions pour imshow
        axes[0, i].set_title("Noisy")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(images_truth[i].permute(1, 2, 0), cmap="gray")
        axes[1, i].set_title("Ground Truth")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show();

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = ConvexRidgeRegularizer(kernel_size=7, channels=[1, 8, 32], activation_params={"knots_range": 0.1, "n_channels": 32, "n_knots": 21})
    # trained_model_path = os.path.join(parent_dir, f"trained_models/IOD_Training/crr_nn_best_model_noise_{noise}.pth")
    # model.load_state_dict(torch.load(trained_model_path, map_location=device)) # Charger les poids depuis le fichier .pth

    # Définissons le débruiteur
    denoise = tStepDenoiser

    # Entraînement du modèle

    # Définir la perte
    criterion = nn.L1Loss(reduction='mean')
    # criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.SmoothL1Loss(reduction='mean')

    # Définir les optimiseurs
    optimizers = []
    optimizer = torch.optim.Adam

    # Optimisation des couches convolutives
    params_conv = model.conv_layer.parameters()
    optimizer_conv = optimizer(params_conv, lr=1e-3)
    optimizers.append(optimizer_conv)

    # Optimisation des fonctions d'activation
    if model.use_splines:
        params_activations = [model.activation.coefficients_vect]
        optimizer_activations = optimizer(params_activations, lr=5e-05)
        optimizers.append(optimizer_activations)

    # Optimisation des poids de régularisation
    optimizer_lmbd = optimizer([model.lmbd], lr=5e-2)
    optimizers.append(optimizer_lmbd)

    # Optimisation des facteurs d'échelle
    optimizer_mu = optimizer([model.mu], lr=5e-2)
    optimizers.append(optimizer_mu)

    # Définir le scheduler et l'arrêt précoce
    schedulers = [StepLR(optim, step_size=10, gamma=0.75) for optim in optimizers]

    # Optimizers_names
    lr_name = ["lr_conv", "lr_params_activ", "lr_lmd", "lr_mu"] if model.use_splines else ["lr_conv", "lr_lmd", "lr_mu"]

    # Early_stopping
    early_stopping = EarlyStopping(patience=10, noise_level=noise, blur_level=blur, operator="denoising")

    num_epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for noisy_images, truth_images in tqdm(train_loader, desc="CRRNN Training"):
        
        # for noisy_images, truth_images, operator in tqdm(train_loader, desc="CRRNN Training"):

            noisy_images, truth_images = noisy_images.to(device, dtype=torch.float32), truth_images.to(device, dtype=torch.float32)

            # noisy_images, truth_images, operator = noisy_images.to(device, dtype=torch.float32), truth_images.to(device, dtype=torch.float32), operator.to(device, dtype=torch.float32)
            
            # Réinitialisation des gradients
            for optimizer in optimizers:
                optimizer.zero_grad()

            noisy_images.requires_grad_(True) # Activer le suivi des gradients
            truth_images.requires_grad_(False) # Désactiver le suivi des gradients
            
            # Forward pass
            outputs = denoise(noisy_images, model, t_steps=100, operator_type ="none", operator_params=None, train=True, auto_params=True, traj=False, verbose=False)
            # outputs = denoise(noisy_images, model, t_steps=100, operator_type ="mask", operator_params={"Mask": operator}, train=True, auto_params=True, traj=False, verbose=False)
            # outputs = denoise(noisy_images, model, t_steps=100, operator_type ="convolution", operator_params={"G": operator}, train=True, auto_params=True, traj=False, verbose=False)

            # Calcul de la perte
            loss = criterion(outputs, truth_images)

            # Backward pass
            loss.backward()

            # Mise à jour des paramètres
            for optimizer in optimizers:
                optimizer.step()
            
            running_loss += loss.item()

        # Affichage des statistiques
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")    
        print(f"Training Loss: {running_loss/len(train_loader):.4f}")
        train_losses.append(running_loss/len(train_loader))

        # Validation du modèle
        model.eval()
        with torch.no_grad():
            val_loss, psnr_val = 0.0, 0.0
            for noisy_images, truth_images in valid_loader:
            # for noisy_images, truth_images, operator in valid_loader:
                noisy_images, truth_images = noisy_images.to(device, dtype=torch.float32), truth_images.to(device, dtype=torch.float32)
                # noisy_images, truth_images, operator = noisy_images.to(device, dtype=torch.float32), truth_images.to(device, dtype=torch.float32), operator.to(device, dtype=torch.float32)
                outputs = denoise(noisy_images, model, t_steps=100, operator_type ="none", operator_params=None, train=True, auto_params=True, traj=False, verbose=False)
                # outputs = denoise(noisy_images, model, t_steps=100, operator_type ="mask", operator_params={"Mask": operator}, train=True, auto_params=True, traj=False, verbose=False)
                # outputs = denoise(noisy_images, model, t_steps=100, operator_type ="convolution", operator_params={"G": operator}, train=True, auto_params=True, traj=False, verbose=False)
                loss = criterion(outputs, truth_images)
                psnr_val += psnr(outputs, truth_images, data_range=1.0).item()
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(valid_loader):.4f} | PSNR: {psnr_val / len(valid_loader):.2f}")
        print("=============================")
        val_losses.append(val_loss)

        # Vérification de l'arrêt précoce
        if early_stopping(val_loss, model, epoch):
            print("Early stopping triggered. Training stopped.")
            break
        
        # Mise à jour du scheduler
        for sc in schedulers:
            sc.step()

        for name, optimizer in zip(lr_name, optimizers) :
            print(f"\nLearning rate {name}: {optimizer.param_groups[0]['lr']:4f}")
        print("\n")

    # # Test du modèle
    sigma_training, epoch = 5, 10
    t = 50
    training_name = f'Sigma_{sigma_training}_t_{t}'
    # checkpoint_dir = os.path.join(parent_dir, f'trained_models/{training_name}/checkpoints/checkpoint_{epoch}.pth')
    checkpoint_dir = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_noise_{noise}.pth')

    modele=CRRNN(model=model, name_model_pre=checkpoint_dir, device=device, load=False, checkpoint=False)
    
    with torch.no_grad():
        noisy_images, truth_images = next(iter(test_loader))
        # noisy_images, truth_images, operator = next(iter(test_loader))
        lmbd = modele.lmbd_transformed
        print(f"Lambda: {lmbd:.2f}")
        mu = modele.mu_transformed
        print(f"Mu: {mu:.2f}")
        i=0
        noisy_image = noisy_images[i].permute(1, 2, 0).cpu().numpy()
        truth_image = truth_images[i].permute(1, 2, 0).cpu().numpy()
        # denoised_image = denoise(noisy_images, CRRNN(), t_steps=50, operator_type ="none", operator_params=None, train=True, auto_params=False, lmbd=10.5e-0, mu=3e-0, step_size=1e-1)
        denoised_image = denoise(noisy_images[i], modele, t_steps=100, operator_type ="none", operator_params=None, train=True, auto_params=True, lmbd=10.5e-0, mu=3e-0, step_size=1e-1, traj=False).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # denoised_image = denoise(noisy_images[i], modele, t_steps=100, operator_type ="mask", operator_params={"Mask": operator[i]}, train=True, auto_params=True, traj=False, verbose=False).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # denoised_image = denoise(noisy_images[i], modele, t_steps=100, operator_type ="convolution", operator_params={"G": operator[i]}, train=True, auto_params=True, traj=False, verbose=False).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # denoised_image = CRRNN()(noisy_images[i]).squeeze(0).permute(1, 2, 0).cpu().numpy()

    psnr_denoised = PSNR(denoised_image, truth_image, max_intensity=1.0)

    print(f"PSNR of denoised image: {psnr_denoised:.2f}")

    # Afficher les images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Image bruitée")
    plt.imshow(noisy_image, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Image débruitée")
    plt.imshow(denoised_image, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Image propre")
    plt.imshow(truth_image, cmap="gray")
    plt.show()


if __name__=="__main__":

    main()

