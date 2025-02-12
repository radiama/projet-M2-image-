import pandas as pd, os, torch
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt, process_image_2, process_image_3, timer
from Proxy_Func import fista, prox_l6, forward_backward
from Denoisers import DRUNet
from Pnp_Algorithms import pnp_pgm, pnp_apgm
from Convex_ridge_regularizer_rev import CRRNN, ConvexRidgeRegularizer
from Utils_rev import tStepDenoiser

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")


    # # # Proxy Algorithms

    # # '''Test Bruitage'''

    sigma_vlow, sigma_low, sigma_mod, sigma_high = 0.02, 0.06, 0.1, 0.2

    Im_butterfly_noised_sig5, Im_butterfly_noised_sig15, Im_butterfly_noised_sig25, Im_butterfly_noised_sig50 = operateur(Im_butterfly).noise(sigma=sigma_vlow), operateur(Im_butterfly).noise(sigma=sigma_low), operateur(Im_butterfly).noise(sigma=sigma_mod), operateur(Im_butterfly).noise(sigma=sigma_high)

    Im_leaves_noised_sig15, Im_leaves_noised_sig25, Im_leaves_noised_sig50 = operateur(Im_leaves).noise(sigma=sigma_low), operateur(Im_leaves).noise(sigma=sigma_mod), operateur(Im_leaves).noise(sigma=sigma_high)
    
    Im_starfish_noised_sig15, Im_starfish_noised_sig25, Im_starfish_noised_sig50 = operateur(Im_starfish).noise(sigma=sigma_low), operateur(Im_starfish).noise(sigma=sigma_mod), operateur(Im_starfish).noise(sigma=sigma_high)

    # Im_butterfly_denoised_sig15, Tb15 = fista(Im_butterfly_noised_sig15, "none", None, 0.01, 0.25, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7, init=False)

    # Im_butterfly_denoised_sig25, Tb25 = fista(Im_butterfly_noised_sig25, "none", None, 0.01, 0.16, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 10}, tol=1e-7, init=False)

    Im_butterfly_denoised_sig50, Tb50 = fista(Im_butterfly_noised_sig50, "none", None, 0.1, 0.65, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 30}, tol=1e-7, init=False)

    # Im_leaves_denoised_sig15, Tl15 = fista(Im_leaves_noised_sig15, "none", None, 0.01, 0.25, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7, init=False)

    # Im_leaves_denoised_sig25, Tl25 = fista(Im_leaves_noised_sig25, "none", None, 0.01, 0.16, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 10}, tol=1e-7, init=False)

    Im_leaves_denoised_sig50, Tl50 = fista(Im_leaves_noised_sig50, "none", None, 0.1, 0.65, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 30}, tol=1e-7, init=False)

    # Im_starfish_denoised_sig15, Ts15 = fista(Im_starfish_noised_sig15, "none", None, 0.01, 0.25, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7, init=False)

    # Im_starfish_denoised_sig25, Ts25 = fista(Im_starfish_noised_sig25, "none", None, 0.01, 0.16, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 10}, tol=1e-7, init=False)

    Im_starfish_denoised_sig50, Ts50 = fista(Im_starfish_noised_sig50, "none", None, 0.1, 0.65, 50, prox=prox_l6, prox_params={"tau": 0.1, "K": 30}, tol=1e-7, init=False)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig15, 1.0))

    
    # # # PNP(Plug and Play Algorithms)

    # # PNP_Débruiteur (Im_butterfly)

    denoiser = DRUNet()

    # Im_butterfly_denoised_sig15_pnp, Tb15_pnp = pnp_pgm(Im_butterfly_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=50, tol=1e-7, init=False)

    # Im_butterfly_denoised_sig25_pnp, Tb25_pnp = pnp_pgm(Im_butterfly_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=50, tol=1e-7, init=False)

    Im_butterfly_denoised_sig50_pnp, Tb50_pnp = pnp_pgm(Im_butterfly_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=50, tol=1e-7, init=False)

    # Im_leaves_denoised_sig15_pnp, Tl15_pnp = pnp_pgm(Im_leaves_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=50, tol=1e-7, init=False)

    # Im_leaves_denoised_sig25_pnp, Tl25_pnp = pnp_pgm(Im_leaves_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=50, tol=1e-7, init=False)

    Im_leaves_denoised_sig50_pnp, Tl50_pnp = pnp_pgm(Im_leaves_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=50, tol=1e-7, init=False)

    # Im_starfish_denoised_sig15_pnp, Ts15_pnp = pnp_pgm(Im_starfish_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=50, tol=1e-7, init=False)

    # Im_starfish_denoised_sig25_pnp, Ts25_pnp = pnp_pgm(Im_starfish_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=50, tol=1e-7, init=False)

    Im_starfish_denoised_sig50_pnp, Ts50_pnp = pnp_pgm(Im_starfish_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=50, tol=1e-7, init=False)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig25_pnp, 1.0))


    # # CRRNN_Débruiteur

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    epoch, denoise = 10, tStepDenoiser
    training_name_5 = f'Sigma_{5}_t_{5}'
    training_name_25 = f'Sigma_{25}_t_{50}'
    checkpoint_dir_5 = os.path.join(parent_dir, f'trained_models/{training_name_5}/checkpoints/checkpoint_{epoch}.pth')
    # checkpoint_dir_5 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_noise_{5}.pth')
    checkpoint_dir_15 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_noise_{15}.pth')
    checkpoint_dir_25 = os.path.join(parent_dir, f'trained_models/{training_name_25}/checkpoints/checkpoint_{epoch}.pth')
    # checkpoint_dir_25 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_noise_{25}.pth')
    checkpoint_dir_50 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_noise_{50}.pth')
    
    # checkpoint_dir = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_{noise}.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modele_0 = ConvexRidgeRegularizer(kernel_size=7, channels=[1, 8, 32], activation_params={"knots_range": 0.1, "n_channels": 32, "n_knots": 21})   
    # model_5=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_5, device=device, load=True)
    # model_5=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_5, device=device, load=True, checkpoint=False)
    # model_15=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_15, device=device, load=True, checkpoint=False)
    # model_25=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_25, device=device, load=True)
    # model_25 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_25, device=device, load=True, checkpoint=False)
    model_50 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_50, device=device, load=True, checkpoint=False)

    # lmbd, mu = model_15.lmbd_transformed, model_15.mu_transformed
    # print(f"Lambda: {lmbd:.2f}")
    # print(f"Mu: {mu:.2f}")

    with torch.no_grad():
        # Im_butterfly_denoised_sig15_crrnn, Tb15_crrnn = process_image_3(Im_butterfly_noised_sig15, operator=denoise, model=model_5, t_steps=50, operator_type ="none", operator_params=None, auto_params=True, lmbd=4.e-0, mu=8e-1, step_size=1e-1, init=False)
        # Im_butterfly_denoised_sig25_crrnn, Tb25_crrnn = process_image_3(Im_butterfly_noised_sig25, operator=denoise, model=model_25, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=12.e-0, mu=1e-0, step_size=1e-1)
        Im_butterfly_denoised_sig50_crrnn, Tb50_crrnn = process_image_3(Im_butterfly_noised_sig50, operator=denoise, model=model_50, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=1.5e-0, mu=2.5e-0, step_size=9e-2)
        # Im_leaves_denoised_sig15_crrnn, Tl15_crrnn = process_image_3(Im_leaves_noised_sig15, operator=denoise, model=model_5, t_steps=50, operator_type ="none", operator_params=None, auto_params=True, lmbd=15.e-0, mu=1e-0, step_size=8e-2)
        # Im_leaves_denoised_sig25_crrnn, Tl25_crrnn = process_image_3(Im_leaves_noised_sig25, operator=denoise, model=model_25, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=12.e-0, mu=1e-0, step_size=1e-1)
        Im_leaves_denoised_sig50_crrnn, Tl50_crrnn = process_image_3(Im_leaves_noised_sig50, operator=denoise, model=model_50, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=1.5e-0, mu=2.5e-0, step_size=9e-2)
        # Im_starfish_denoised_sig15_crrnn, Ts15_crrnn = process_image_3(Im_starfish_noised_sig15, operator=denoise, model=model_5, t_steps=50, operator_type ="none", operator_params=None, auto_params=True, lmbd=15.e-0, mu=1e-0, step_size=8e-2)
        # Im_starfish_denoised_sig25_crrnn, Ts25_crrnn = process_image_3(Im_starfish_noised_sig25, operator=denoise, model=model_25, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=12.e-0, mu=1e-0, step_size=1e-1)
        Im_starfish_denoised_sig50_crrnn, Ts50_crrnn = process_image_3(Im_starfish_noised_sig50, operator=denoise, model=model_50, t_steps=50, operator_type ="none", operator_params=None, auto_params=False, lmbd=1.5e-0, mu=2.5e-0, step_size=9e-2)


    # # # '''Recherche paramètres optimaux (Proxy and PNP)'''
    
    # # Proxy

    # param_ranges = { "K": [5, 10, 15, 20], "lambd": [0.01, 0.1, 0.5], "tau": [0.01, 0.1, 0.5]}

    # prox_params_ranges = {"tau": [0.01, 0.1, 0.5],  "K": [5, 10, 15, 20]}

    # func_params = {"u": Im_butterfly_noised_sig25, "operator_type": "none", "operator_params": {}, "prox": prox_l6, "tol": 1e-7, "init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=fista, u_truth=Im_butterfly, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)


    # # PNP

    # param_ranges_pnp = { "K": [5, 10, 15], "tau": [0.01, 0.1, 1], "sigma": [sigma_low, sigma_mod, sigma_high]}
    
    # func_params_pnp = {"u": Im_butterfly_noised_sig50, "operator_type": "none", "operator_params": {}, "denoiser": denoiser, "tol": 1e-7,"init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=pnp_pgm, u_truth=Im_butterfly, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)


    # # Résultats

    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # Save images 
    
    # Images Denoising

    reference_image_list = [Im_butterfly, Im_leaves, Im_starfish]

    # names_denoised_sig15 = [["Low Noise ($\sigma=0.06$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                         ["Low Noise ($\sigma=0.06$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                         ["Low Noise ($\sigma=0.06$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    # images_denoised_sig15 =[[Im_butterfly_noised_sig15, Im_butterfly_denoised_sig15, Im_butterfly_denoised_sig15_pnp, Im_butterfly_denoised_sig15_crrnn],
    #                         [Im_leaves_noised_sig15, Im_leaves_denoised_sig15, Im_leaves_denoised_sig15_pnp, Im_leaves_denoised_sig15_crrnn],
    #                         [Im_starfish_noised_sig15, Im_starfish_denoised_sig15, Im_starfish_denoised_sig15_pnp, Im_starfish_denoised_sig15_crrnn]]

    # denoise_trajectories_sig_15 = [[Tb15, Tb15_pnp, Tb15_crrnn], [Tl15, Tl15_pnp, Tl15_crrnn], [Ts15, Ts15_pnp, Ts15_crrnn]]

    # names_denoised_sig25 = [["Moderate Noise ($\sigma=0.1$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                         ["Moderate Noise ($\sigma=0.1$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                         ["Moderate Noise ($\sigma=0.1$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    # images_denoised_sig25 =[[Im_butterfly_noised_sig25, Im_butterfly_denoised_sig25, Im_butterfly_denoised_sig25_pnp, Im_butterfly_denoised_sig25_crrnn],
    #                         [Im_leaves_noised_sig25, Im_leaves_denoised_sig25, Im_leaves_denoised_sig25_pnp, Im_leaves_denoised_sig25_crrnn],
    #                         [Im_starfish_noised_sig25, Im_starfish_denoised_sig25, Im_starfish_denoised_sig25_pnp, Im_starfish_denoised_sig25_crrnn]]

    # denoise_trajectories_sig_25 = [[Tb25, Tb25_pnp, Tb25_crrnn], [Tl25, Tl25_pnp, Tl25_crrnn], [Ts25, Ts25_pnp, Ts25_crrnn]]

    names_denoised_sig50 = [["High Noise ($\sigma=0.2$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
                            ["High Noise ($\sigma=0.2$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
                            ["High Noise ($\sigma=0.2$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    images_denoised_sig50 =[[Im_butterfly_noised_sig50, Im_butterfly_denoised_sig50, Im_butterfly_denoised_sig50_pnp, Im_butterfly_denoised_sig50_crrnn],
                            [Im_leaves_noised_sig50, Im_leaves_denoised_sig50, Im_leaves_denoised_sig50_pnp, Im_leaves_denoised_sig50_crrnn],
                            [Im_starfish_noised_sig50, Im_starfish_denoised_sig50, Im_starfish_denoised_sig50_pnp, Im_starfish_denoised_sig50_crrnn]]

    denoise_trajectories_sig_50 = [[Tb50, Tb50_pnp, Tb50_crrnn], [Tl50, Tl50_pnp, Tl50_crrnn], [Ts50, Ts50_pnp, Ts50_crrnn]]


    # save_path("results_2", "Imgs_denoised_sig15", names_denoised_sig15, images_denoised_sig15, reference_image_list= reference_image_list, 
    #           psnr=True, trajectories_list=denoise_trajectories_sig_15)
    
    # save_path("results_2", "Imgs_denoised_sig25", names_denoised_sig25, images_denoised_sig25, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=denoise_trajectories_sig_25)

    save_path("results_2", "Imgs_denoised_sig50", names_denoised_sig50, images_denoised_sig50, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=denoise_trajectories_sig_50)


    # # # Timer

    # # Initialisation des listes pour stocker les temps d'exécution
    # methods = ["TV", "DRUNet", "CRRNN"]
    # sigma_values = [15, 25, 50]

    # # Création d'un dictionnaire pour stocker les résultats
    # # Timer_map = {sigma: [] for sigma in sigma_values}
    # Timer_map = {method:[] for method in methods}

    # # Boucle sur les différents niveaux de bruit sigma ou méthode
        
    # # Méthode 1 : TV
    # time_tv_15 = timer(forward_backward, Im_butterfly_noised_sig15, "none", None, 0.01, 0.25, 50, 
    #                 prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7, init=False)
    
    # time_tv_25 = timer(forward_backward, Im_butterfly_noised_sig25, "none", None, 0.01, 0.16, 50, 
    #                    prox=prox_l6, prox_params={"tau": 0.1, "K": 10}, tol=1e-7, init=False)

    # time_tv_50 = timer(forward_backward, Im_butterfly_noised_sig50, "none", None, 0.1, 0.65, 50, 
    #                     prox=prox_l6, prox_params={"tau": 0.1, "K": 30}, tol=1e-7, init=False)
    
    # # Timer_map[15].append(time_tv)
    # Timer_map["TV"].append(time_tv_15)
    # Timer_map["TV"].append(time_tv_25)
    # Timer_map["TV"].append(time_tv_50)

    # # Méthode 2 : DRUNet
    # time_drunet_15 = timer(pnp_pgm, Im_butterfly_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, 
    #                     K=50, tol=1e-7, init=False)
    
    # time_drunet_25 =  timer(pnp_pgm, Im_butterfly_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, 
    #                         K=50, tol=1e-7, init=False)

    # time_drunet_50 = timer(pnp_pgm, Im_butterfly_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, 
    #                        K=50, tol=1e-7, init=False)


    # # Timer_map[15].append(time_drunet)
    # Timer_map["DRUNet"].append(time_drunet_15)
    # Timer_map["DRUNet"].append(time_drunet_25)
    # Timer_map["DRUNet"].append(time_drunet_50)

    # # Méthode 3 : CRRNN (avec torch.no_grad() pour éviter le calcul des gradients)

    # model_5=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_5, device=device, load=True)

    # with torch.no_grad():
    #     time_crrnn_15 = timer(process_image_3, Im_butterfly_noised_sig15, operator=denoise, model=model_5, 
    #                     t_steps=50, operator_type="none", operator_params=None, 
    #                     auto_params=True, lmbd=15.e-0, mu=1e-0, step_size=8e-2, init=False)
        
    # model_25=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_25, device=device, load=True)

    # with torch.no_grad():
    #     time_crrnn_25 = timer(process_image_3, Im_butterfly_noised_sig25, operator=denoise, model=model_25, 
    #                                     t_steps=50, operator_type ="none", operator_params=None, 
    #                                     auto_params=False, lmbd=12.e-0, mu=1e-0, step_size=1e-1)
        
    # model_50 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_50, device=device, load=True, checkpoint=False)
    
    # with torch.no_grad():
    #     time_crrnn_50 = timer(process_image_3, Im_butterfly_noised_sig50, operator=denoise, model=model_50, 
    #                                     t_steps=50, operator_type ="none", operator_params=None, 
    #                                     auto_params=False, lmbd=1.5e-0, mu=2.5e-0, step_size=9e-2)
 
    # # Timer_map[15].append(time_crrnn)
    # Timer_map["CRRNN"].append(time_crrnn_15)
    # Timer_map["CRRNN"].append(time_crrnn_25)
    # Timer_map["CRRNN"].append(time_crrnn_50)

    # # Création d'un DataFrame Pandas
    # # Timer_map_df = pd.DataFrame.from_dict(Timer_map, orient="index", columns=methods)
    # Timer_map_df = pd.DataFrame.from_dict(Timer_map, orient="index", columns=[f"Sigma_{sigma}" for sigma in sigma_values])

    # # Affichage du tableau formaté
    # print("Temps d'exécution pour chaque méthode et niveau de bruit (en secondes) :")
    
    # print(Timer_map_df)

    # import numpy as np
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(18, 6), dpi=200)
    # plt.subplot(131)
    # plt.imshow(Im_butterfly)
    # plt.subplot(132)
    # plt.imshow(Im_butterfly_noised_sig15)
    # plt.subplot(133)
    # plt.imshow(Im_butterfly_denoised_sig25_crrnn)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig25_crrnn, 1.0))

    # print([PSNR(Im_butterfly, Im_butterfly_denoised_sig50_pnp),
    #        PSNR(Im_leaves, Im_leaves_denoised_sig50_pnp),
    #         PSNR(Im_starfish, Im_starfish_denoised_sig50_pnp)])

if __name__ == "__main__" :

    main()