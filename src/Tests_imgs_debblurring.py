import torch, os, pandas as pd
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt, process_image_2, process_image_3

from Proxy_Func import fista, prox_l6
from Denoisers import DRUNet
from Pnp_Algorithms import pnp_pgm, pnp_apgm
import pandas as pd
from Convex_ridge_regularizer_rev import CRRNN, ConvexRidgeRegularizer
from Utils_rev import tStepDenoiser

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")


    # # # Proxy Algorithms


    # # '''Test Défloutage '''

    sigma_low_2, sigma_mod_2, sigma_high_2 = (1, 1), (2, 2), (3, 3)

    Im_butterfly_blurred_low, G_butterfly_low = operateur(Im_butterfly).blur(sigma = sigma_low_2, angle = 0)
    
    Im_butterfly_blurred_mod, G_butterfly_mod = operateur(Im_butterfly).blur(sigma = sigma_mod_2, angle = 0)

    Im_butterfly_blurred_high, G_butterfly_high = operateur(Im_butterfly).blur(sigma = sigma_high_2, angle = 0)
    
    Im_leaves_blurred_low, G_leaves_low = operateur(Im_leaves).blur(sigma = sigma_low_2, angle = 0)
    
    Im_leaves_blurred_mod, G_leaves_mod = operateur(Im_leaves).blur(sigma = sigma_mod_2, angle = 0)

    Im_leaves_blurred_high, G_leaves_high = operateur(Im_leaves).blur(sigma = sigma_high_2, angle = 0)

    Im_starfish_blurred_low, G_starfish_low = operateur(Im_starfish).blur(sigma = sigma_low_2, angle = 0)
    
    Im_starfish_blurred_mod, G_starfish_mod = operateur(Im_starfish).blur(sigma = sigma_mod_2, angle = 0)

    Im_starfish_blurred_high, G_starfish_high = operateur(Im_starfish).blur(sigma = sigma_high_2, angle = 0)


    # Im_butterfly_deblurred_low, Tb_low = fista(Im_butterfly_blurred_low, "convolution", {"G": G_butterfly_low}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 20}, tol=1e-7, init=False)

    # Im_butterfly_deblurred_mod, Tb_mod = fista(Im_butterfly_blurred_mod, "convolution", {"G": G_butterfly_mod}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 25}, tol=1e-7, init=False)

    Im_butterfly_deblurred_high, Tb_high = fista(Im_butterfly_blurred_high, "convolution", {"G": G_butterfly_high}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-2, "K": 5}, tol=1e-7, init=False)
    
    # Im_leaves_deblurred_low, Tl_low = fista(Im_leaves_blurred_low, "convolution", {"G": G_leaves_low}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 20}, tol=1e-7, init=False)

    # Im_leaves_deblurred_mod, Tl_mod = fista(Im_leaves_blurred_mod, "convolution", {"G": G_leaves_mod}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 25}, tol=1e-7, init=False)

    Im_leaves_deblurred_high, Tl_high = fista(Im_leaves_blurred_high, "convolution", {"G": G_leaves_high}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-2, "K": 5}, tol=1e-7, init=False)

    # Im_starfish_deblurred_low, Ts_low = fista(Im_starfish_blurred_low, "convolution", {"G": G_starfish_low}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 20}, tol=1e-7, init=False)

    # Im_starfish_deblurred_mod, Ts_mod = fista(Im_starfish_blurred_mod, "convolution", {"G": G_starfish_mod}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 25}, tol=1e-7, init=False)

    Im_starfish_deblurred_high, Ts_high = fista(Im_starfish_blurred_high, "convolution", {"G": G_starfish_high}, 1e-4, 1.3, 200, prox=prox_l6, prox_params={"tau": 1e-4, "K": 25}, tol=1e-7, init=False)

    # print(PSNR(Im_leaves, Im_leaves_deblurred_high, 1.0))



    # # # PNP(Plug and Play Algorithms)

    # # PNP_Défloutage

    denoiser = DRUNet()

    # Im_butterfly_deblurred_low_pnp, Tb_low_pnp = pnp_pgm(Im_butterfly_blurred_low, "convolution", {"G": G_butterfly_low}, 1.9, denoiser, sigma=8e-3, K=200, tol=1e-7, init=False)

    # Im_butterfly_deblurred_mod_pnp, Tb_mod_pnp = pnp_pgm(Im_butterfly_blurred_mod, "convolution", {"G": G_butterfly_mod}, 1.9, denoiser, sigma=3e-2, K=200, tol=1e-7, init=False)

    Im_butterfly_deblurred_high_pnp, Tb_high_pnp = pnp_pgm(Im_butterfly_blurred_high, "convolution", {"G": G_butterfly_high}, 1.9, denoiser, sigma=5e-2, K=200, tol=1e-7, init=False)

    # Im_leaves_deblurred_low_pnp, Tl_low_pnp = pnp_pgm(Im_leaves_blurred_low, "convolution", {"G": G_leaves_low}, 1.9, denoiser, sigma=1e-2, K=200, tol=1e-7, init=False)

    # Im_leaves_deblurred_mod_pnp, Tl_mod_pnp = pnp_pgm(Im_leaves_blurred_mod, "convolution", {"G": G_leaves_mod}, 1.9, denoiser, sigma=5e-2, K=200, tol=1e-7, init=False)

    Im_leaves_deblurred_high_pnp, Tl_high_pnp = pnp_pgm(Im_leaves_blurred_high, "convolution", {"G": G_leaves_high}, 1.9, denoiser, sigma=3e-2, K=200, tol=1e-7, init=False)
    
    # Im_starfish_deblurred_low_pnp, Ts_low_pnp = pnp_pgm(Im_starfish_blurred_low, "convolution", {"G": G_starfish_low}, 1.9, denoiser, sigma=1e-2, K=200, tol=1e-7, init=False)

    # Im_starfish_deblurred_mod_pnp, Ts_mod_pnp = pnp_pgm(Im_starfish_blurred_mod, "convolution", {"G": G_starfish_mod}, 1.9, denoiser, sigma=5e-2, K=200, tol=1e-7, init=False)

    Im_starfish_deblurred_high_pnp, Ts_high_pnp = pnp_pgm(Im_starfish_blurred_high, "convolution", {"G": G_starfish_high}, 1.9, denoiser, sigma=3e-2, K=200, tol=1e-7, init=False)

    # print(PSNR(Im_starfish, Im_starfish_deblurred_high_pnp, 1.0))

   # # CRRNN_Débruiteur

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    epoch, denoise = 10, tStepDenoiser
    training_name_5 = f'Sigma_{5}_t_{5}'
    training_name_25 = f'Sigma_{25}_t_{30}'
    checkpoint_dir_5 = os.path.join(parent_dir, f'trained_models/{training_name_5}/checkpoints/checkpoint_{epoch}.pth')
    checkpoint_dir_25 = os.path.join(parent_dir, f'trained_models/{training_name_25}/checkpoints/checkpoint_{epoch}.pth')
    
    # checkpoint_dir = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_{noise}.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modele_5 = ConvexRidgeRegularizer(kernel_size=7, channels=[1, 8, 32], activation_params={"knots_range": 0.1, "n_channels": 32, "n_knots": 21})
    modele_25 = ConvexRidgeRegularizer(kernel_size=7, channels=[1, 8, 32], activation_params={"knots_range": 0.1, "n_channels": 32, "n_knots": 21})

    model_5=CRRNN(model=modele_5, name_model_pre=checkpoint_dir_5, device=device, load=True)
    model_25=CRRNN(model=modele_25, name_model_pre=checkpoint_dir_25, device=device, load=True)

    with torch.no_grad():

        # Im_butterfly_deblurred_low_crrnn, Tb_low_crrnn = process_image_3(Im_butterfly_blurred_low, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_butterfly_low}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        # Im_butterfly_deblurred_medium_crrnn, Tb_mod_crrnn = process_image_3(Im_butterfly_blurred_mod, operator=denoise, model=model_5, t_steps=50, operator_type ="convolution", operator_params={"G": G_butterfly_mod}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        Im_butterfly_deblurred_high_crrnn, Tb_high_crrnn = process_image_3(Im_butterfly_blurred_high, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_butterfly_high}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        # Im_leaves_deblurred_low_crrnn, Tl_low_crrnn = process_image_3(Im_leaves_blurred_low, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_leaves_low}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        # Im_leaves_deblurred_medium_crrnn, Tl_mod_crrnn = process_image_3(Im_leaves_blurred_mod, operator=denoise, model=model_5, t_steps=50, operator_type ="convolution", operator_params={"G": G_leaves_mod}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        Im_leaves_deblurred_high_crrnn, Tl_high_crrnn = process_image_3(Im_leaves_blurred_high, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_leaves_high}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        # Im_starfish_deblurred_low_crrnn, Ts_low_crrnn = process_image_3(Im_starfish_blurred_low, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_starfish_low}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        # Im_starfish_deblurred_medium_crrnn, Ts_mod_crrnn = process_image_3(Im_starfish_blurred_mod, operator=denoise, model=model_5, t_steps=50, operator_type ="convolution", operator_params={"G": G_starfish_mod}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)
        Im_starfish_deblurred_high_crrnn, Ts_high_crrnn = process_image_3(Im_starfish_blurred_high, operator=denoise, model=model_5, t_steps=200, operator_type ="convolution", operator_params={"G": G_starfish_high}, auto_params=False, lmbd=2e-1, mu=5e-0, step_size=1.9, init=False)

    
    # # # '''Recherche paramètres optimaux (Proxy and PNP)'''
    
    # Proxy

    # param_ranges = { "K": [5, 10], "lambd": [0.0001, 0.01, 0.1], "tau": [0.01, 0.1, 1.3]}

    # prox_params_ranges = {"tau": [0.0001, 0.01, 0.1],  "K": [5, 10]}

    # func_params = {"u": Im_leaves_blurred_high, "operator_type": "convolution", "operator_params": {"G": G_leaves_high}, "prox": prox_l6, "tol": 1e-7, "init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=fista, u_truth=Im_leaves, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)


    # # PNP

    # param_ranges_pnp = { "K": [10, 20, 30], "tau": [0.25, 1, 2], "sigma": [1e-2, 3e-2, 5e-2]}

    # func_params_pnp = {"u": Im_leaves_blurred_high, "operator_type": "convolution", "operator_params": {"G": G_leaves_high}, "denoiser": denoiser, "tol": 1e-7, "init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=pnp_pgm, u_truth=Im_leaves, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)

    # # Résultats

    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # Save images 

    # Images Deblurring

    reference_image_list = [Im_butterfly, Im_leaves, Im_starfish]

    # names_deblurred_sig_1 = [["Low Blur ($\sigma=1$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                             ["Low Blur ($\sigma=1$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                             ["Low Blur ($\sigma=1$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    # images_deblurred_sig_1 =[[Im_butterfly_blurred_low, Im_butterfly_deblurred_low, Im_butterfly_deblurred_low_pnp, Im_butterfly_deblurred_low_crrnn], 
    #                             [Im_leaves_blurred_low, Im_leaves_deblurred_low, Im_leaves_deblurred_low_pnp, Im_leaves_deblurred_low_crrnn], 
    #                             [Im_starfish_blurred_low, Im_starfish_deblurred_low, Im_starfish_deblurred_low_pnp, Im_starfish_deblurred_low_crrnn]]

    # deblur_trajectories_sig_1 = [[Tb_low, Tb_low_pnp, Tb_low_crrnn], [Tl_low, Tl_low_pnp, Tl_low_crrnn], [Ts_low, Ts_low_pnp, Ts_low_crrnn]]


    # names_deblurred_sig_2 = [["Moderate Blur B ($\sigma=2$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
    #                             ["Moderate Blur L ($\sigma=2$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
    #                             ["Moderate Blur S ($\sigma=2$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]    

    # images_deblurred_sig_2 =[[Im_butterfly_blurred_mod, Im_butterfly_deblurred_mod, Im_butterfly_deblurred_mod_pnp], 
    #                             [Im_leaves_blurred_mod, Im_leaves_deblurred_mod, Im_leaves_deblurred_mod_pnp], 
    #                             [Im_starfish_blurred_mod, Im_starfish_deblurred_mod, Im_starfish_deblurred_mod_pnp]]

    # deblur_trajectories_sig_2 = [[Tb_mod, Tb_mod_pnp], [Tl_mod, Tl_mod_pnp], [Ts_mod, Ts_mod_pnp]]


    names_deblurred_sig_3 = [["High Blur B ($\sigma=3$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
                                ["High Blur L ($\sigma=3$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
                                ["High Blur S ($\sigma=3$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]    

    images_deblurred_sig_3 =[[Im_butterfly_blurred_high, Im_butterfly_deblurred_high, Im_butterfly_deblurred_high_pnp, Im_butterfly_deblurred_high_crrnn], 
                                [Im_leaves_blurred_high, Im_leaves_deblurred_high, Im_leaves_deblurred_high_pnp, Im_leaves_deblurred_high_crrnn], 
                                [Im_starfish_blurred_high, Im_starfish_deblurred_high, Im_starfish_deblurred_high_pnp, Im_starfish_deblurred_high_crrnn]]

    deblur_trajectories_sig_3 = [[Tb_high, Tb_high_pnp, Tb_high_crrnn], [Tl_high, Tl_high_pnp, Tl_high_crrnn], [Ts_high, Ts_high_pnp, Ts_high_crrnn]]

    # save_path("results_2", "Imgs_deblurred_sig_1", names_deblurred_sig_1, images_deblurred_sig_1, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=deblur_trajectories_sig_1)
    
    # save_path("results", "Imgs_deblurred_sig_2", names_deblurred_sig_2, images_deblurred_sig_2, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=deblur_trajectories_sig_2)

    save_path("results_2", "Imgs_deblurred_sig_3", names_deblurred_sig_3, images_deblurred_sig_3, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=deblur_trajectories_sig_3)


    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(18, 6), dpi=200)
    # plt.subplot(131)
    # plt.imshow(Im_butterfly_blurred_low)
    # plt.subplot(132)
    # plt.imshow(G_butterfly_low)
    # plt.subplot(133)
    # plt.imshow(Im_butterfly_deblurred_low_crrnn)

    # print(PSNR(Im_butterfly, Im_butterfly_deblurred_low_crrnn, 1.0))

    # print([PSNR(Im_butterfly, Im_butterfly_deblurred_high_pnp),
    #        PSNR(Im_leaves, Im_leaves_deblurred_high_pnp),
    #         PSNR(Im_starfish, Im_starfish_deblurred_high_pnp)])


if __name__ == "__main__" :

    main()