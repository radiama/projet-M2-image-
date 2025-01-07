#import torch
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt, process_image_2
from Proxy_Func import fista, prox_l6
from Denoisers import DRUNet
from Pnp_Algorithms import pnp_pgm, pnp_apgm
import pandas as pd

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")


    # # # Proxy Algorithms

    # # '''Test Bruitage'''

    sigma_low, sigma_mod, sigma_high = 0.06, 0.1, 0.2

    Im_butterfly_noised_sig15, Im_butterfly_noised_sig25, Im_butterfly_noised_sig50 = operateur(Im_butterfly).noise(sigma=sigma_low), operateur(Im_butterfly).noise(sigma=sigma_mod), operateur(Im_butterfly).noise(sigma=sigma_high)

    Im_leaves_noised_sig15, Im_leaves_noised_sig25, Im_leaves_noised_sig50 = operateur(Im_leaves).noise(sigma=sigma_low), operateur(Im_leaves).noise(sigma=sigma_mod), operateur(Im_leaves).noise(sigma=sigma_high)
    
    Im_starfish_noised_sig15, Im_starfish_noised_sig25, Im_starfish_noised_sig50 = operateur(Im_starfish).noise(sigma=sigma_low), operateur(Im_starfish).noise(sigma=sigma_mod), operateur(Im_starfish).noise(sigma=sigma_high)

    Im_butterfly_denoised_sig15, Tb15 = fista(Im_butterfly_noised_sig15, "none", None, 0.01, 0.25, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    Im_butterfly_denoised_sig25, Tb25 = fista(Im_butterfly_noised_sig25, "none", None, 0.01, 1e-3, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_butterfly_denoised_sig50, Tb50 = fista(Im_butterfly_noised_sig50, "none", None, 0.01, 1e-3, 9, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_leaves_denoised_sig15, Tl15 = fista(Im_leaves_noised_sig15, "none", None, 0.01, 0.25, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    Im_leaves_denoised_sig25, Tl25 = fista(Im_leaves_noised_sig25, "none", None, 0.01, 0.01, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_leaves_denoised_sig50, Tl50 = fista(Im_leaves_noised_sig50, "none", None, 0.01, 0.01, 9, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_starfish_denoised_sig15, Ts15 = fista(Im_starfish_noised_sig15, "none", None, 0.01, 0.25, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    Im_starfish_denoised_sig25, Ts25 = fista(Im_starfish_noised_sig25, "none", None, 0.01, 0.01, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_starfish_denoised_sig50, Ts50 = fista(Im_starfish_noised_sig50, "none", None, 0.01, 0.01, 9, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig50, 1.0))

    
    # # # PNP(Plug and Play Algorithms)

    # # PNP_Débruiteur (Im_butterfly)

    denoiser = DRUNet()

    Im_butterfly_denoised_sig15_pnp, Tb15_pnp = pnp_apgm(Im_butterfly_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=5, tol=1e-7)

    Im_butterfly_denoised_sig25_pnp, Tb25_pnp = pnp_apgm(Im_butterfly_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=5, tol=1e-7)

    Im_butterfly_denoised_sig50_pnp, Tb50_pnp = pnp_pgm(Im_butterfly_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=5, tol=1e-7)

    Im_leaves_denoised_sig15_pnp, Tl15_pnp = pnp_apgm(Im_leaves_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=5, tol=1e-7)

    Im_leaves_denoised_sig25_pnp, Tl25_pnp = pnp_apgm(Im_leaves_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=5, tol=1e-7)

    Im_leaves_denoised_sig50_pnp, Tl50_pnp = pnp_pgm(Im_leaves_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=5, tol=1e-7)

    Im_starfish_denoised_sig15_pnp, Ts15_pnp = pnp_apgm(Im_starfish_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=5, tol=1e-7)

    Im_starfish_denoised_sig25_pnp, Ts25_pnp = pnp_apgm(Im_starfish_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=5, tol=1e-7)

    Im_starfish_denoised_sig50_pnp, Ts50_pnp = pnp_pgm(Im_starfish_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=5, tol=1e-7)

    # print(PSNR(Im_starfish, Im_starfish_denoised_sig50_pnp, 1.0))


    # # # '''Recherche paramètres optimaux (Proxy and PNP)'''
    
    # # Proxy

    # param_ranges = { "K": [5, 10, 15, 20], "lambd": [0.01, 0.1, 0.5], "tau": [0.01, 0.1, 0.5]}

    # prox_params_ranges = {"tau": [0.01, 0.1, 0.5],  "K": [5, 10, 15, 20]}

    # func_params = {"u": Im_butterfly_noised_sig50, "operator_type": "none", "operator_params": {}, "prox": prox_l6, "tol": 1e-7}

    # best_params, best_score, score_map = search_opt(func=fista, u_truth=Im_butterfly, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)


    # # PNP

    # param_ranges_pnp = { "K": [5, 10, 15], "tau": [0.01, 0.1, 1], "sigma": [sigma_low, sigma_mod, sigma_high]}
    
    # func_params_pnp = {"u": Im_leaves_noised_sig50, "operator_type": "none", "operator_params": {}, "denoiser": denoiser, "tol": 1e-7}

    # best_params, best_score, score_map = search_opt(func=pnp_apgm, u_truth=Im_leaves, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)


    # # Résultats

    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # Save images 
    
    # Images Denoising

    reference_image_list = [Im_butterfly, Im_leaves, Im_starfish]

    names_denoised_sig15 = [["Low Noise ($\sigma=0.06$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                            ["Low Noise ($\sigma=0.06$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                            ["Low Noise ($\sigma=0.06$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]

    images_denoised_sig15 =[[Im_butterfly_noised_sig15, Im_butterfly_denoised_sig15, Im_butterfly_denoised_sig15_pnp],
                            [Im_leaves_noised_sig15, Im_leaves_denoised_sig15, Im_leaves_denoised_sig15_pnp],
                            [Im_starfish_noised_sig15, Im_starfish_denoised_sig15, Im_starfish_denoised_sig15_pnp]]

    denoise_trajectories_sig_15 = [[Tb15, Tb15_pnp], [Tl15, Tl15_pnp], [Ts15, Ts15_pnp]]

    names_denoised_sig25 = [["Moderate Noise ($\sigma=0.1$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                            ["Moderate Noise ($\sigma=0.1$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                            ["Moderate Noise ($\sigma=0.1$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]

    images_denoised_sig25 =[[Im_butterfly_noised_sig25, Im_butterfly_denoised_sig25, Im_butterfly_denoised_sig25_pnp],
                            [Im_leaves_noised_sig25, Im_leaves_denoised_sig25, Im_leaves_denoised_sig25_pnp],
                            [Im_starfish_noised_sig25, Im_starfish_denoised_sig25, Im_starfish_denoised_sig25_pnp]]

    denoise_trajectories_sig_25 = [[Tb25, Tb25_pnp], [Tl25, Tl25_pnp], [Ts25, Ts25_pnp]]

    names_denoised_sig50 = [["High Noise ($\sigma=0.2$)", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                            ["High Noise ($\sigma=0.2$)", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                            ["High Noise ($\sigma=0.2$)", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]

    images_denoised_sig50 =[[Im_butterfly_noised_sig50, Im_butterfly_denoised_sig50, Im_butterfly_denoised_sig50_pnp],
                            [Im_leaves_noised_sig50, Im_leaves_denoised_sig50, Im_leaves_denoised_sig50_pnp],
                            [Im_starfish_noised_sig50, Im_starfish_denoised_sig50, Im_starfish_denoised_sig50_pnp]]

    denoise_trajectories_sig_50 = [[Tb50, Tb50_pnp], [Tl50, Tl50_pnp], [Ts50, Ts50_pnp]]


    save_path("results", "Imgs_denoised_sig15", names_denoised_sig15, images_denoised_sig15, reference_image_list= reference_image_list, 
              psnr=True, trajectories_list=denoise_trajectories_sig_15)
    
    save_path("results", "Imgs_denoised_sig25", names_denoised_sig25, images_denoised_sig25, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=denoise_trajectories_sig_25)

    save_path("results", "Imgs_denoised_sig50", names_denoised_sig50, images_denoised_sig50, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=denoise_trajectories_sig_50)


    # import numpy as np
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(18, 6), dpi=200)
    # plt.subplot(131)
    # plt.imshow(Im_butterfly)
    # plt.subplot(132)
    # plt.imshow(Im_butterfly_noised_sig50)
    # plt.subplot(133)
    # plt.imshow(Im_butterfly_denoised_sig50)

    # print([PSNR(Im_butterfly, Im_butterfly_denoised_sig50_pnp),
    #        PSNR(Im_leaves, Im_leaves_denoised_sig50_pnp),
    #         PSNR(Im_starfish, Im_starfish_denoised_sig50_pnp)])

if __name__ == "__main__" :

    main()