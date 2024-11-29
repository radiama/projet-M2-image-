import torch
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt
from Proxy_Func import fista, prox_l6
from Denoisers import DRUNet, DnCNN
from Pnp_Algorithms import pnp_apgm
import pandas as pd

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")


    # # # Proxy Algorithms


    # # '''Test Bruitage avec Im_Butterfly'''

    # sigma_low, sigma_mod, sigma_high = 15/255, 25/255, 50/255

    # Im_butterfly_noised_sig15, Im_butterfly_noised_sig25, Im_butterfly_noised_sig50 = operateur(Im_butterfly).noise(sigma=sigma_low), operateur(Im_butterfly).noise(sigma=sigma_mod), operateur(Im_butterfly).noise(sigma=sigma_high)

    # Im_butterfly_denoised_sig15, Tb15 = fista(Im_butterfly_noised_sig15, "none", None, 0.01, 0.25, 10, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    # Im_butterfly_denoised_sig25, Tb25 = fista(Im_butterfly_noised_sig25, "none", None, 0.01, 0.01, 5, prox=prox_l6, prox_params={"tau": 0.1, "K": 10}, tol=1e-7)

    # Im_butterfly_denoised_sig50, Tb50 = fista(Im_butterfly_noised_sig50, "none", None, 0.01, 0.01, 10, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig50, 1.0))


    # # '''Test Floutage avec Im_leaves'''

    # sigma_low_2, sigma_mod_2, sigma_high_2 = (1, 1), (2, 2), (3, 3)
    
    # Im_leaves_blurred_low, G_leaves_low = operateur(Im_leaves).blur(sigma = sigma_low_2, angle = 0)
    
    # Im_leaves_blurred_mod, G_leaves_mod = operateur(Im_leaves).blur(sigma = sigma_mod_2, angle = 0)

    # Im_leaves_blurred_high, G_leaves_high = operateur(Im_leaves).blur(sigma = sigma_high_2, angle = 0)
    
    # Im_leaves_deblurred_low, Tl_low = fista(Im_leaves_blurred_low, "convolution", {"G": G_leaves_low}, 0.0001, 1, 25, prox=prox_l6, prox_params={"tau": 0.0001, "K": 20}, tol=1e-7)

    # Im_leaves_deblurred_mod, Tl_mod = fista(Im_leaves_blurred_mod, "convolution", {"G": G_leaves_mod}, 0.0001, 1, 30, prox=prox_l6, prox_params={"tau": 0.0001, "K": 20}, tol=1e-7)

    # Im_leaves_deblurred_high, Tl_high = fista(Im_leaves_blurred_high, "convolution", {"G": G_leaves_high}, 0.0001, 1, 30, prox=prox_l6, prox_params={"tau": 0.5, "K": 20}, tol=1e-7)

    # print(PSNR(Im_leaves, Im_leaves_deblurred_high, 1.0))

   
    # # # '''Test Inpainting avec Im_starfish'''

    # simple_mask = torch.ones(1, 1, 256, 256)

    # simple_mask[:, :, 0::24, :] = 0

    # simple_mask[:, :, :, 0::24] = 0

    # simple_mask = simple_mask.to(torch.bool) # mask.type(torch.uint8)

    # medium_mask = torch.rand(1, 1, 256, 256) > 0.5

    # complex_mask = torch.rand(1, 1, 256, 256) > 0.8

    # Im_starfish_masked_simple, Mask_star_simple = operateur(Im_starfish).inpaint(mask =simple_mask , sigma=sigma_low)

    # Im_starfish_masked_medium, Mask_star_medium = operateur(Im_starfish).inpaint(mask =medium_mask , sigma=sigma_mod)

    # Im_starfish_masked_complex, Mask_star_complex = operateur(Im_starfish).inpaint(mask =complex_mask , sigma=sigma_high)

    # Im_starfish_demasked_simple, Ts_simple = fista(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 0.01, 0.25, 25, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7)

    # Im_starfish_demasked_meduim, Ts_medium = fista(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 1, 0.5, 25, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7)

    # Im_starfish_demasked_complex, Ts_complex = fista(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 1, 0.5, 25, prox=prox_l6, prox_params={"tau": 0.01, "K": 15}, tol=1e-7)
    
    # print(PSNR(Im_starfish, Im_starfish_demasked_simple, 1.0))

    
    # # # PNP(Plug and Play Algorithms)

    # # PNP_Débruiteur (Im_butterfly)

    # denoiser = DRUNet()

    # Im_butterfly_denoised_sig15_pnp, Tb15_pnp = pnp_apgm(Im_butterfly_noised_sig15, "none", None, 1, denoiser, sigma=sigma_low, K=15, tol=1e-7)

    # Im_butterfly_denoised_sig25_pnp, Tb25_pnp = pnp_apgm(Im_butterfly_noised_sig25, "none", None, 1, denoiser, sigma=sigma_mod, K=15, tol=1e-7)

    # Im_butterfly_denoised_sig50_pnp, Tb50_pnp = pnp_apgm(Im_butterfly_noised_sig50, "none", None, 1, denoiser, sigma=sigma_high, K=15, tol=1e-7)

    # print(PSNR(Im_butterfly, Im_butterfly_denoised_sig50_pnp, 1.0))


    # # PNP_Floutage (Im_leaves)

    # denoiser = DnCNN()

    # Im_leaves_deblurred_low_pnp, Tl_low_pnp = pnp_apgm(Im_leaves_blurred_low, "convolution", {"G": G_leaves_low}, 1.5, denoiser, sigma=None, K=15, tol=1e-7)

    # Im_leaves_deblurred_mod_pnp, Tl_mod_pnp = pnp_apgm(Im_leaves_blurred_mod, "convolution", {"G": G_leaves_mod}, 1.5, denoiser, sigma=None, K=25, tol=1e-7)

    # Im_leaves_deblurred_high_pnp, Tl_high_pnp = pnp_apgm(Im_leaves_blurred_high, "convolution", {"G": G_leaves_high}, 1.5, denoiser, sigma=None , K=25, tol=1e-7)

    # print(PSNR(Im_leaves, Im_leaves_deblurred_high_pnp, 1.0))


    # # PNP_Inpainting (Im_starfish)

    # denoiser = DRUNet()

    # Im_starfish_demasked_simple_pnp, Ts_simple_pnp = pnp_apgm(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 1, denoiser, sigma=0.17, K=25, tol=1e-7)

    # Im_starfish_demasked_meduim_pnp, Ts_medium_pnp = pnp_apgm(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 1, denoiser, sigma=0.13, K=25, tol=1e-7)

    # Im_starfish_demasked_complex_pnp, Ts_complex_pnp = pnp_apgm(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 1, denoiser, sigma=0.17, K=25, tol=1e-7)
    
    # print(PSNR(Im_starfish, Im_starfish_demasked_complex_pnp, 1.0))

    
    # # # '''Recherche paramètres optimaux (Proxy and PNP)'''
    
    # # Proxy

    # param_ranges = { "K": [5, 10, 20], "lambd": [0.01, 0.1, 1], "tau": [0.01, 0.1, 1]}

    # prox_params_ranges = {"tau": [0.01, 0.1, 1],  "K": [5, 10, 20]}

    # func_params = {"u": Im_butterfly_noised_sig50, "operator_type": "none", "operator_params": {}, "prox": prox_l6, "tol": 1e-7}
    # func_params = {"u": Im_starfish_masked_simple, "operator_type": "mask", "operator_params": {"Mask": Mask_star_simple}, "prox": prox_l6, "tol": 1e-7}
    # func_params = {"u": Im_leaves_blurred_high, "operator_type": "convolution", "operator_params": {"G": G_leaves_high}, "prox": prox_l6, "tol": 1e-7}

    # best_params, best_score, score_map = search_opt(func=fista, u_truth=Im_leaves, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)

    # # PNP

    # param_ranges_pnp = { "K": [15, 25, 50], "tau": [0.25, 0.5, 1], "sigma": [sigma_low, sigma_mod, sigma_high]}

    # param_ranges_pnp = { "K": [15, 25, 50], "tau": [0.25, 0.5, 1, 1.5]}
    
    # func_params_pnp = {"u": Im_butterfly_noised_sig15, "operator_type": "none", "operator_params": {}, "denoiser": denoiser, "tol": 1e-7}

    # func_params_pnp = {"u": Im_leaves_blurred_high, "operator_type": "convolution", "operator_params": {"G": G_leaves_high}, "denoiser": denoiser, "sigma": None , "tol": 1e-7}

    # func_params_pnp = {"u": Im_starfish_masked_simple, "operator_type": "mask", "operator_params": {"Mask": Mask_star_simple}, "denoiser": denoiser, "tol": 1e-7}

    # best_params, best_score, score_map = search_opt(func=pnp_apgm, u_truth=Im_leaves, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)

    # # Résultats

    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # # Save images 
    
    # # Image Denoising (Im_butterfly)

    # names_denoised_sig15 = ["Low Noise ($\sigma=15$)", "FISTA Denoising ($\sigma=15$)", "PNP_DRUNet Denoising ($\sigma=15$)"]

    # images_denoised_sig15 =[Im_butterfly_noised_sig15, Im_butterfly_denoised_sig15, Im_butterfly_denoised_sig15_pnp]

    # denoise_trajectories_sig_15 = [Tb15, Tb15_pnp]

    # names_denoised_sig25 = ["Moderate Noise ($\sigma=25$)", "FISTA Denoising ($\sigma=25$)", "PNP_DRUNet Denoising ($\sigma=25$)"]

    # images_denoised_sig25 =[Im_butterfly_noised_sig25, Im_butterfly_denoised_sig25, Im_butterfly_denoised_sig25_pnp]

    # denoise_trajectories_sig_25 = [Tb25, Tb25_pnp]

    # names_denoised_sig50 = ["High Noise ($\sigma=50$)", "FISTA Denoising ($\sigma=50$)", "PNP_DRUNet Denoising ($\sigma=50$)"]

    # images_denoised_sig50 =[Im_butterfly_noised_sig50, Im_butterfly_denoised_sig50, Im_butterfly_denoised_sig50_pnp]

    # denoise_trajectories_sig_50 = [Tb50, Tb50_pnp]

    # save_path("results", "Im_butterfly_denoised_15", names_denoised_sig15, images_denoised_sig15, reference_image=Im_butterfly, psnr=True, trajectories=denoise_trajectories_sig_15)
    
    # save_path("results", "Im_butterfly_denoised_25", names_denoised_sig25, images_denoised_sig25, reference_image=Im_butterfly, psnr=True, trajectories=denoise_trajectories_sig_25)

    # save_path("results", "Im_butterfly_denoised_50", names_denoised_sig50, images_denoised_sig50, reference_image=Im_butterfly, psnr=True, trajectories=denoise_trajectories_sig_50)



    # # Image Deblurring (Im_leaves)

    # names_deblurred_sig_1 = ["Low Blur ($\sigma=1$)", "FISTA Deblurring ($\sigma=1$)", "PNP_DnCNN Deblurring ($\sigma=1$)"]

    # images_deblurred_sig_1 =[Im_leaves_blurred_low, Im_leaves_deblurred_low, Im_leaves_deblurred_low_pnp]

    # deblur_trajectories_sig_1 = [Tl_low, Tl_low_pnp]

    # names_deblurred_sig_2 = ["Moderate Blur ($\sigma=2$)", "FISTA Deblurring ($\sigma=2$)", "PNP_DnCNN Deblurring ($\sigma=2$)"]

    # images_deblurred_sig_2 =[Im_leaves_blurred_mod, Im_leaves_deblurred_mod, Im_leaves_deblurred_mod_pnp]

    # deblur_trajectories_sig_2 = [Tl_mod, Tl_mod_pnp]

    # names_deblurred_sig_3 = ["High Blur ($\sigma=3$)", "FISTA Deblurring ($\sigma=3$)", "PNP_DnCNN Deblurring ($\sigma=3$)"]

    # images_deblurred_sig_3 =[Im_leaves_blurred_high, Im_leaves_deblurred_high, Im_leaves_deblurred_high_pnp]

    # deblur_trajectories_sig_3 = [Tl_high, Tl_high_pnp]

    # save_path("results", "Im_leaves_deblurred_sig_1", names_deblurred_sig_1, images_deblurred_sig_1, reference_image=Im_leaves, psnr=True, trajectories=deblur_trajectories_sig_1)
    
    # save_path("results", "Im_leaves_deblurred_sig_2", names_deblurred_sig_2, images_deblurred_sig_2, reference_image=Im_leaves, psnr=True, trajectories=deblur_trajectories_sig_2)

    # save_path("results", "Im_leaves_deblurred_sig_3", names_deblurred_sig_3, images_deblurred_sig_3, reference_image=Im_leaves, psnr=True, trajectories=deblur_trajectories_sig_3)


    # # Image Inpainting (Im_starfish)

    # names_inpaint_simple = ["Simple Mask with Low Noise", "FISTA Inpainting", "PNP_DRUNet Inpainting"]
    
    # images_inpaint_simple = [Im_starfish_masked_simple, Im_starfish_demasked_simple, Im_starfish_demasked_simple_pnp]

    # inpaint_trajectories_simple =[Ts_simple, Ts_simple_pnp]
    
    # names_inpaint_medium = ["Medium Mask with Moderate Noise", "FISTA Inpainting", "PNP_DRUNet Inpainting"] 

    # images_inpaint_medium = [ Im_starfish_demasked_meduim_pnp, Im_starfish_demasked_meduim, Im_starfish_demasked_complex_pnp]

    # inpaint_trajectories_medium =[Ts_medium, Ts_medium_pnp]

    # names_inpaint_complex = ["Complex Mask with High Noise", "FISTA Inpainting", "PNP_DRUNet Inpainting"]

    # images_inpaint_complex = [Im_starfish_masked_complex,Im_starfish_demasked_complex, Im_starfish_demasked_complex_pnp]

    # inpaint_trajectories_complex =[Ts_complex, Ts_complex_pnp]

    # save_path("results", "Im_starfish_inpaint_simple", names_inpaint_simple, images_inpaint_simple, reference_image=Im_starfish, psnr=True, trajectories=inpaint_trajectories_simple)

    # save_path("results", "Im_starfish_inpaint_medium", names_inpaint_medium, images_inpaint_medium, reference_image=Im_starfish, psnr=True, trajectories=inpaint_trajectories_medium)

    # save_path("results", "Im_starfish_inpaint_complex", names_inpaint_complex, images_inpaint_complex, reference_image=Im_starfish, psnr=True, trajectories=inpaint_trajectories_complex)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8,4))
    # plt.subplot(121)
    # plt.imshow(Im_butterfly_noised_sig50)
    # plt.subplot(122)
    # plt.imshow(Im_butterfly_denoised_sig50_pnp)

if __name__ == "__main__" :

    main()