import torch
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt, process_image_2
from Proxy_Func import fista, prox_l6
from Denoisers import DRUNet
from Pnp_Algorithms import pnp_pgm, pnp_apgm, pnp_apgm2
import pandas as pd

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")


    # # # Proxy Algorithms

   
    # #  '''Test Inpainting'''

    sigma_low, sigma_mod, sigma_high = 0.06, 0.1, 0.2

    simple_mask = torch.ones(1, 1, 256, 256)

    simple_mask[:, :, 0::24, :] = 0

    simple_mask[:, :, :, 0::24] = 0

    simple_mask = simple_mask.to(torch.bool) # mask.type(torch.uint8)

    medium_mask = torch.rand(1, 1, 256, 256) > 0.5

    complex_mask = torch.rand(1, 1, 256, 256) > 0.8

    Im_butterfly_masked_simple, Mask_butterfly_simple = operateur(Im_butterfly).inpaint(mask=simple_mask, sigma=sigma_low)

    Im_butterfly_masked_medium, Mask_butterfly_medium = operateur(Im_butterfly).inpaint(mask=medium_mask, sigma=sigma_mod)

    Im_butterfly_masked_complex, Mask_butterfly_complex = operateur(Im_butterfly).inpaint(mask=complex_mask, sigma=sigma_high)

    Im_leaves_masked_simple, Mask_leaves_simple = operateur(Im_leaves).inpaint(mask=simple_mask, sigma=sigma_low)

    Im_leaves_masked_medium, Mask_leaves_medium = operateur(Im_leaves).inpaint(mask=medium_mask, sigma=sigma_mod)

    Im_leaves_masked_complex, Mask_leaves_complex = operateur(Im_leaves).inpaint(mask=complex_mask, sigma=sigma_high)

    Im_starfish_masked_simple, Mask_star_simple = operateur(Im_starfish).inpaint(mask=simple_mask, sigma=sigma_low)

    Im_starfish_masked_medium, Mask_star_medium = operateur(Im_starfish).inpaint(mask=medium_mask, sigma=sigma_mod)

    Im_starfish_masked_complex, Mask_star_complex = operateur(Im_starfish).inpaint(mask=complex_mask, sigma=sigma_high)

    Im_butterfly_demasked_simple, Tb_simple = fista(Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    Im_butterfly_demasked_medium, Tb_medium = fista(Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    Im_butterfly_demasked_complex, Tb_complex = fista(Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)
    
    Im_leaves_demasked_simple, Tl_simple = fista(Im_leaves_masked_simple, "mask", {"Mask": Mask_leaves_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    Im_leaves_demasked_medium, Tl_medium = fista(Im_leaves_masked_medium, "mask", {"Mask": Mask_leaves_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    Im_leaves_demasked_complex, Tl_complex = fista(Im_leaves_masked_complex, "mask", {"Mask": Mask_leaves_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)

    Im_starfish_demasked_simple, Ts_simple = fista(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    Im_starfish_demasked_medium, Ts_medium = fista(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    Im_starfish_demasked_complex, Ts_complex = fista(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)
                               
    # print(PSNR(Im_butterfly, Im_butterfly_demasked_simple, 1.0))

    
    # # # PNP(Plug and Play Algorithms)


    # # PNP_Inpainting

    denoiser = DRUNet()

    Im_butterfly_demasked_simple_pnp, Tb_simple_pnp = pnp_pgm(Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 1.5, denoiser, sigma=1.2e-1, K=400, tol=1e-7, init=False)

    Im_butterfly_demasked_medium_pnp, Tb_medium_pnp = pnp_pgm(Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 1.5, denoiser, sigma=1.3e-1, K=200, tol=1e-7, init=False)

    Im_butterfly_demasked_complex_pnp, Tb_complex_pnp = pnp_pgm(Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 1.5, denoiser, sigma=2e-1, K=200, tol=1e-7, init=False)

    Im_leaves_demasked_simple_pnp, Tl_simple_pnp = pnp_pgm(Im_leaves_masked_simple, "mask", {"Mask": Mask_leaves_simple}, 1.5, denoiser, sigma=1.3e-1, K=400, tol=1e-7, init=False)

    Im_leaves_demasked_medium_pnp, Tl_medium_pnp = pnp_pgm(Im_leaves_masked_medium, "mask", {"Mask": Mask_leaves_medium}, 1.5, denoiser, sigma=1.9e-1, K=200, tol=1e-7, init=False)  

    Im_leaves_demasked_complex_pnp, Tl_complex_pnp = pnp_pgm(Im_leaves_masked_complex, "mask", {"Mask": Mask_leaves_complex}, 1.7, denoiser, sigma=1.7e-1, K=200, tol=1e-7, init=False)  

    Im_starfish_demasked_simple_pnp, Ts_simple_pnp = pnp_pgm(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 1.5, denoiser, sigma=1.2e-1, K=400, tol=1e-7, init=False)

    Im_starfish_demasked_medium_pnp, Ts_medium_pnp = pnp_pgm(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 1.5, denoiser, sigma=1.2e-1, K=200, tol=1e-7, init=False)

    Im_starfish_demasked_complex_pnp, Ts_complex_pnp = pnp_pgm(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 1.9, denoiser, sigma=1.5e-1, K=200, tol=1e-7, init=False)

    # print(PSNR(Im_starfish, Im_starfish_demasked_medium_pnp, 1.0))

    
    # # # '''Recherche paramètres optimaux (Proxy and PNP)'''

    
    # Proxy

    # param_ranges = { "K": [10, 20, 30], "lambd": [0.01, 0.1, 1], "tau": [0.01, 0.1, 1]}

    # prox_params_ranges = {"tau": [0.01, 0.1, 1],  "K": [10, 20, 30]}

    # func_params = {"u": Im_starfish_masked_complex, "operator_type": "mask", "operator_params": {"Mask": Mask_star_complex}, "prox": prox_l6, "tol": 1e-7, "init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=fista, u_truth=Im_starfish, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)


    # # PNP

    # param_ranges_pnp = {"K": [10, 20, 30], "tau": [0.01, 0.1, 1], "sigma": [sigma_low, sigma_mod, sigma_high]}

    # param_ranges_pnp = {"K": [10, 20, 30], "tau": [0.01, 0.1, 1], "sigma": [0.12, 0.13, 0.14], "alpha": [1/2, 1], "z": [1, 10, 100]}

    # func_params_pnp = {"u": Im_leaves_masked_complex, "operator_type": "mask", "operator_params": {"Mask": Mask_leaves_complex}, "denoiser": denoiser, "tol": 1e-7, "init": False, "verbose": False}

    # best_params, best_score, score_map = search_opt(func=pnp_apgm, u_truth=Im_starfish, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)

    # best_params, best_score, score_map = search_opt(func=pnp_apgm2, u_truth=Im_leaves, param_ranges=param_ranges_pnp, metric=PSNR, func_params=func_params_pnp)

    # # Résultats

    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # # Save images 

    # # Images Inpainting

    reference_image_list = [Im_butterfly, Im_leaves, Im_starfish]

    names_inpaint_simple = [["Simple Mask with Low Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                            ["Simple Mask with Low Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                            ["Simple Mask with Low Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]
    
    images_inpaint_simple = [[Im_butterfly_masked_simple, Im_butterfly_demasked_simple, Im_butterfly_demasked_simple_pnp], 
                             [Im_leaves_masked_simple, Im_leaves_demasked_simple, Im_leaves_demasked_simple_pnp], 
                             [Im_starfish_masked_simple, Im_starfish_demasked_simple, Im_starfish_demasked_simple_pnp]]

    inpaint_trajectories_simple = [[Tb_simple, Tb_simple_pnp], [Tl_simple, Tl_simple_pnp], [Ts_simple, Ts_simple_pnp]]

    names_inpaint_medium = [["Medium Mask with Moderate Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                            ["Medium Mask with Moderate Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                            ["Medium Mask with Moderate Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]

    images_inpaint_medium = [[Im_butterfly_masked_medium, Im_butterfly_demasked_medium, Im_butterfly_demasked_medium_pnp], 
                             [Im_leaves_masked_medium, Im_leaves_demasked_medium, Im_leaves_demasked_medium_pnp], 
                             [Im_starfish_masked_medium, Im_starfish_demasked_medium, Im_starfish_demasked_medium_pnp]]

    inpaint_trajectories_medium =[[Tb_medium, Tb_medium_pnp], [Tl_medium, Tl_medium_pnp], [Ts_medium, Ts_medium_pnp]]

    names_inpaint_complex = [["Complex Mask with High Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)"], 
                             ["Complex Mask with High Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)"], 
                             ["Complex Mask with High Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)"]]

    images_inpaint_complex = [[Im_butterfly_masked_complex, Im_butterfly_demasked_complex, Im_butterfly_demasked_complex_pnp], 
                              [Im_leaves_masked_complex, Im_leaves_demasked_complex, Im_leaves_demasked_complex_pnp], 
                              [Im_starfish_masked_complex, Im_starfish_demasked_complex, Im_starfish_demasked_complex_pnp]]

    inpaint_trajectories_complex =[[Tb_complex, Tb_complex_pnp], [Tl_complex, Tl_complex_pnp], [Ts_complex, Ts_complex_pnp]]

    save_path("results", "Imgs_inpaint_simple", names_inpaint_simple, images_inpaint_simple, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=inpaint_trajectories_simple)

    save_path("results", "Imgs_inpaint_medium", names_inpaint_medium, images_inpaint_medium, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=inpaint_trajectories_medium)

    save_path("results", "Imgs_inpaint_complex", names_inpaint_complex, images_inpaint_complex, reference_image_list=reference_image_list, 
              psnr=True, trajectories_list=inpaint_trajectories_complex)
    

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(18, 6), dpi=200)
    # plt.subplot(131)
    # plt.imshow(Im_starfish_masked_medium)
    # plt.subplot(132)
    # plt.imshow(Mask_star_medium)
    # plt.subplot(133)
    # plt.imshow(Im_starfish_demasked_medium_pnp)

if __name__ == "__main__" :

    main()