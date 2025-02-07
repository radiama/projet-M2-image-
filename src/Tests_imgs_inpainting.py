import torch
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt, process_image_2, process_image_3, timer
from Proxy_Func import fista, prox_l6, forward_backward
from Denoisers import DRUNet
from Pnp_Algorithms import pnp_pgm, pnp_apgm, pnp_apgm2
import pandas as pd, os
from Convex_ridge_regularizer_rev import CRRNN, ConvexRidgeRegularizer
from Utils_rev import tStepDenoiser

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

    # Im_butterfly_demasked_simple, Tb_simple = fista(Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    # Im_butterfly_demasked_medium, Tb_medium = fista(Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    # Im_butterfly_demasked_complex, Tb_complex = fista(Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)
    
    # Im_leaves_demasked_simple, Tl_simple = fista(Im_leaves_masked_simple, "mask", {"Mask": Mask_leaves_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    # Im_leaves_demasked_medium, Tl_medium = fista(Im_leaves_masked_medium, "mask", {"Mask": Mask_leaves_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    # Im_leaves_demasked_complex, Tl_complex = fista(Im_leaves_masked_complex, "mask", {"Mask": Mask_leaves_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)

    # Im_starfish_demasked_simple, Ts_simple = fista(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)

    # Im_starfish_demasked_medium, Ts_medium = fista(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)

    # Im_starfish_demasked_complex, Ts_complex = fista(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)
                               
    # print(PSNR(Im_butterfly, Im_butterfly_demasked_simple, 1.0))

    
    # # # PNP(Plug and Play Algorithms)


    # # PNP_Inpainting

    denoiser = DRUNet()

    # Im_butterfly_demasked_simple_pnp, Tb_simple_pnp = pnp_pgm(Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 1.5, denoiser, sigma=1.2e-1, K=400, tol=1e-7, init=False)

    # Im_butterfly_demasked_medium_pnp, Tb_medium_pnp = pnp_pgm(Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 1.5, denoiser, sigma=1.3e-1, K=200, tol=1e-7, init=False)

    # Im_butterfly_demasked_complex_pnp, Tb_complex_pnp = pnp_pgm(Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 1.5, denoiser, sigma=2e-1, K=200, tol=1e-7, init=False)

    # Im_leaves_demasked_simple_pnp, Tl_simple_pnp = pnp_pgm(Im_leaves_masked_simple, "mask", {"Mask": Mask_leaves_simple}, 1.5, denoiser, sigma=1.3e-1, K=400, tol=1e-7, init=False)

    # Im_leaves_demasked_medium_pnp, Tl_medium_pnp = pnp_pgm(Im_leaves_masked_medium, "mask", {"Mask": Mask_leaves_medium}, 1.5, denoiser, sigma=2e-1, K=200, tol=1e-7, init=False)  

    # Im_leaves_demasked_complex_pnp, Tl_complex_pnp = pnp_pgm(Im_leaves_masked_complex, "mask", {"Mask": Mask_leaves_complex}, 1.7, denoiser, sigma=1.7e-1, K=200, tol=1e-7, init=False)  

    # Im_starfish_demasked_simple_pnp, Ts_simple_pnp = pnp_pgm(Im_starfish_masked_simple, "mask", {"Mask": Mask_star_simple}, 1.5, denoiser, sigma=1.3e-1, K=400, tol=1e-7, init=False)

    # Im_starfish_demasked_medium_pnp, Ts_medium_pnp = pnp_pgm(Im_starfish_masked_medium, "mask", {"Mask": Mask_star_medium}, 1.5, denoiser, sigma=1.2e-1, K=200, tol=1e-7, init=False)

    # Im_starfish_demasked_complex_pnp, Ts_complex_pnp = pnp_pgm(Im_starfish_masked_complex, "mask", {"Mask": Mask_star_complex}, 1.9, denoiser, sigma=1.5e-1, K=200, tol=1e-7, init=False)

    # print(PSNR(Im_starfish, Im_starfish_demasked_medium_pnp, 1.0))


    # # CRRNN_Débruiteur

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    epoch, denoise = 10, tStepDenoiser
    training_name_5 = f'Sigma_{5}_t_{5}'
    training_name_25 = f'Sigma_{25}_t_{30}'
    checkpoint_dir_5 = os.path.join(parent_dir, f'trained_models/{training_name_5}/checkpoints/checkpoint_{epoch}.pth')
    checkpoint_dir_25 = os.path.join(parent_dir, f'trained_models/{training_name_25}/checkpoints/checkpoint_{epoch}.pth')
    checkpoint_dir_inp_5 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_inpaint_5.pth')
    checkpoint_dir_inp_15 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_inpaint_15.pth')
    checkpoint_dir_inp_25 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_inpaint_25.pth')
    checkpoint_dir_inp_50 = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_inpaint_50.pth')
    
    # checkpoint_dir = os.path.join(parent_dir, f'trained_models/IOD_Training/crr_nn_best_model_{noise}.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modele_0 = ConvexRidgeRegularizer(kernel_size=7, channels=[1, 8, 32], activation_params={"knots_range": 0.1, "n_channels": 32, "n_knots": 21})   

    # model_5=CRRNN(model=modele_0, name_model_pre=checkpoint_dir_5, device=device, load=True)
    # model_inp_5 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_5, device=device, load=True, checkpoint=False)
    # model_inp_15 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_15, device=device, load=True, checkpoint=False)
    model_inp_25 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_25, device=device, load=True, checkpoint=False)
    # model_inp_50 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_50, device=device, load=True, checkpoint=False)

    # with torch.no_grad():

        # Im_butterfly_demasked_simple_crrnn, Tb_simple_crrnn = process_image_3(Im_butterfly_masked_simple, operator=denoise, model=model_inp_5, t_steps=400, operator_type ="mask", operator_params={"Mask": Mask_butterfly_simple}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_butterfly_demasked_medium_crrnn, Tb_medium_crrnn = process_image_3(Im_butterfly_masked_medium, operator=denoise, model=model_inp_25, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_butterfly_medium}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_butterfly_demasked_complex_crrnn, Tb_complex_crrnn = process_image_3(Im_butterfly_masked_complex, operator=denoise, model=model_5, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_butterfly_complex}, auto_params=False, lmbd=1e-4, mu=1e-0, step_size=1.9, init=False)

        # Im_leaves_demasked_simple_crrnn, Tl_simple_crrnn = process_image_3(Im_leaves_masked_simple, operator=denoise, model=model_inp_5, t_steps=400, operator_type ="mask", operator_params={"Mask": Mask_leaves_simple}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_leaves_demasked_medium_crrnn, Tl_medium_crrnn = process_image_3(Im_leaves_masked_medium, operator=denoise, model=model_inp_25, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_leaves_medium}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_leaves_demasked_complex_crrnn, Tl_complex_crrnn = process_image_3(Im_leaves_masked_complex, operator=denoise, model=model_5, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_leaves_complex}, auto_params=False, lmbd=1e-4, mu=1e-0, step_size=1.9, init=False)

        # Im_starfish_demasked_simple_crrnn, Ts_simple_crrnn = process_image_3(Im_starfish_masked_simple, operator=denoise, model=model_inp_5, t_steps=400, operator_type ="mask", operator_params={"Mask": Mask_star_simple}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_starfish_demasked_medium_crrnn, Ts_medium_crrnn = process_image_3(Im_starfish_masked_medium, operator=denoise, model=model_inp_25, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_star_medium}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

        # Im_starfish_demasked_complex_crrnn, Ts_complex_crrnn = process_image_3(Im_starfish_masked_complex, operator=denoise, model=model_5, t_steps=200, operator_type ="mask", operator_params={"Mask": Mask_star_complex}, auto_params=False, lmbd=1e-4, mu=1e-0, step_size=1.9, init=False)



    
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

    # reference_image_list = [Im_butterfly, Im_leaves, Im_starfish]

    # names_inpaint_simple = [["Simple Mask with Low Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                         ["Simple Mask with Low Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                         ["Simple Mask with Low Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]
    
    # images_inpaint_simple = [[Im_butterfly_masked_simple, Im_butterfly_demasked_simple, Im_butterfly_demasked_simple_pnp, Im_butterfly_demasked_simple_crrnn], 
    #                          [Im_leaves_masked_simple, Im_leaves_demasked_simple, Im_leaves_demasked_simple_pnp, Im_leaves_demasked_simple_crrnn], 
    #                          [Im_starfish_masked_simple, Im_starfish_demasked_simple, Im_starfish_demasked_simple_pnp, Im_starfish_demasked_simple_crrnn]]

    # inpaint_trajectories_simple = [[Tb_simple, Tb_simple_pnp, Tb_simple_crrnn], [Tl_simple, Tl_simple_pnp, Tl_simple_crrnn], [Ts_simple, Ts_simple_pnp, Ts_simple_crrnn]]

    # names_inpaint_medium = [["Medium Mask with Moderate Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                         ["Medium Mask with Moderate Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                         ["Medium Mask with Moderate Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    # images_inpaint_medium = [[Im_butterfly_masked_medium, Im_butterfly_demasked_medium, Im_butterfly_demasked_medium_pnp, Im_butterfly_demasked_medium_crrnn ], 
    #                          [Im_leaves_masked_medium, Im_leaves_demasked_medium, Im_leaves_demasked_medium_pnp, Im_leaves_demasked_medium_crrnn ], 
    #                          [Im_starfish_masked_medium, Im_starfish_demasked_medium, Im_starfish_demasked_medium_pnp, Im_starfish_demasked_medium_crrnn]]

    # inpaint_trajectories_medium =[[Tb_medium, Tb_medium_pnp, Tb_medium_crrnn], [Tl_medium, Tl_medium_pnp, Tl_medium_crrnn], [Ts_medium, Ts_medium_pnp, Ts_medium_crrnn]]

    # names_inpaint_complex = [["Complex Mask with High Noise", "TV (Img_Butterfly)", "DRUNet (Img_Butterfly)", "CRRNN (Img_Butterfly)"], 
    #                          ["Complex Mask with High Noise", "TV (Img_Leaves)", "DRUNet (Img_Leaves)", "CRRNN (Img_Leaves)"], 
    #                          ["Complex Mask with High Noise", "TV (Img_Starfish)", "DRUNet (Img_Starfish)", "CRRNN (Img_Starfish)"]]

    # images_inpaint_complex = [[Im_butterfly_masked_complex, Im_butterfly_demasked_complex, Im_butterfly_demasked_complex_pnp, Im_butterfly_demasked_complex_crrnn], 
    #                           [Im_leaves_masked_complex, Im_leaves_demasked_complex, Im_leaves_demasked_complex_pnp, Im_leaves_demasked_complex_crrnn], 
    #                           [Im_starfish_masked_complex, Im_starfish_demasked_complex, Im_starfish_demasked_complex_pnp, Im_starfish_demasked_complex_crrnn]]

    # inpaint_trajectories_complex =[[Tb_complex, Tb_complex_pnp, Tl_complex_crrnn], [Tl_complex, Tl_complex_pnp, Tl_complex_crrnn], [Ts_complex, Ts_complex_pnp, Ts_complex_crrnn]]

    # save_path("results_2", "Imgs_inpaint_simple", names_inpaint_simple, images_inpaint_simple, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=inpaint_trajectories_simple)

    # save_path("results_2", "Imgs_inpaint_medium", names_inpaint_medium, images_inpaint_medium, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=inpaint_trajectories_medium)

    # save_path("results", "Imgs_inpaint_complex", names_inpaint_complex, images_inpaint_complex, reference_image_list=reference_image_list, 
    #           psnr=True, trajectories_list=inpaint_trajectories_complex)
    

    # # Timer

    # Initialisation des listes pour stocker les temps d'exécution
    methods = ["TV", "DRUNet", "CRRNN"]
    sigma_values = [15, 25, 50]

    # Création d'un dictionnaire pour stocker les résultats
    Timer_map = {method:[] for method in methods}

    # Boucle sur les différents niveaux de bruit sigma ou méthode
        
    # Méthode 1 : TV

    time_tv_simple = timer(forward_backward, Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 
                                         0.1, 1, 400, prox=prox_l6, prox_params={"tau": 0.01, "K": 10}, tol=1e-7, init=False)
    
    time_tv_medium = timer(forward_backward, Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 
                        0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 20}, tol=1e-7, init=False)
    
    # time_tv_complex= timer(forward_backward, Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 
    #                     0.1, 1, 200, prox=prox_l6, prox_params={"tau": 0.01, "K": 30}, tol=1e-7, init=False)
    
    Timer_map["TV"].append(time_tv_simple)
    Timer_map["TV"].append(time_tv_medium)
    # Timer_map["TV"].append(time_tv_complex)

    # Méthode 2 : DRUNet

    time_drunet_simple = timer(pnp_pgm, Im_butterfly_masked_simple, "mask", {"Mask": Mask_butterfly_simple}, 
                            1.5, denoiser, sigma=1.2e-1, K=400, tol=1e-7, init=False)
    
    time_drunet_medium =  timer(pnp_pgm, Im_butterfly_masked_medium, "mask", {"Mask": Mask_butterfly_medium}, 
                             1.5, denoiser, sigma=1.3e-1, K=200, tol=1e-7, init=False)


    # time_drunet_complex = timer(pnp_pgm, Im_butterfly_masked_complex, "mask", {"Mask": Mask_butterfly_complex}, 
    #                         1.5, denoiser, sigma=2e-1, K=200, tol=1e-7, init=False)

    Timer_map["DRUNet"].append(time_drunet_simple)
    Timer_map["DRUNet"].append(time_drunet_medium)
    # Timer_map["DRUNet"].append(time_drunet_complex)

    # Méthode 3 : CRRNN (avec torch.no_grad() pour éviter le calcul des gradients)

    model_inp_5 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_5, device=device, load=True, checkpoint=False)

    with torch.no_grad():

        time_crrnn_simple = timer(process_image_3, Im_butterfly_masked_simple, operator=denoise, model=model_inp_5, t_steps=200, 
                               operator_type ="mask", operator_params={"Mask": Mask_butterfly_simple}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

    model_inp_25 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_25, device=device, load=True, checkpoint=False)

    with torch.no_grad():

        time_crrnn_medium = timer(process_image_3, Im_butterfly_masked_medium, operator=denoise, model=model_inp_25, t_steps=200, 
                               operator_type ="mask", operator_params={"Mask": Mask_butterfly_medium}, auto_params=False, lmbd=2.e-0 , mu=1e-0, step_size=5e-1, init=False)

    # model_inp_50 = CRRNN(model=modele_0, name_model_pre=checkpoint_dir_inp_50, device=device, load=True, checkpoint=False)

    # with torch.no_grad():

    #     time_crrnn_complex = timer(process_image_3, Im_butterfly_masked_complex, operator=denoise, model=model_5, t_steps=200, 
    #                            operator_type ="mask", operator_params={"Mask": Mask_butterfly_complex}, auto_params=False, lmbd=1e-4, mu=1e-0, step_size=1.9, init=False)

    Timer_map["CRRNN"].append(time_crrnn_simple)
    Timer_map["CRRNN"].append(time_crrnn_medium)
    # Timer_map["CRRNN"].append(time_crrnn_complex)

    # Création d'un DataFrame Pandas
    # Timer_map_df = pd.DataFrame.from_dict(Timer_map, orient="index", columns=methods)
    Timer_map_df = pd.DataFrame.from_dict(Timer_map, orient="index", columns=[f"Sigma_{sigma}" for sigma in sigma_values[:2]])

    # Affichage du tableau formaté
    print("Temps d'exécution pour chaque méthode et niveau de bruit (en secondes) :")
    print(Timer_map_df.head())

    # print(PSNR(Im_butterfly, Im_butterfly_demasked_medium_crrnn, 1.0))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(18, 6), dpi=200)
    # plt.subplot(131)
    # plt.imshow(Im_butterfly_masked_medium)
    # plt.subplot(132)
    # plt.imshow(Mask_butterfly_medium)
    # plt.subplot(133)
    # plt.imshow(Im_butterfly_demasked_medium_crrnn)

if __name__ == "__main__" :

    main()