import torch
from deepinv.utils import load_url_image
from Begin_Func import load_img, save_path, operateur, numpy_to_tensor, tensor_to_numpy, PSNR, search_opt
from Proxy_Func import fista, prox_l6
import pandas as pd

def main():

    # # '''Chargement des images'''

    Im_butterfly, Im_leaves, Im_starfish = load_img("butterfly.png"), load_img("leaves.png"), load_img("starfish.png")

    
    # # '''Test Bruitage'''

    Im_butterfly_noised, Im_leaves_noised, Im_starfish_noised = operateur(Im_butterfly).noise(sigma=0.2), operateur(Im_leaves).noise(sigma=0.2), operateur(Im_starfish).noise(sigma=0.2)

    Im_butterfly_restored = fista(Im_butterfly_noised, "none", None, 0.01, 0.05, 10, prox=prox_l6, prox_params={"tau": 0.1, "K": 15}, tol=1e-7)

    Im_starfish_restored = fista(Im_starfish_noised, "none", None, 0.008, 0.01, 10, prox=prox_l6, prox_params={"tau": 0.01, "K": 5}, tol=1e-7)

    Im_leaves_restored = fista(Im_leaves_noised, "none", None, 0.01, 0.1, 25, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)
    
    save_path("results", "Im_starfish_noised.png", Im_starfish_noised, True, Im_starfish)

    save_path("results", "Im_leaves_noised.png", Im_leaves_noised, True, Im_leaves)

    save_path("results", "Im_butterfly_noised.png", Im_butterfly_noised, True, Im_butterfly)

    save_path("results", "Im_starfish_restored.png", Im_starfish_restored, True, Im_starfish)

    save_path("results", "Im_leaves_restored.png", Im_leaves_restored, True, Im_leaves)

    save_path("results", "Im_butterfly_restored.png", Im_butterfly_restored, True, Im_butterfly)

    # print(PSNR(Im_leaves, Im_leaves_restored, 1.0))


    # # '''Test Floutage'''
    
    Im_butterfly_blurred, G_butterfly = operateur(Im_butterfly).blur(sigma = (3, 3), angle = 45) 
    
    Im_leaves_blurred, G_leaves = operateur(Im_leaves).blur(sigma = (3, 3), angle = 45)
    
    Im_starfish_blurred, G_starfish = operateur(Im_starfish).blur(sigma = (3, 3), angle = 45)

    Im_butterfly_deblurred = fista(Im_butterfly_blurred, "convolution", {"G": G_butterfly}, 0.0001, 1, 25, prox=prox_l6, prox_params={"tau": 0.0001, "K": 25}, tol=1e-7)

    Im_leaves_deblurred = fista(Im_leaves_blurred, "convolution", {"G": G_leaves}, 0.0001, 1, 25, prox=prox_l6, prox_params={"tau": 0.0001, "K": 25}, tol=1e-7)

    Im_starfish_deblurred = fista(Im_starfish_blurred, "convolution", {"G": G_starfish}, 0.0001, 1, 25, prox=prox_l6, prox_params={"tau": 0.0001, "K": 25}, tol=1e-7)
    
    save_path("results", "Im_starfish_blurred.png", Im_starfish_blurred, True, Im_starfish)

    save_path("results", "Im_leaves_blurred.png", Im_leaves_blurred, True, Im_leaves)

    save_path("results", "Im_butterfly_blurred.png", Im_butterfly_blurred, True, Im_butterfly)

    save_path("results", "Im_starfish_deblurred.png", Im_starfish_deblurred, True, Im_starfish)

    save_path("results", "Im_leaves_deblurred.png", Im_leaves_deblurred, True, Im_leaves)

    save_path("results", "Im_butterfly_deblurred.png", Im_butterfly_deblurred, True, Im_butterfly)
    
    # print(PSNR(Im_starfish, Im_starfish_deblurred, 1.0))

   
    # # '''Test Inpainting'''

    mask = torch.ones(1, 1, 256, 256)

    mask[:, :, 0::24, :] = 0

    mask[:, :, :, 0::24] = 0

    # # mask[:, :, 64:-64, 64:-64] = 0

    mask = mask.to(torch.bool) # mask.type(torch.uint8)

    Im_starfish_masked, Mask_star = operateur(Im_starfish).inpaint(mask = mask , sigma=.08)

    Im_leaves_masked, Mask_leaves = operateur(Im_leaves).inpaint(mask = mask , sigma=.08)

    Im_butterfly_masked, Mask_butterfly = operateur(Im_butterfly).inpaint(mask = mask , sigma=.08)

    Im_butterfly_demasked = fista(Im_butterfly_masked, "mask", {"Mask": Mask_butterfly }, 0.01, 0.25, 25, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    Im_leaves_demasked = fista(Im_leaves_masked, "mask", {"Mask": Mask_leaves }, 0.01, 0.25, 25, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    Im_starfish_demasked = fista(Im_starfish_masked, "mask", {"Mask": Mask_star }, 0.01, 0.25, 25, prox=prox_l6, prox_params={"tau": 0.1, "K": 5}, tol=1e-7)

    save_path("results", "Im_starfish_masked.png", Im_starfish_masked, True, Im_starfish)

    save_path("results", "Im_leaves_masked.png", Im_leaves_masked, True, Im_leaves)

    save_path("results", "Im_butterfly_masked.png", Im_butterfly_masked, True, Im_butterfly)

    save_path("results", "Im_starfish_demasked.png", Im_starfish_demasked, True, Im_starfish)

    save_path("results", "Im_leaves_demasked.png", Im_leaves_demasked, True, Im_leaves)

    save_path("results", "Im_butterfly_demasked.png", Im_butterfly_demasked, True, Im_butterfly)

    # print(PSNR(Im_leaves, Im_leaves_demasked, 1.0))

   
    # # '''Recherche paramètres optimaux'''
    
    # param_ranges = { "K": [5, 10, 25], "lambd": [0.01, 0.1, 0.5], "tau": [0.01, 0.25, 0.5] }

    # prox_params_ranges = {"tau": [0.01, 0.1, 0.5],  "K": [5, 10, 15]}

    # func_params = {"u": Im_leaves_noised, "operator_type": "none", "operator_params": {}, "prox": prox_l6, "tol": 1e-7}
    # func_params = {"u": Im_leaves_masked, "operator_type": "mask", "operator_params": {"Mask": Mask_leaves }, "prox": prox_l6, "tol": 1e-7}
    # func_params = {"u": Im_starfish_blurred, "operator_type": "convolution", "operator_params": {"G": G_starfish }, "prox": prox_l6, "tol": 1e-7}

    # best_params, best_prox_params, best_score, score_map = search_opt(func=fista, u_truth= Im_starfish, param_ranges=param_ranges, 
    # metric=PSNR, func_params=func_params, prox_params_ranges=prox_params_ranges)

    # # Résultats
    # print("Meilleurs paramètres globaux:", best_params)
    # print("Meilleurs paramètres du prox:", best_prox_params)
    # print("Meilleur score:", best_score)
    # print(score_map.head())


    # # PNP(Plug and Play Algorithms)


    # url = ("https://huggingface.co/datasets/deepinv/images/resolve/main/cameraman.png?download=true")
    # x = load_url_image(url=url, img_size=512, grayscale=True, device='cpu')

    # physics = dinv.physics.Inpainting((1, 512, 512), mask = 0.5, \
    #                                     noise_model=dinv.physics.GaussianNoise(sigma=0.01))

    # data_fidelity = dinv.optim.data_fidelity.L2()
    # prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
    # model = dinv.optim.optim_builder(iteration="HQS", prior=prior, data_fidelity=data_fidelity, \
    #                                 params_algo={"stepsize": 1.0, "g_param": 0.1})
    # y = physics(x)
    # x_hat = model(y, physics)
    # dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')


if __name__ == "__main__" :

    main()