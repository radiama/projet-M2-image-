import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
from metrics import psnr

def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    except FileNotFoundError:
        parent_folder = "/".join(folder_path.split("/")[:-1])
        create_folder(parent_folder)
        create_folder(folder_path)


def clip_image(input_img:np.ndarray, method="cut")-> np.ndarray:
    """
    Clips the input image values in the interval [0,1].
    Because a numpy array is a mutable object in python, we must copy it beforehand in order to not modify the
       original. In most cases this is unnecessary and suboptimal in terms of memory, but at least it avoids
       unwanted errors popping up from shared use of memory.
    """
    output = input_img.copy()
    if method in ["cut", "normalize"]:
        if method == "cut":
            output[output > 1] = 1
            output[output < 0] = 0
        elif method == "normalize":
            output = output - np.min(output)
            if np.max(output) > 0:
                output = output / np.max(output)
    else:
        raise ValueError(f"incorrect method choice. Available choices : {['«cut»', '«normalize»']}")
    return output


def get_img(img_path:str)->np.ndarray:
    img = np.array(Image.open(img_path)).astype('float64') / 255
    return img

def get_tensor_img(img_path:str, device=None):
    output_tensor = torch.tensor(get_img(img_path).astype(np.float32))

    if len(output_tensor.shape) == 3:
        output_tensor = torch.permute(output_tensor, (2, 0, 1)).unsqueeze(0)
    elif len(output_tensor.shape) == 2:
        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unrecognised input shape")
    if device is not None:
        output_tensor = output_tensor.to(device)
    return output_tensor


def save_numpy_img(img:np.ndarray, save_path="output_img.png"):
    """
    save_img: saves an image that is a np.ndarray.
    """
    if img.shape[-1] == 3:
        Image.fromarray((clip_image(img) * 255).astype('uint8'), 'RGB').save(save_path)
    else:
        Image.fromarray((clip_image(img) * 255).astype('uint8')).save(save_path, optimize=True, compress_level=0)

def save_tensor_img(input_img:torch.Tensor, save_path="output_img.png"):
    """
    save_tensor_img: saves an image that is a torch.Tensor

    """
    if len(input_img.shape) == 4:
        input_img = input_img.squeeze(0)

    number_channels = input_img.shape[0]
    if number_channels == 3:
        img_numpy = torch.permute(input_img, (1, 2, 0)).detach().cpu().numpy()
    elif number_channels == 1:
        img_numpy = input_img.squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unrecognized number of channels: {number_channels} is neither 1 or 3.")

    save_numpy_img(img_numpy, save_path)

def save_img(img, save_path="output_img.png"):
    if isinstance(img, np.ndarray):
        save_numpy_img(img, save_path)
    elif isinstance(img, torch.Tensor):
        save_tensor_img(img, save_path)
    else:
        raise ValueError(f"Unrecognized image type, {type(img)} is neither torch.Tensor or numpy.ndarray")

def add_psnr_to_tensor(input_tensor_img, ref_tensor_img):
    """
    Adds the PSNR with respect to ref_tenosr_img to the input_tensor_img.
    Args:
        input_tensor_img:
        ref_tensor_img:

    Returns: a tensor image with the psnr indicated

    """
    cmu_font = ImageFont.truetype("../cmunorm.ttf", 24)
    psnr_ = psnr(input_tensor_img, ref_tensor_img)

    x1 = input_tensor_img[0, 0].detach().cpu().numpy()
    if x1.shape[0] < 256 or x1.shape[1] < 256:
        output_tensor_img = zoom(x1, 2, order=0, mode="constant")
        output_tensor_img = torch.tensor(output_tensor_img, dtype=input_tensor_img.dtype, device=input_tensor_img.device)
        output_tensor_img = output_tensor_img.unsqueeze(0).unsqueeze(0)
    else:
        output_tensor_img = input_tensor_img.detach().clone().cpu()
    n, m = x1.shape

    out = Image.fromarray(np.zeros((n,m)).astype("uint8"))
    drawer = ImageDraw.Draw(out)
    drawer.text((0, 230), f"PSNR: {psnr_:.2f}", font=cmu_font, fill=(255,))
    text = np.array(out).astype(np.float32) / 255
    text = torch.tensor(text, dtype=output_tensor_img.dtype, device=output_tensor_img.device)
    text = text.unsqueeze(0).unsqueeze(0)

    output_tensor_img[..., 230:, :136] = 1.0
    output_tensor_img -= text

    return output_tensor_img


