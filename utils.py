import glob
import os
import random
import string

import cv2
import numpy as np
import torch
import wandb

from config import WAND_API_KEY
from models import GenModel_FC


def get_model(download=False):
    """Downloads the model artifact from wandb and loads the weights from it
    into a new generator object.

    Returns:
        gen (torch.nn.module): The pretrained generator model.
        device (string): The device the model is on (cuda/cpu).
    """
    os.environ["WANDB_API_KEY"] = WAND_API_KEY
    if download:
        api = wandb.Api()
        artifact = api.artifact(
            "bijin/GANwriting_Reproducibilty_Challenge/GANwriting:v237", type="model"
        )
        model_dir = artifact.download() + "/contran-5000.model"
    else:
        model_dir = "./artifacts/GANwriting-v237/contran-5000.model"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weights = torch.load(model_dir, map_location=torch.device("cpu"))
    gen = GenModel_FC(12)
    state_dict = gen.state_dict()

    for key in state_dict.keys():
        state_dict[key] = weights["gen." + key]

    gen.load_state_dict(state_dict)
    gen.eval()
    gen = gen.to(device)
    return gen, device


def get_run_id():
    """REPLACED WITH TEMPFILE.

    Produces a random string of lowercase ascii characters of size 10 to serve as a run id.
    Also creates a dir of the same name in ./temp to store run artifacts temporarily.
    Ensures that the id is not currently in use by checking for existing dir of same name.

    Returns:
        id (string): Identifier string.
    """
    size = 10
    while True:
        id = "".join(random.choice(string.ascii_lowercase) for _ in range(size))
        if not os.path.isdir("./temp/" + id):
            os.mkdir("./temp/" + id)
            break
    return id


def resize_and_threshold(img, thresh, high):
    """Resizes the image to (216, 64) and does Otsu's thresholding on it.

    Args:
        img (np.array[np.uint8]): Image to be processed.
        mid (int|float): Initial threshold for Otsu's method.
        high (int|float): Max value of the image for Otsu's thresholding.

    Returns:
        img (np.array[np.uint8]): The processed image, pixels will be either 0 or high.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Grayscaling the image
    img = cv2.resize(img, (216, 64), interpolation=cv2.INTER_CUBIC)  # resizing the image for VGG
    _, img = cv2.threshold(
        img, thresh, high, cv2.THRESH_OTSU
    )  # thresholding with Otsu's method for binarization
    return img


def denormalize(img):
    """Denormalizes images to the range 0..255.

    Args:
        img (np.array): 3D array of floats.

    Returns:
        img (np.array): 3D array of 8-bit unsigned ints, pixels will be in range 0..255.
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = 1 - img
    img *= 255
    img = np.uint8(img)
    return img


def convert_to_images(gen, text_dataloader, preprocessed_imgs, device):
    """Converts the words from the document to handwritten word images in the
    style of preprocessed_images.

    Args:
        gen (torch.nn.module): The generator model.
        text_dataloader (torch.utils.data.DataLoader): DataLoader object for the words
            from the document.
        preprocessed_imgs (torch.tensor): The handwritting images after preprocessing.
        device (string): The device on which to do the conversion(cuda/cpu).

    Returns:
        imgs (List[np.array]): A list of images as numpy arrays, pixels will be in range 0..255.
    """
    with torch.no_grad():
        style = gen.enc_image(preprocessed_imgs.to(device))
        imgs = []
        for idx, word_batch in enumerate(text_dataloader):
            word_batch = word_batch[0].to(device).long()

            f_xt, f_embed = gen.enc_text(word_batch, style.shape)

            # the size we need the style tensor to be, 0th index is usually batch size
            # but sometimes smaller, rest is unchanged
            size = [word_batch.shape[0]] + list(style.shape[1:])
            f_mix = gen.mix(style.expand(size), f_embed)

            xg = gen.decode(f_mix, f_xt).cpu().detach().numpy()

            for x in xg:
                imgs.append(denormalize(x.squeeze()))

    return imgs


def cleanup_temp_files(id):
    """REPLACED WITH TEMPFILE MODULE.
    Deletes current temp dir including all its files.
    Is actually a ticking timebonb. Will surely break somthing.

    Args:
        id (string): An identifier for the run, used as the name for a temp directory.
    """
    files = glob.glob("./temp/" + id + "/*.*")
    for f in files:
        try:
            os.remove(f)
        except Exception:
            print(f, "failed to delete")

    # Another pass to cleanup pdf files, only deletes unused pdfs on windows
    # Could cause problems with multiple simultaeneous requests
    pdfs = glob.glob("./temp/*/*.pdf")
    for pdf in pdfs:
        try:
            dirname = os.path.dirname(pdf)
            os.remove(pdf)
            os.rmdir(dirname)
        except Exception:
            print(pdf, "failed to delete")


def convert_pic_to_mini_array(image):
    """
    Crop out images of individual handwrittten words from the whole image.

    Args:
        image (PIL.Image): The image file having handwritten text.

    Returns:
        cropped_images (List[np.array]): A list of np.array of imgs of words cropped out from
        the source image.
    """
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # im2 = img.copy()
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = image[y : y + h, x : x + w]
        cropped_images.append(np.array(cropped))

    return cropped_images


def filter_mini_array(images):
    """Filter mini array of tiny images.

    Args:
        images (List[np.array]): List of images to filter.

    Returns:
        List[np.array]: Array of filtered images.
    """
    heights = []
    for image in images:
        heights.append(image.shape[0])
    mean_height = np.mean(np.array(heights))
    filtered_images = []
    for height, image in zip(heights, images):
        if height > 0.6 * mean_height:
            filtered_images.append(image)
    return filtered_images
