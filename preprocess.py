import random
import string
from collections import defaultdict

import docx2txt
import numpy as np
import torch
from PIL import Image as im
from torch.utils.data import DataLoader, TensorDataset

import utils

# + 3 (because of the START, STOP and PADDING)
letter2index = {l: n + 3 for n, l in enumerate(string.ascii_letters)}
index2letter = {n + 3: l for n, l in enumerate(string.ascii_letters)}

# special tokens
START = 0
STOP = 1
PADDING = 2


def convert_files(id, imgs, text):
    """Converts the image files received through request into PIL Images and
    the text file into a string.

    Args:
        id (string): An identifier for the run, used as the path for a temp directory.
        imgs (List[file]): A list of image files to convert.
        text (file): The document file to convert.

    Returns:
        new_imgs (List[Image]): A list of PIL Image objects, to be preprocessed.
        new_text (string): The text file converted to string form.
    """
    new_imgs = []
    for i, img in enumerate(imgs):
        # Current directory to save images in.
        img_path = id + f"/{str(i)}" + ".jpg"
        img.save(img_path)
        new_img = im.open(img_path).convert("RGB")
        new_imgs.append(new_img)

    text_path = id + "/text.docx"
    text.save(text_path)
    new_text = docx2txt.process(text_path)

    return new_imgs, new_text


def strip(s):
    return s.rstrip()


def update_dicts(w, words, d, imgs_per_line, idx):
    """Updates the given dict as well as imgs_per_line and words list.

    Args:
        w (string): Current word.
        words (List[string]): The list of found words.
        d (dict[set[int]]): The dict to be updated.
        imgs_per_line (dict[int]): A dict of ints to store number of images in each line.
        idx (int): An index to the above dicts and list.
    """
    if len(w):
        words.append("".join(w))
        imgs_per_line[idx] += (len(w) - 1) // 10 + 1

    d[idx].add(imgs_per_line[idx])
    imgs_per_line[idx] += 1


def get_words(text):
    """Converts a long string of text into constituent words, and
    produces a dict of indices to put the spaces and indents.
    Each word is counted as one or more images depending on its size.
    Each line is considered as an array of images.
    The spaces and indents dicts have indices to where the spaces and indents would be in the array.
    Each space and indent counts as one image.

    Args:
        text (string): The document to convert in string form.

    Returns:
        words (List[List[string]]): A list of lists of words.
        spaces (dict[set[int]]): A dict of sets where each key in the dict refers to the line
            number and each item in the sets refer to indices of spaces.
        indents (dict[set[int]]): A dict of sets where each key in the dict refers to the line
            number and each item in the sets refer to indices of indents.
        imgs_per_line (dict[int]): A dict of the number of images in each line.
    """
    lines = list(map(strip, text.split("\n")))[::2]

    words = []
    spaces = defaultdict(set)
    indents = defaultdict(set)
    imgs_per_line = defaultdict(int)

    for i, line in enumerate(lines):
        w = []
        for c in line:
            if c == "\t":
                update_dicts(w, words, indents, imgs_per_line, i)
                w = []

            elif c == " ":
                update_dicts(w, words, spaces, imgs_per_line, i)
                w = []

            else:
                w.append(c)

        if len(w):
            words.append("".join(w))
            imgs_per_line[i] += (len(w) - 1) // 10 + 1

    return words, spaces, indents, imgs_per_line


def convert_and_pad(word):
    """Converts the word to a list of tokens padded to read length 12.

    Args:
        word (string): A string of characters of max length 10.

    Returns:
       new_word (List[int]): A list of ints representing the tokens.
    """
    new_word = []
    for w in word:
        if w in letter2index:
            # Converting each character to its token value, ignoring non alphabetic characters
            new_word.append(letter2index[w])
    new_word = [START] + new_word + [STOP]  # START + chars + STOP
    if len(new_word) < 12:  # if too short, pad with PADDING token
        new_word.extend([PADDING] * (12 - len(new_word)))
    return new_word


def preprocess_text(text, max_input_size=10):
    """Converts the each word into a list of tokens, bounded by start and end token.
    Padding tokens added if necessary to reach max_input_size and splitting if the original
    word is too long.

    Args:
        text (string): The document to convert in string form.
        max_input_size (int): The max number of tokens in each input

    Returns:
        (torch.data.utils.DataLoader): A dataloader to the dataset of words converted to tensors
            with batch size 8.
        spaces (dict[set[int]]): A dict of sets where each key in the dict refers to the line
            number and each item in the sets refer to indices of spaces.
        indents (dict[set[int]]): A dict of sets where each key in the dict refers to the line
            number and each item in the sets refer to indices of indents.
        imgs_per_line (dict[int]): A dict of the number of images in each line.
    """
    words, spaces, indents, imgs_per_line = get_words(text)
    new_words = []

    for w in words:
        w_len = len(w)
        while w_len > 0:
            new_words.append(convert_and_pad(w[:max_input_size]))
            w = w[max_input_size:]
            w_len -= max_input_size

    new_words = torch.from_numpy(np.array(new_words))
    dataset = TensorDataset(new_words)

    return DataLoader(dataset, batch_size=8, shuffle=False), spaces, indents, imgs_per_line


def shuffle_and_repeat(imgs):
    """Takes the original list or images, shuffles them and if there are less than 50 images,
    repeats them until we get 50.

    Args:
        imgs (List[np.array]): A list of images as numpy arrays.

    Returns:
        new_imgs (List[np.array]): A list of images as numpy arrays of size 50.
    """
    new_imgs = []
    orig_len = len(imgs)
    idx = orig_len
    shuf = list(range(orig_len))
    while len(new_imgs) < 50:
        if idx == orig_len:
            random.shuffle(shuf)
            idx = 0
        new_imgs.append(imgs[shuf[idx]])
        idx += 1
    return new_imgs


def preprocess_images(imgs) -> torch.tensor:
    """Rescales, resizes and binarizes a batch of images of handwritten words and
    returns it as a tensor. If there are less than 50 images, the original list is shuffled
    and repeated until 50 is reached.

    Args:
        imgs (List[Image]): Original batch of handwritten word image.

    Returns:
        torch.tensor: Preprocessed word image batch, pixels will be in range -1..1.
    """
    new_imgs = []
    for i in imgs:
        i = np.array(i)
        i = utils.resize_and_threshold(i, 0.5, 1)
        i = np.float32(i)
        i = 1 - i
        i = (i - 0.5) / 0.5
        new_imgs.append(i)
    new_imgs = shuffle_and_repeat(new_imgs)
    new_imgs = torch.tensor(new_imgs)
    return new_imgs.unsqueeze(0)
