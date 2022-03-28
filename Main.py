import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import random
from tqdm import tqdm
from PIL import Image

# batch size = 32 train = true
def load_emoji(batch_size):
    dataset = []
    batch_counter = 0
    batch = []
    for dir in tqdm(os.listdir('./datasets')):
        # print(dir)
        for file in tqdm(os.listdir('./datasets/'+dir)):
            img = Image.open( './datasets/'+dir+'/'+ file)
            # img.convert("RGBA").save('./datasets/'+dir+'/'+ file) #run this once if you are getting shape errors
            img = np.asarray(img).reshape(4, 72,72)
            if batch_counter < batch_size:
                batch.append(img)
                batch_counter += 1
            else:
                dataset.append(np.array(batch))
                batch = []
                batch_counter = 0

    return np.array(dataset)





def main():
    print("Hello World!")
    load_emoji(32)

if __name__ == "__main__":
    main()