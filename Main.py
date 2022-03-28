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



class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(4,6,(10,10))
        self.conv2 = nn.Conv2d(6,10,(5,5))

        self.lin1 = nn.Linear(34810,200)
        self.lin2 = nn.Linear(200,1)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.flatten(out)

        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)

        x = nn.Sigmoid()(out)

        return x


class Generator(nn.Module):
    def __init__(self, input_size, output_shape):
        super(Generator, self).__init__()
        self.input_size = input_size

        self.lin1 = nn.Linear(input_size,200)
        self.lin2 = nn.Linear(200,np.prod(output_shape))
        self.relu = nn.ReLU()


    def forward(self, x):

        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)

        out = torch.reshape(out,(-1,4,72,72))

        return nn.Sigmoid()(out)

def training(generator, discriminator, loss, g_optimizer, d_optimizer, train_dataloader, n_epochs, update_interval, noise_samples):
    
    g_losses = []
    d_losses = []
    
    for epoch in range(n_epochs):
        for i, image in enumerate(tqdm(train_dataloader)):

            image = image.float()

            real_classifications = discriminator(image)
            real_labels = torch.ones(image.shape[0])



            noise = torch.from_numpy((np.random.randn(image.shape[0], noise_samples) - 0.5) / 0.5).float()  ## This is us sampling gaussian noise
            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)
            fake_labels = torch.zeros(image.shape[0])

            classifications = torch.cat((real_classifications, fake_classifications), 0).reshape(len(real_classifications) + len(fake_classifications))
            targets = torch.cat((real_labels, fake_labels), 0)


            d_optimizer.zero_grad()
            d_loss = loss(classifications,targets)
            d_loss.backward()
            d_optimizer.step()


            if i % update_interval == 0:
                d_losses.append(round(d_loss.item(), 2))
            

            noise = torch.from_numpy((np.random.randn(image.shape[0], noise_samples) - 0.5) / 0.5).float()
            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)
            fake_labels = torch.zeros(image.shape[0], 1)

            g_optimizer.zero_grad()
            g_loss = -loss(fake_classifications,fake_labels)
            g_loss.backward()
            g_optimizer.step()


            if i % update_interval == 0:
                g_losses.append(round(g_loss.item(), 2))
                
    return (generator, discriminator), (g_losses, d_losses) 


# batch size = 32 
# batch size controls how many images are placed into a 'batch' -> NP.array
# This function takes all the emojis and loads them into NP array 
def load_emoji(batch_size):
    dataset = []
    batch_counter = 0
    batch = []
    for dir in tqdm(os.listdir('./datasets')):
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

# plot the image only || important for visualization 
def plot_image(image): 
    image = image.reshape(-1, 72,72,4)
    plt.imshow(image[0])
    plt.show()
    return



def main():
    dataset = load_emoji(batch_size=32)
    dataset = torch.from_numpy(dataset)
    ex_image = dataset[random.randint(0,20)]

    # print("image shape:", ex_image.shape)
    # plot_image(ex_image)

    discriminator = Discriminator((4,72,72))
    ex_output = discriminator(ex_image.float())

    # plot_image(ex_image)
    # this is just for testing discriminator
    # print("Output of the discriminator given this input:", ex_output[0].detach().numpy()[0])  

    # this is for testing generator 
    # test_gen = Generator(128, (4, 72, 72))
    # noise = (torch.rand(1, 128) - 0.5) / 0.5
    # test_output = test_gen(noise)
    # plot_image(test_output.detach().byte())







if __name__ == "__main__":
    main()