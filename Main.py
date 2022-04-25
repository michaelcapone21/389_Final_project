# import time
import torch
import torch.nn as nn
# import torchvision
import numpy as np
# import tensorflow as tf
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split, Subset
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as tt
# from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# from matplotlib.image import imread
import os
import random
from tqdm import tqdm
from PIL import Image

image_size = (4,28,28)

class Discriminator(nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # using these layers is significantly slower

        # self.conv1 = nn.Conv2d(4,input_shape[1] , 4, 2, bias=False)
        # self.batchNorm1 = nn.BatchNorm2d(image_size[1] )

        # self.conv2 = nn.Conv2d(input_shape[1],image_size[1]*2, 3, 2, 1, bias=False)
        # self.batchNorm2 = nn.BatchNorm2d(image_size[1] * 2)

        # self.conv3 = nn.Conv2d(image_size[1]*2,image_size[1]*4, 3,2,1, bias=False)
        # self.batchNorm3 = nn.BatchNorm2d(image_size[1] * 4)

        # self.conv4 = nn.Conv2d(image_size[1]*4, 1, 2,2,1)

        self.conv1 = nn.Conv2d(4,input_shape[0] , 4, 2, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(image_size[0] )

        self.conv2 = nn.Conv2d(input_shape[0],image_size[0]*2, 3, 2, 1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(image_size[0] * 2)

        self.conv3 = nn.Conv2d(image_size[0]*2,image_size[0]*4, 3,2,1, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(image_size[0] * 4)

        self.conv4 = nn.Conv2d(image_size[0]*4, 1, 2,2,1)

        self.lin = nn.Linear(9, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):

        out = self.relu(self.batchNorm1((self.conv1(x))))
        out = self.relu(self.batchNorm2(self.conv2(out)))
        out = self.relu(self.batchNorm3(self.conv3(out)))
        out = self.relu((self.conv4(out))) 

        out = self.flatten(out)
        out = self.lin(out)

        x = nn.Sigmoid()(out)

        return x


class Generator(nn.Module):
    def __init__(self, input_size, output_shape):
        super(Generator, self).__init__()
        self.input_size = input_size
        # using these layers is significantly slower

        # self.conv2dT1 = nn.ConvTranspose2d(input_size,image_size[1] *8, 4, 1 ,0,bias=False)
        # self.batchNorm1 = nn.BatchNorm2d(image_size[1] * 8)

        # self.conv2dT2 = nn.ConvTranspose2d(image_size[1]*8,image_size[1]*4, 3, 2, 1, bias=False)
        # self.batchNorm2 = nn.BatchNorm2d(image_size[1] * 4)

        # self.conv2dT3 = nn.ConvTranspose2d(image_size[1]*4,image_size[1]*2, 3,2,1, bias=False)
        # self.batchNorm3 = nn.BatchNorm2d(image_size[1] * 2)

        # self.conv2dT4 = nn.ConvTranspose2d(image_size[1]*2,4, 4,2,0, bias=False)

        self.conv2dT1 = nn.ConvTranspose2d(input_size,image_size[0] *8, 4, 1 ,0,bias=False)
        self.batchNorm1 = nn.BatchNorm2d(image_size[0] * 8)

        self.conv2dT2 = nn.ConvTranspose2d(image_size[0]*8,image_size[0]*4, 3, 2, 1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(image_size[0] * 4)

        self.conv2dT3 = nn.ConvTranspose2d(image_size[0]*4,image_size[0]*2, 3,2,1, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(image_size[0] * 2)

        self.conv2dT4 = nn.ConvTranspose2d(image_size[0]*2,4, 4,2,0, bias=False)


        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.batchNorm1(self.conv2dT1(x)))
        out = self.relu(self.batchNorm2(self.conv2dT2(out)))
        out = self.relu(self.batchNorm3(self.conv2dT3(out)))
        out = ((self.conv2dT4(out))) 


        out = torch.reshape(out,(-1,4,28,28))


        return nn.Sigmoid()(out)


def training(generator, discriminator, loss, g_optimizer, d_optimizer, train_dataloader, n_epochs, update_interval, noise_samples, path):
    
    g_losses = []
    d_losses = []



    for epoch in range(n_epochs):
        for i, image in enumerate(tqdm(train_dataloader)):

            image = image.float()

            real_classifications = discriminator(image)
            real_labels = torch.ones(image.shape[0])

            noise = torch.from_numpy((np.random.randn(4,noise_samples,1,1) - 0.5) / 0.5).float()  ## This is us sampling gaussian noise

            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)

            fake_labels = torch.zeros(image.shape[0])
     

            classifications = torch.cat((real_classifications, fake_classifications), 0).reshape((len(real_classifications) + len(fake_classifications)))
            targets = torch.cat((real_labels, fake_labels), 0)


            d_optimizer.zero_grad()
            d_loss = loss(classifications,targets)
            d_loss.backward()
            d_optimizer.step()


            if i % update_interval == 0:
                d_losses.append(round(d_loss.item(), 2))
            

            noise = torch.from_numpy((np.random.randn(image.shape[0], noise_samples,1,1) - 0.5) / 0.5).float()
            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)
            fake_labels = torch.zeros(image.shape[0], 1)

            g_optimizer.zero_grad()
            g_loss = -loss(fake_classifications,fake_labels)
            g_loss.backward()
            g_optimizer.step()


            if i % update_interval == 0:
                g_losses.append(round(g_loss.item(), 2))
                torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': g_optimizer.state_dict(),
                'discriminator_state_dict': d_optimizer.state_dict(),
                }, path)
                
    return (generator, discriminator), (g_losses, d_losses) 


def do_training(dataset, ex_image,checkPoint = False):

    lr_g = .0001
    lr_d = .00006
    batch_size = 32        
    update_interval = 64  
    n_epochs = 100
    noise_samples = 64    

    path = "./checkpoints/model.pt"


    loss_function = nn.BCELoss()

    G_model = Generator(noise_samples, image_size)
    D_model = Discriminator(image_size)
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=lr_g)       
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=lr_d)      

    train_dataset = dataset
    
    if checkPoint :
        checkpoint = torch.load(path)
        G_model.load_state_dict(checkpoint['generator_state_dict'])
        D_model.load_state_dict(checkpoint['discriminator_state_dict'])
        G_optimizer.load_state_dict(checkpoint['generator_state_dict'])
        D_optimizer.load_state_dict(checkpoint['discriminator_state_dict'])

        G_model.eval()
        D_model.eval()
        G_model.train()
        G_optimizer.train()

    models, losses = training(G_model, D_model, loss_function, G_optimizer, D_optimizer, train_dataset, n_epochs, update_interval, noise_samples, path)

    G_model, D_model = models
    g_losses, d_losses = losses

    plt.plot(np.arange(len(g_losses)) * batch_size * update_interval, g_losses)
    plt.title("training curve for generator")
    plt.xlabel("number of images trained on")
    plt.ylabel("loss")
    plt.show()

    plt.plot(np.arange(len(d_losses)) * batch_size * update_interval, d_losses)
    plt.title("training curve for discriminator")
    plt.xlabel("number of images trained on")
    plt.ylabel("loss")
    plt.show()

    trained_output = D_model(ex_image.float())

    plot_image(ex_image)
    print("Output of the discriminator given this input:", trained_output[0].detach().numpy()[0])
    plt.show()

    noise = (torch.rand(4, G_model.input_size,1,1) - 0.5) / 0.5

    trained_gen = G_model(noise)

    plot_image(trained_gen.detach())

    trained_output = D_model(trained_gen.float())

    print("Output of the discriminator given this generated input:", trained_output[0].detach().numpy()[0])

    noise = (torch.rand(4, G_model.input_size,1,1) - 0.5) / 0.5

    trained_output = G_model(noise)

    plot_image(trained_output.detach()) 

# batch size = 32 
# batch size controls how many images are placed into a 'batch' -> NP.array
# This function takes all the emojis and loads them into NP array 
def load_emoji(batch_size):
    dataset = []
    batch_counter = 0
    batch = []
    # this will create a directory to store the checkpoints if one does not already exist.
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    for dir in tqdm(os.listdir('./Datasets')):

        for file in tqdm(os.listdir('./Datasets/'+dir)):
            img = Image.open( './Datasets/'+dir+'/'+ file)
            img = np.asarray(img).reshape(image_size)
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
    image = image.reshape(-1,28,28,4)
    plt.imshow(image[0])
    plt.show()
    return

def main():
    dataset = load_emoji(batch_size=4)
    dataset = torch.from_numpy(dataset)
    ex_image = dataset[random.randint(0,276)]
    # do_training(dataset,ex_image,True)
    do_training(dataset,ex_image,False)

    print('l')




if __name__ == "__main__":
    main()