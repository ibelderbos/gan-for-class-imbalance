import argparse
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import warnings
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.nn.functional as F
warnings.filterwarnings('ignore')
torch.manual_seed(0)

from utils import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--loss_func", type=str, default='wloss', help="loss function: choose between ['wloss','bce']")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--z_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--csv_file", type=str, default='HR_10000_kmeans_6c.csv', help="name of the csv file")
parser.add_argument("--model_name", type=str, default='model_name0', help="name of the model")
parser.add_argument("--img_size", type=int, default=96, help="size of the image")
parser.add_argument("--save_nr", type=int, default=50, help="save the images after this number of epochs")
parser.add_argument("--n_classes", type=int, default=6, help="number of classes in the dataset")
parser.add_argument("--checkpoint_nr", type=int, default=10, help="save model after checkpoint_start + this number of epochs")
parser.add_argument("--checkpoint_start", type=int, default=200, help="start saving the model after this number of epochs")
parser.add_argument("--disc_repeats", type=int, default=1, help="number of times to update disc before updating gen")
parser.add_argument("--gp_weight", type=int, default=10, help="weight of gradient penalty")
parser.add_argument("--penalty_bool", type=bool, default=False, help="boolean whether to add gradient penalty")
parser.add_argument("--clip_bool", type=bool, default=False, help="boolean whether to use weight clipping")
parser.add_argument("--clip_value", type=float, default=0.01, help="parameter value if using weight clipping")
parser.add_argument("--attention_bool", type=bool, default=False, help="boolean whether to use attention (recommended: False)")
args = parser.parse_args()
print(args)


n_epochs = args.n_epochs
batch_size = args.batch_size
loss_func = args.loss_func
lr = args.lr
b1 = args.b1
b2 = args.b2
z_dim = args.z_dim
csv_file = args.csv_file
model_name = args.model_name
img_size = args.img_size
save_nr = args.save_nr
n_classes = args.n_classes
checkpoint_nr = args.checkpoint_nr
checkpoint_start = args.checkpoint_start
disc_repeats = args.disc_repeats
gp_weight = args.gp_weight
penalty_bool = args.penalty_bool
clip_bool = args.clip_bool
clip_value = args.clip_value


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_shape = (3, img_size, img_size)
cuda = True if torch.cuda.is_available() else False

   
# DEFINE THE PATH FOR RESULTS
result_path = 'D:/GAN_results/model_' + model_name
dataset_path = 'D:/Heerlen_HR_2018/Heerlen_HR_2018/Heerlen_HR_2018/full/'
path_to_csv = 'D:/'


# CREATE THE PATHS IN CHOSEN DIRECTORY
loss_plots_path = result_path + '/loss_plots'
gen_imgs_path = result_path + '/gen_imgs'
checkpoints_path = result_path + '/checkpoints'
grad_norms_path = result_path + '/grad_norms'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(loss_plots_path):
    os.makedirs(loss_plots_path)
if not os.path.exists(gen_imgs_path):
    os.makedirs(gen_imgs_path)
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)
if not os.path.exists(grad_norms_path) and loss_func == 'wloss':
    os.makedirs(grad_norms_path)


# GET THE DATASET
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
])

dataset = HeerlenDataset(csv_file= path_to_csv + csv_file,
                                      root_dir=dataset_path,
                                      transform=transform)
                                      
# INITIALIZE GEN AND DISC
gen = Generator(z_dim, n_classes).to(device)
disc = Discriminator(data_shape, n_classes).to(device)

# OPTIMIZERS
opt_G = torch.optim.Adam(gen.parameters(), lr= lr, betas=(b1, b2))
opt_D = torch.optim.Adam(disc.parameters(), lr= lr, betas=(b1, b2))

def weights_init(x):
    if isinstance(x, nn.Conv2d) or isinstance(x, nn.ConvTranspose2d):
        torch.nn.init.normal_(x.weight, 0.0, 0.02)
    if isinstance(x, nn.BatchNorm2d):
        torch.nn.init.normal_(x.weight, 1.0, 0.02)
        torch.nn.init.constant_(x.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


# TRAINING
dataloader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True)

generator_losses = []
discriminator_losses = []
#avg_batch_grad_norms = []
#penalties = []

for epoch in range(n_epochs):
    # i is the batch number
    for i, (real, labels) in enumerate(tqdm(dataloader)):
        real = real.to(device)
        
        # get the one-hot labels for the gen and disc
        one_hot_labels = F.one_hot(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, data_shape[1], data_shape[2])
        
        mean_iteration_disc_loss = 0
        for _ in range(disc_repeats):
            
            # ====================
            # UPDATE DISCRIMINATOR
            # ====================

            # zero out the gradients
            opt_D.zero_grad()

            # get the noise
            fake_noise = get_noise(len(real), z_dim, device=device)

            # concatenate the noise to the one-hot labels    
            noise_and_labels = torch.cat([fake_noise.float(),one_hot_labels.float()], dim=1)

            # generate the fakes
            fake = gen(noise_and_labels)

            # concatenate the images to the labels (make sure to detach the fakes)
            fake_image_and_labels = torch.cat([fake.float(),image_one_hot_labels.float()], dim=1)
            fake_images = fake_image_and_labels.detach()
            real_image_and_labels = torch.cat([real.float(),image_one_hot_labels.float()], dim=1)

            # get the discriminator predictions
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)
            
            if loss_func == 'bce':
                criterion = nn.BCEWithLogitsLoss()
                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                mean_iteration_disc_loss += disc_loss.item() / disc_repeats
            
            if loss_func == 'wloss':
                # gradient penalty & grad norm computation
                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                grad = get_gradient(disc, real, fake.detach(), epsilon)
                penalty, avg_batch_grad_norm = gradient_penalty(grad, gp_weight)

                if penalty_bool:
                    disc_loss = -disc_real_pred.mean() + disc_fake_pred.mean() + penalty
                else:
                    disc_loss = -disc_real_pred.mean() + disc_fake_pred.mean()

                # mean_iteration_disc_loss += disc_loss.item() / disc_repeats
                mean_iteration_disc_loss += disc_loss.item() / disc_repeats
                # avg_batch_grad_norms += [avg_batch_grad_norm.item()]
                # penalties += [penalty.item()]

            # update gradients
            disc_loss.backward(retain_graph=True)

            # update optimizer
            opt_D.step()
            
            if clip_bool:
                for p in disc.parameters():
                    p.data.clamp_(-clip_value, clip_value)


        # ================
        # UPDATE GENERATOR 
        # ================
        
        # zero out the gradients
        opt_G.zero_grad()
        
        # get new noise
        fake_noise_2 = get_noise(len(real), z_dim, device=device)
        noise_and_labels_2 = torch.cat([fake_noise_2.float(),one_hot_labels.float()], dim=1)
        fake_2 = gen(noise_and_labels_2)

        # concatenate the fakes to the one-hot img labels
        
        fake_image_and_labels_2 = torch.cat([fake_2.float(),image_one_hot_labels.float()], dim=1)
        
        # get the predictions for the fakes
        disc_fake_pred = disc(fake_image_and_labels_2)
        
        if loss_func == 'wloss':
            gen_loss = -disc_fake_pred.mean()
        
        if loss_func == 'bce':
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        
        # backpropagation to compute gradients for all layers
        gen_loss.backward()
        
        # update weights for this batch
        opt_G.step()
        
        # current epoch nr * iter per epochs + iterations in current loop 
        batches_per_epoch = len(dataloader)
        total_finished_batches = (epoch + 1) * (batches_per_epoch) + (i + 1)
        
        # save loss in list after each epoch
        if total_finished_batches % batches_per_epoch == 0:
            discriminator_losses += [mean_iteration_disc_loss] 
            generator_losses += [gen_loss.item()]

        # only save gen/loss progress images every save_nr epochs (if 5000 epochs)
        if total_finished_batches % (batches_per_epoch * save_nr) == 0:
            save_mixed_images(epoch, gen_imgs_path, n_classes, img_size, z_dim, gen)
            plot_losses(generator_losses, discriminator_losses, loss_plots_path)
            if loss_func == 'wloss':
                plot_grad_norms(avg_batch_grad_norms, grad_norms_path)
        if (epoch + 1) > checkpoint_start and total_finished_batches % (batches_per_epoch * checkpoint_nr) == 0:
            torch.save({
                'G_state_dict': gen.state_dict(),
                'G_loss': generator_losses,
                'D_loss': discriminator_losses,
                #'grad_norm': avg_batch_grad_norms,
                #'penalties': penalties
            }, checkpoints_path + '/chkpt_epoch%d.pt' % (epoch + 2))

    # print the numbers after each epoch
    print('[Epoch %d/%d] [D loss: %f] [G loss: %f]' % 
    (epoch+1,  n_epochs,
    discriminator_losses[-1], generator_losses[-1]))



