import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torch.nn.functional as F
warnings.filterwarnings('ignore')
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HeerlenDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
    
    
def show_image(img):
    
    img = (img + 1) / 2 
    img = img.detach().cpu()
    plt.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()

    
def get_noise(n_samples, z_dim, device='cpu'):

    return torch.randn(n_samples, z_dim, device=device)

    
def save_mixed_images(epoch, gen_imgs_path, n_classes, img_size, z_dim, gen):
    
    nr_images = n_classes*n_classes
    imgs_per_class = round(nr_images/n_classes)
    nr_images = imgs_per_class * n_classes
    
    generated_imgs = torch.empty(n_classes,imgs_per_class,3,img_size,img_size)
    for class_nr in range(n_classes):
        label_shape = torch.empty(imgs_per_class)
        labels = label_shape.fill_(class_nr).to(torch.int64)
        one_hot_labels = F.one_hot(labels.to(device),n_classes)
        noise = get_noise(imgs_per_class, z_dim, device=device)
        noise_and_labels = torch.cat([noise.float(),one_hot_labels.float()],dim=1)
        gen_imgs = gen(noise_and_labels)
        generated_imgs[class_nr] = gen_imgs
    generated_imgs = generated_imgs.view(-1,3,img_size,img_size)
    
    nrow = n_classes
    epochs_finished = epoch + 2
    save_image(tensor=generated_imgs.data,
               fp=gen_imgs_path + '/images_epoch%d.png' % (epochs_finished),
              normalize=True,
               nrow=nrow) 
    

def plot_losses(generator_losses, discriminator_losses, loss_plots_path, epoch):
    fig = plt.figure()
    epochs_finished = epoch + 1
    plt.plot(generator_losses[-100:], label='Generator loss')
    plt.plot(discriminator_losses[-100:], label='Discriminator loss')
    plt.title('Losses')
    plt.xlabel('Last 100 epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plots_path + '/losses_epoch%d.png' % epochs_finished)
    plt.close(fig)


def plot_grad_norms(avg_batch_grad_norms, grad_norms_path):
    fig = plt.figure()
    epochs_finished = epoch + 1
    plt.plot(avg_batch_grad_norms, label='Average gradient norm of batch per iteration')
    plt.title('Average gradient norm of a batch per iteration')
    plt.xlabel('Last 1000 iterations')
    plt.ylabel('Loss')
    plt.ylim([-1,1])
    plt.legend()
    plt.savefig(grad_norms_path + '/losses_epoch%d.png' % epochs_finished)
    plt.close(fig)
    
    
def get_gradient(disc, real, fake, eps):
    mixed_images = real * eps + fake * (1-eps)
    mixed_images_and_labels = torch.cat([mixed_images.float(),image_one_hot_labels.float()], dim=1)
    mixed_scores = disc(mixed_images_and_labels)
    grad = torch.autograd.grad(
        inputs = mixed_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True
    )[0]
    return grad


def gradient_penalty(grad):
    grad = grad.view(len(grad), -1)
    avg_batch_grad_norm = (grad.norm(2, dim=1)).mean()
    penalty = ((avg_batch_grad_norm-1)**2) * gp_weight
    return penalty, avg_batch_grad_norm