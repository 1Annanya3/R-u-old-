import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#parameters
# lr = 0.0002
# max_epoch = 8
# z_dim = 100
# g_conv_dim = 64
# d_conv_dim = 64
# log_step = 100
# sample_step = 500
# sample_num = 32
batch_size = 60
image_size = 64

#directory of images
img_dir = Path(r"C:\Users\leyih\OneDrive\Desktop\8.7.2024\The Years\Y3\Sem 1\CS3237\Project\archive")

#Transform pipeline
transform = transforms.Compose(
    [transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#Target encoding, each folder is an age
target_transform = transforms.Compose(
    [transforms.Lambda(lambda x: F.one_hot(torch.tensor(x), 110))])

#getting the dataset
images = torchvision.datasets.ImageFolder(img_dir,transform=transform, target_transform = target_transform)
img_data = datasets.ImageFolder(root=img_dir, transform=transform)
train_size = int(0.8 * len(img_data))  #80% for training
test_size = len(img_data) - train_size  #Remaining 20% for testing
train_dataset, test_dataset = random_split(img_data, [train_size, test_size])

#create DataLoader for training and testing data
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
print(f'Training dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

def imshow(img):
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

dataiter = iter(train_loader)
pics, labels = next(dataiter)
#show images
imshow(torchvision.utils.make_grid(pics))


#after the last convolution layer, flatten output then pass it onto fully connected layer

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN()
