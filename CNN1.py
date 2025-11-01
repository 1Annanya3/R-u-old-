import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
# import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

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
"""
class FaceModel(nn.Module):
    def __init__(self, in_channels):
        # call the parent constructor
        super(FaceModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=90, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
    x = self.l1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return output

model = FaceModel()

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

"""
