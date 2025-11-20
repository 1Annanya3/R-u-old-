import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from config import config
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

transform_train = T.Compose([T.Resize((config['img_width'], config['img_height'])),
                             T.RandomHorizontalFlip(p=0.05),
                             T.RandomVerticalFlip(p=0.05),
                             T.RandomRotation(degrees=20),
                             T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                             T.RandomAffine(degrees=10, shear=0.2),
                             T.ToTensor(),
                             T.Normalize(mean=config['mean'], std=config['std'])
                             ])

transform_test = T.Compose([T.Resize((config['img_width'], config['img_height'])),
                            T.ToTensor(),
                            T.Normalize(mean=config['mean'], std=config['std'])
                            ])

class FacialAgeDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_row = self.data.iloc[idx].values #image_name, age

        #pass by reference
        full_dir = root_dir + f"/{one_row[1]:03}" + f"/{one_row[0]}"
        image = Image.open(full_dir).convert('RGB')

        if self.transform:
            image = self.transform(image)

        age = torch.tensor([one_row[1]], dtype=torch.float32)

        return image, age


root_dir = '/Users/leyih/OneDrive/Desktop/8.7.2024/The Years/Y3/Sem 1/MyCNN/face_age'
csv_file_train = './csv_dataset/train_set.csv'
csv_file_test = './csv_dataset/test_set.csv'
csv_file_valid = './csv_dataset/valid_set.csv'

train_set = FacialAgeDataset(root_dir, csv_file_train, transform_train)
valid_set = FacialAgeDataset(root_dir, csv_file_valid, transform_test)
test_set = FacialAgeDataset(root_dir, csv_file_test, transform_test)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=config['eval_batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=config['eval_batch_size'])

images, ages = next(iter(train_loader))
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    img = images[i].permute(1, 2, 0)
    age = ages[i].item()
    axes[i].imshow(img)
    axes[i].set_title(f"Age: {int(age)}")
    axes[i].axis("off")

plt.show()