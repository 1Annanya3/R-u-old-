import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class FacialAgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root dataset folder (containing age folders)
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Go through each age folder (001 to 110)
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue  # skip non-folder files

            try:
                age = int(folder_name)
            except ValueError:
                continue  # skip invalid folder names

            # Collect all images in the folder
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(age)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        age = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

dataset = FacialAgeDataset("archive/face_age", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

images, ages = next(iter(train_loader))
# Undo normalization for visualization
images = images * 0.5 + 0.5

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    img = images[i].permute(1, 2, 0)
    age = ages[i].item()
    axes[i].imshow(img)
    axes[i].set_title(f"Age: {int(age)}")
    axes[i].axis("off")
plt.show()

class AgeCNN(nn.Module):
    def __init__(self):
        super(AgeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeCNN().to(device)

criterion = nn.MSELoss()  # since age is a continuous value
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 14
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, ages in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, ages = images.to(device), ages.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    torch.save(save_state, 'checkpoint.pth')

model.eval()
predictions, actuals = [], []
torch.save(model.state_dict(), "age_cnn.pt")

with torch.no_grad():
    for images, ages in test_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images).squeeze()
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(ages.cpu().numpy())

# Compute mean absolute error
mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)
print(f"Mean Absolute Error: {mae:.2f} years")

plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
