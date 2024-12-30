import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np

# 1. Create Synthetic Dataset
class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(256, 256)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = T.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly generate synthetic images and masks
        img = np.random.rand(*self.image_size, 3)  
        mask = np.random.randint(0, 2, self.image_size)  
        img = self.transform(img).float()
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

# Dataset and Dataloader
train_dataset = SyntheticSegmentationDataset(num_samples=200)
val_dataset = SyntheticSegmentationDataset(num_samples=50)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 2.SAM-Model
class SimpleSAM(nn.Module):
    def __init__(self):
        super(SimpleSAM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)  # Upsample
        x = self.decoder(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleSAM().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. Training Loop
def train_model(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).unsqueeze(1).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Train the Model
train_model(model, train_loader, val_loader, epochs=25)

#save the model 
#torch.save(model, "sam_full_models.pth")