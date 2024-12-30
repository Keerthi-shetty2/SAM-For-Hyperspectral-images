from ultralytics import SAM
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

IMG_DIR = "./training/img"
ANN_DIR = "./training/ann"
MODEL_SAVE_PATH = "./sam_model_trained.pt"


EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_paths = list(Path(img_dir).glob("*.jpg"))
        self.ann_paths = list(Path(ann_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]


        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)


        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  
    transforms.ToTensor()
])


train_dataset = SegmentationDataset(IMG_DIR, ANN_DIR, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


sam_model = SAM('vit_h').to(DEVICE)  

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(sam_model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_model(model, dataloader, epochs, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model(sam_model, train_loader, EPOCHS, criterion, optimizer)
