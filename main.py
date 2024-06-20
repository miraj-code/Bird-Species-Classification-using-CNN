import cv2
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

from model import model

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.image_paths = list(self.data_path.glob('*/*'))
        self.classes = {cls.name: i for i, cls in enumerate(self.data_path.glob('*'))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        resized_image = cv2.resize(image, (155, 155))
        blur_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
        gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            gray_image = self.transform(gray_image)
        label = self.classes[image_path.parent.name]
        return gray_image, label

# Define paths
train_data_path = './dataset/train/'
test_data_path = './dataset/test/'

# Define transformer
transformer = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets
train_dataset = CustomDataset(train_data_path, transform=transformer)
test_dataset = CustomDataset(test_data_path, transform=transformer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def process_batch(model, data_loader, criterion, optimizer=None):
    total_correct = 0
    total_loss = 0.0

    for X, y in data_loader:
        if optimizer:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        total_loss += loss.item() * X.size(0)  # Accumulate total loss
        predicted = torch.max(y_pred.data, 1)[1]
        total_correct += (predicted == y).sum().item()

        if optimizer:
            loss.backward()
            optimizer.step()

    return total_correct, total_loss / len(data_loader.dataset)

epochs = 5
train_correct = []
test_correct = []
train_losses = []
test_losses = []

for epoch in tqdm(range(epochs)):
    trn_cor, trn_loss = process_batch(model, train_loader, criterion, optimizer)
    tst_cor, tst_loss = process_batch(model, test_loader, criterion)

    train_correct.append(trn_cor)
    test_correct.append(tst_cor)
    train_losses.append(trn_loss)
    test_losses.append(tst_loss)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train loss: {trn_loss:.4f}, Train accuracy: {trn_cor / len(train_loader.dataset):.4f}")
    print(f"Test loss: {tst_loss:.4f}, Test accuracy: {tst_cor / len(test_loader.dataset):.4f}")

print('Training and evaluation completed.')