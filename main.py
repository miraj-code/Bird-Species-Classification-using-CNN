import cv2
from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.image_paths = list(self.data_path.glob('*/*'))

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
        return gray_image

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
