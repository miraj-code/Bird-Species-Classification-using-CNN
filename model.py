import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the model using Convolutional Neural Network
class ConvolutionalNeuralModel(nn.Module):
    # Initialize the model class 
    def __init__(self) -> None:
        super().__init__()
        # Create three convolution layers with increasing filter size
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 16 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 32 filters, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, 3, 1) # 64 filters, 3x3 kernel

        # Batch normalization after each convolutional layer
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Create three max pooling layers 
        self.pool = nn.MaxPool2d(2, 2)

        # Create three fully connected layers with Dropout
        self.fc1 = nn.Linear(17 * 17 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 525)

        # Dropout layers with 20% probability
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    # A function that forwards the data through different layers
    def forward(self, X):
        # Apply ReLU activation, BatchNorm, and Max Pooling after each convolution
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.pool(X)
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.pool(X)
        X = F.relu(self.bn3(self.conv3(X)))
        X = self.pool(X)

        # Flatten the output of convolutional layers
        X = X.view(X.size(0), -1)  # Ensure batch size is maintained correctly

        # Apply ReLU activation and Dropout after each fully-connected layer
        X = F.relu(self.dropout1(self.fc1(X)))
        X = F.relu(self.dropout2(self.fc2(X)))
        X = self.fc3(X)

        # Use log_softmax for output probability distribution
        return F.log_softmax(X, dim=1)

# Set the manual seed
torch.manual_seed(41)
# initialize the model
model = ConvolutionalNeuralModel()
