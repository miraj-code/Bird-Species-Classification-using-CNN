# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the model using Convolutional Neural Network
class ConvolutionalNeuralModel(nn.Module):
    # Initialize the model class    
    def __init__(self) -> None:
        super().__init__()
        # Create two convolution layers
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        # Create three fully connected Linear layer
        self.fc1 = nn.Linear(5*5*16, 1200)
        self.fc2 = nn.Linear(1200, 800)
        self.fc3 = nn.Linear(800, 525)
        
    # A function that forward the data through different layers
    def forward(self, X):
        # Create four relu and two max_pool2d layers 
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, 16*5*5)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

# Set the manual seed
torch.manual_seed(41)
# initialize the model
model = ConvolutionalNeuralModel()