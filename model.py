# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the model using Convolutional Neural Network
class ConvolutionalNeuralModel(nn.Module):
    # Initialize the class    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)