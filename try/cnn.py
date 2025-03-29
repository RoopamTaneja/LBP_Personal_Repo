import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels=3):
        """
        CNN implementation in PyTorch that mirrors the TensorFlow implementation in maddpg.py

        Args:
            input_channels: Number of input channels in the state representation
                            (default 3: map data, UAV positions, access information)
        """
        super(CNN, self).__init__()

        # Convolutional layers matching the environment's 80x80 input
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)

        # Batch normalization
        self.bn = nn.BatchNorm2d(64)

        # Pre-calculate output feature dimension for 80x80 input
        self.feature_dim = 64 * 9 * 9 

    def forward(self, x):
        """
        Forward pass through the CNN
        
        Args:
            x: Input tensor with shape [batch_size, height, width, channels] from environment
               or [batch_size, channels, height, width] if already permuted
               
        Returns:
            Flattened feature representation of shape [batch_size, 5184]
        """
        # Handle input format conversion from environment to PyTorch
        if len(x.shape) == 3:  # If input is (height, width, channels) from NumPy
            x = torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
            x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, channels, height, width)
        elif len(x.shape) == 4 and x.shape[3] == 3:  # If input is (batch, height, width, channels) from NumPy
            x = torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
            x = x.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width)

        # Process through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply batch normalization after the final convolution layer (as in TensorFlow version)
        x = self.bn(x)
        
        # Flatten directly to [batch_size, feature_dim]
        x = x.view(x.size(0), self.feature_dim)
        
        return x


class ActorNetwork(nn.Module):
    def __init__(self, input_channels=3, action_dim=5, hidden_dim=600):
        """
        Actor network with CNN feature extractor followed by MLP layers

        Args:
            input_channels: Number of channels in input image
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layer (equivalent to --num-units in your code)
        """
        super(ActorNetwork, self).__init__()

        # CNN feature extractor
        self.cnn = CNN(input_channels)

        # Assuming 80x80 input -> 9x9 feature maps after 3 conv layers
        # Feature dimension will be 64*9*9 = 5184
        # Adjust this calculation based on your actual input dimensions
        feature_dim = self._calculate_conv_output_dim()

        # MLP layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def _calculate_conv_output_dim(self, input_dim=80):
        """Calculate the output dimension after CNN layers"""
        # For each layer with stride 2 and kernel 3, the output size is (input_size - 3)/2 + 1
        dim = input_dim
        for _ in range(3):  # 3 conv layers
            dim = (dim - 3) // 2 + 1
        return 64 * dim * dim  # 64 channels * height * width

    def forward(self, x):
        """Forward pass through the actor network"""
        features = self.cnn(x)
        x = F.relu(self.fc1(features))
        # Final layer without activation (will typically use tanh or softmax depending on your action space)
        actions = self.fc2(x)
        return actions


# Example usage
if __name__ == "__main__":
    # Create a sample input (batch_size=1, channels=3, height=80, width=80)
    sample_input = torch.rand(1, 3, 80, 80)

    # Initialize the CNN
    cnn = CNN()
    features = cnn(sample_input)
    print(f"Feature shape: {features.shape}")

    # Initialize the actor network
    actor = ActorNetwork(action_dim=5)
    actions = actor(sample_input)
    print(f"Action shape: {actions.shape}")
