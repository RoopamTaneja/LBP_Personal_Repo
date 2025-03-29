import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImageToStateConverter:
    """
    Converts RGB images to grid-based state representation compatible with the existing CNN
    """
    def __init__(self, map_width, map_height, channel=3):
        """
        Initialize the converter
        
        Args:
            map_width: Width of the map/state grid
            map_height: Height of the map/state grid
            channel: Number of channels in the state representation
        """
        self.map_width = map_width
        self.map_height = map_height
        self.channel = channel
        
        # Preprocessing transforms for the input image
        self.transforms = transforms.Compose([
            transforms.Resize((map_width, map_height)),
            transforms.ToTensor(),
        ])
    
    def image_to_state(self, image_path):
        """
        Convert an RGB image to a state representation
        
        Args:
            image_path: Path to the RGB image
            
        Returns:
            state: State representation compatible with the CNN [width, height, channel]
        """
        # Load the image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            # Convert to tensor [C, H, W]
            image_tensor = self.transforms(image)
        else:
            # If it's already a tensor or numpy array
            if isinstance(image_path, np.ndarray):
                # Convert numpy to tensor
                if image_path.shape[2] == 3:  # [H, W, C]
                    image_path = np.transpose(image_path, (2, 0, 1))  # Convert to [C, H, W]
                image_tensor = torch.from_numpy(image_path).float()
            else:
                image_tensor = image_path
            
            # Ensure correct dimensions
            if image_tensor.shape[1] != self.map_width or image_tensor.shape[2] != self.map_height:
                image_tensor = transforms.Resize((self.map_width, self.map_height))(image_tensor)
        
        # Create the state representation
        state = np.zeros((self.map_width, self.map_height, self.channel), dtype=np.float16)
        
        # Channel 0: Map data (obstacles/walls) - extract from grayscale version of the image
        gray_image = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        gray_image = gray_image.numpy()
        
        # Threshold to detect obstacles (assuming darker pixels are obstacles)
        obstacle_mask = (gray_image < 0.3).astype(np.float16)
        state[:, :, 0] = obstacle_mask
        
        # Channel 1: UAV positions - use a color detection (e.g., red objects might be UAVs)
        # Look for high red, low green/blue values
        red_channel = image_tensor[0].numpy()
        green_channel = image_tensor[1].numpy()
        blue_channel = image_tensor[2].numpy()
        
        uav_mask = ((red_channel > 0.7) & (green_channel < 0.3) & (blue_channel < 0.3)).astype(np.float16)
        state[:, :, 1] = uav_mask
        
        # Channel 2: Access/track information - use another color (e.g., green might be data points)
        # Look for high green, low red/blue values
        data_mask = ((green_channel > 0.7) & (red_channel < 0.3) & (blue_channel < 0.3)).astype(np.float16)
        state[:, :, 2] = data_mask
        
        return state