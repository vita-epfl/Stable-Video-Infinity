
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import label, find_objects
import torch
import os

def to_normalized(x):
    return x / 255.0 * 2.0 - 1.0

def from_normalized(x):
    return (x + 1.0) / 2.0 * 255.0

def tensor2pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if not torch.is_tensor(x):
        raise TypeError(f"Expect tensor or PIL, got {type(x)}")

    # C,H,W -> H,W,C
    if x.shape[0] in (1, 3) and x.dim() == 3:
        x = x.permute(1, 2, 0)

    # [0,1] float → [0,255] uint8
    arr = x.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) * 255
        arr = arr.astype(np.uint8)

    return Image.fromarray(arr)


def calculate_dimensions(image_input, max_width=640):
    """
    Calculate dimensions for image resizing
    
    Args:
        image_input: Can be either a file path (str) or PIL Image object
        max_width: Maximum width for the output
    
    Returns:
        tuple: (height, width) rounded to multiples of 16
    """
    # Check if input is already a PIL Image or a file path
    if isinstance(image_input, Image.Image):
        img = image_input
    else:
        # Assume it's a file path
        img = Image.open(image_input)
    
    original_width, original_height = img.size
    
    if original_width <= max_width:
        width = original_width
        height = original_height
    else:
        aspect_ratio = original_height / original_width
        width = max_width
        height = int(width * aspect_ratio)
    
    width = (width // 16) * 16
    height = (height // 16) * 16
    
    return height, width


def save_tensor_as_image(tensor, save_path, auto_detect_channels=True):
    """
    Save a PyTorch tensor as an image file with automatic channel dimension detection.
    
    Args:
        tensor (torch.Tensor): Input tensor to save as image
        save_path (str): Path where the image will be saved
        auto_detect_channels (bool): Whether to automatically detect channel dimension position
    
    Returns:
        None
    """
    # Convert tensor to numpy array
    mask_np = tensor.detach().cpu().numpy()
    
    if auto_detect_channels:
        # Automatically detect channel dimension position
        # Channel count is usually 1, 3, or 4, and typically the smallest dimension
        shape = mask_np.shape
        channel_candidates = [i for i, dim in enumerate(shape) if dim in [1, 3, 4]]
        
        if len(channel_candidates) == 1:
            channel_dim = channel_candidates[0]
        elif len(channel_candidates) > 1:
            # If multiple candidates, choose the one with smallest size
            channel_dim = min(channel_candidates, key=lambda x: shape[x])
        else:
            # If no obvious channel candidates, assume smallest dimension is channel
            channel_dim = shape.index(min(shape))
        
        print(f"Detected shape: {shape}")
        print(f"Channel dimension position: {channel_dim} (size: {shape[channel_dim]})")
        
        # Move channel dimension to last position [H, W, C]
        if channel_dim == 0:  # [C, H, W] -> [H, W, C]
            mask_np = np.transpose(mask_np, (1, 2, 0))
        elif channel_dim == 1:  # [H, C, W] -> [H, W, C]
            mask_np = np.transpose(mask_np, (0, 2, 1))
        elif channel_dim == 2:  # [H, W, C] - already correct format
            pass
        else:
            print("Warning: Cannot handle arrays with more than 3 dimensions")
            return
    
    # Normalize to [0, 255] and convert to uint8
    if mask_np.max() != mask_np.min():  # Avoid division by zero
        mask_np = ((mask_np - mask_np.min()) / (mask_np.max() - mask_np.min()) * 255).astype(np.uint8)
    else:
        mask_np = (mask_np * 255).astype(np.uint8)
    
    # Handle single channel case - PIL expects 2D array for grayscale
    if len(mask_np.shape) == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np.squeeze(-1)
    
    # Save image
    Image.fromarray(mask_np).save(save_path)
    print(f"Image saved successfully to: {save_path}")



# Resize and pad a tensor to a target size while maintaining aspect ratio
def resize_and_pad_to_target(tensor, target_size, pad_value=0):

    target_h, target_w = target_size
    batch_size, channels, current_h, current_w = tensor.shape
    
    # Calculate scale factor to maintain aspect ratio
    scale_factor = min(target_h / current_h, target_w / current_w)
    
    # Calculate new dimensions after scaling
    new_h = int(current_h * scale_factor)
    new_w = int(current_w * scale_factor)
    
    # Resize with bilinear interpolation
    resized_tensor = F.interpolate(
        tensor, 
        size=(new_h, new_w), 
        mode='nearest',  # Use 'nearest' to avoid introducing artifacts
    )
    # Calculate padding dimensions
    total_pad_h = target_h - new_h
    total_pad_w = target_w - new_w
    
    # Distribute padding evenly (center the resized tensor)
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left
    

    padded_tensor = F.pad(
        resized_tensor, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', 
        value=pad_value
    )
    
    return padded_tensor


def find_reference_image(ref_image_root):
    """
    Automatically find reference image file, supporting jpg and png formats
    Priority: frame.jpg > frame.png > first jpg file > first png file
    """
    # First check for default frame.jpg
    jpg_path = os.path.join(ref_image_root, "frame.jpg")
    if os.path.exists(jpg_path):
        return jpg_path
    
    # Check for frame.png
    png_path = os.path.join(ref_image_root, "frame.png")
    if os.path.exists(png_path):
        return png_path
    
    # Find all image files in directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # First look for jpg format files
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        for file in os.listdir(ref_image_root):
            if file.endswith(ext):
                return os.path.join(ref_image_root, file)
    
    # Then look for png format files
    for ext in ['.png', '.PNG']:
        for file in os.listdir(ref_image_root):
            if file.endswith(ext):
                return os.path.join(ref_image_root, file)
    
    # If none found, raise error
    raise FileNotFoundError(f"No reference image (jpg/png) found in {ref_image_root}")
