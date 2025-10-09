import torch
import numpy as np
import imageio
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def save_dwpose_as_mp4(dwpose_data, output_path, fps=30, quality=9, normalize=True, ffmpeg_params=None):
    """
    Save dwpose tensor data as MP4 video for debugging using imageio.
    
    Args:
        dwpose_data: torch.Tensor of shape [T, C, H, W]
        output_path: str, output MP4 file path
        fps: int, frames per second (default=30)
        quality: int, video quality 0-10 (default=9)
        normalize: bool, whether to normalize data to 0-255 range
        ffmpeg_params: dict, additional ffmpeg parameters
    """
    if type(dwpose_data) == np.ndarray:
        dwpose_data = torch.from_numpy(dwpose_data)
    if dwpose_data.shape[0] == 3:
        dwpose_data = dwpose_data.permute(1, 0, 2, 3)  # Convert from [C, T, H, W] to [T, C, H, W]

    T, C, H, W = dwpose_data.shape
    
    # Convert to numpy and move to CPU
    data = dwpose_data.detach().cpu().numpy()
    
    # Handle different channel numbers
    if C == 1:
        # Grayscale - repeat to 3 channels
        data = np.repeat(data, 3, axis=1)
    elif C == 3:
        # RGB - use as is
        pass
    else:
        # Multi-channel - take first 3 channels or average
        if C > 3:            
            data = data[:, :3, :, :]
        else:
            # Pad to 3 channels
            padding = np.zeros((T, 3-C, H, W))
            data = np.concatenate([data, padding], axis=1)
    
    # Normalize to 0-255 range
    if normalize:
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    data = (data * 255).astype(np.uint8)
    
    # Transpose from [T, C, H, W] to [T, H, W, C] for imageio
    frames = data.transpose(0, 2, 3, 1)
    
    # Save using imageio
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(frame)
    writer.close()
    
    print(f"Video saved to: {output_path}")



def image_compose_width(imag, imag_1):
    # read the size of image1
    rom_image = imag
    width, height = imag.size
    # read the size of image2
    rom_image_1 = imag_1
    
    width1 = rom_image_1.size[0]
    # create a new image
    to_image = Image.new('RGB', (width+width1, height))
    # paste old images
    to_image.paste(rom_image, (0, 0))
    to_image.paste(rom_image_1, (width, 0))
    return to_image
