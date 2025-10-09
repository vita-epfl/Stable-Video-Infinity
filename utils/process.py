import os
import pickle
from PIL import Image
import io
import imageio
import numpy as np
from tqdm import tqdm

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, codec="libx264", ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    data_root = "./data/toy_train/svi-dance/preprocessed"
    for subdir in os.listdir(data_root):
        subdir_path = os.path.join(data_root, subdir)
        frames = []
        with open(os.path.join(subdir_path, "frame_data.pkl"), "rb") as f:
            frame_data = pickle.load(f)
            for k in sorted(frame_data.keys(), key=lambda x: int(x.split(".")[0])):
                img = Image.open(io.BytesIO(frame_data[k]))
                frames.append(img)
        frames[0].save(os.path.join(subdir_path, "image.png"))
        save_video(frames, os.path.join(subdir_path, "video.mp4"), 24, 5)
        frames = []
        with open(os.path.join(subdir_path, "dw_pose_with_foot_wo_face.pkl"), "rb") as f:
            frame_data = pickle.load(f)
            for k in sorted(frame_data.keys(), key=lambda x: int(x.split(".")[0])):
                img = Image.open(io.BytesIO(frame_data[k]))
                frames.append(img)
                print(k, img.size)
        save_video(frames, os.path.join(subdir_path, "pose.mp4"), 24, 5)
