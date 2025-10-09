# Video and Audio Processing Script
# Processes video frames and extracts audio embeddings for training

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import pickle
import logging
import sys
import pyloudnorm as pyln
from transformers import Wav2Vec2FeatureExtractor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.src.audio_analysis.wav2vec2 import Wav2Vec2Model
import librosa
import soundfile as sf


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def extract_audio_from_video(video_path, sr=16000):
    """Extract audio from video using ffmpeg -> numpy"""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmpname = tmp.name
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-loglevel",
        "error",
        tmpname,
    ]
    subprocess.run(cmd, check=True)
    audio, _ = sf.read(tmpname)
    os.remove(tmpname)
    return loudness_norm(audio, sr)

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        return extract_audio_from_video(audio_path, sample_rate)
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    return loudness_norm(audio, sample_rate)

def custom_init(device, wav2vec_dir):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir, local_files_only=True
    )
    acoustic_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_dir, local_files_only=True
    ).to(device)
    acoustic_encoder.feature_extractor._freeze_parameters()
    return feature_extractor, acoustic_encoder

def get_embedding(
    speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"
):
    audio_dur = len(speech_array) / sr
    video_len = int(audio_dur * 25)  # Assume video 25 fps

    inputs = wav2vec_feature_extractor(
        speech_array, sampling_rate=sr, return_tensors="pt"
    ).input_values.to(device)
    with torch.no_grad():
        outputs = audio_encoder(inputs, seq_len=video_len, output_hidden_states=True)

    if not outputs.hidden_states:
        logger.error("Fail to extract audio embedding")
        return None
    emb = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)  # (T, C)
    emb = emb.cpu()
    return emb


def get_audio_embedding(
    audio_path,
    num_frames,
    wav2vec_feature_extractor,
    audio_encoder,
    audio_start_idx=0,
    device="cpu",
):
    audio_embed = get_embedding(
        audio_prepare_single(audio_path), wav2vec_feature_extractor, audio_encoder, device=device
    )  
    return audio_embed

def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger

def process_video_pair(
    video_name,
    rgb_video_dir,
    audio_dir,
    output_dir,
    wav2vec_feature_extractor,
    audio_encoder,
    device="cpu",
):

    audio_path = os.path.join(audio_dir, video_name.replace(".mp4", ".wav"))
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return

    rgb_video_path = os.path.join(rgb_video_dir, video_name)

    if not os.path.exists(rgb_video_path):
        logger.warning(f"RGB video not found: {rgb_video_path}")
        return

    video_name_no_ext = video_name.split('.mp4')[0]
    output_folder = os.path.join(output_dir, video_name_no_ext)
    os.makedirs(output_folder, exist_ok=True)
    
    rgb_frames = []
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    while rgb_cap.isOpened():
        ret, frame = rgb_cap.read()
        if ret:
            rgb_frames.append(frame)
        else:
            break
    rgb_cap.release()
    
    if len(rgb_frames) >= 20000:
        logger.warning(f"Video {video_name} has too many frames ({len(rgb_frames)}), skipping...")
        return

    audio_embed = get_audio_embedding(
        audio_path,
        num_frames=len(rgb_frames),
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        audio_encoder=audio_encoder,
        device=device,
    )

    video_frame_all = {}
    
    for i_index, rgb_frame in enumerate(rgb_frames):
        frame_name = str(i_index).zfill(6) + ".jpg"
        
        if rgb_frame.shape[1] > rgb_frame.shape[0]:
            margin = (rgb_frame.shape[1] - rgb_frame.shape[0]) // 2
            rgb_frame = rgb_frame[:, margin:-margin]
        
        frame_h, frame_w, _ = rgb_frame.shape
        if frame_w >= 2048:
            rgb_frame = cv2.resize(rgb_frame, (frame_w//2, frame_h//2))
        
        _, img_encode = cv2.imencode('.jpg', rgb_frame)
        img_bytes = img_encode.tobytes()
        video_frame_all[frame_name] = img_bytes
    
    # Save processed data
    with open(os.path.join(output_folder, 'frame_data.pkl'), "wb") as tf:
        pickle.dump(video_frame_all, tf)
    
    with open(os.path.join(output_folder, "audio_embedding.pkl"), "wb") as f:
        pickle.dump(audio_embed, f)


logger = get_logger('video processing')


if __name__ == '__main__':

    wav2vec_dir = './weights/chinese-wav2vec2-base'
    wav2vec_fe, wav2vec_enc = custom_init('cpu', wav2vec_dir)
   
    rgb_video_dir  = "./data/toy_train/svi-talk/raw/videos/"
    audio_dir      = "./data/toy_train/svi-talk/raw/audios/"
    output_dir     = "./data/toy_train/svi-talk/preprocessed/"

    if not all(map(os.path.exists, [rgb_video_dir, audio_dir])):
        logger.error("Some input directories are missing, abort.")
        sys.exit(1)

    wav2vec_feature_extractor, audio_encoder = custom_init('cpu', wav2vec_dir)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting video processing...")
    videos = sorted([f for f in os.listdir(rgb_video_dir) if f.endswith(".mp4")])
    logger.info(f"Found {len(videos)} videos")

    for i, v in enumerate(videos, 1):
        logger.info(f"[{i}/{len(videos)}] {v}")
        process_video_pair(
            v,
            rgb_video_dir,
            audio_dir,
            output_dir,
            wav2vec_feature_extractor,
            audio_encoder,
        )

    logger.info("Processing completed!")