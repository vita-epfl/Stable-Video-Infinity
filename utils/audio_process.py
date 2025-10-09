from einops import rearrange
import pyloudnorm as pyln

import librosa
import subprocess
import torch
import os
import numpy as np

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device) 
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array



def cut_audio_embedding(audio_embed, num_frames, audio_start_idx=0, audio_end_idx=81):
        audio_embed = rearrange(audio_embed, "b s d -> s b d")
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        clip_length = num_frames
        audio_end_idx = audio_start_idx + clip_length # NEED CHECK
        center_indices = torch.arange(
            audio_start_idx,
            audio_end_idx,
            1,
        ).unsqueeze(
            1
        ) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embed.shape[0]-1)
        audio_embed = audio_embed[center_indices][None,...]
        audio_window = 5
        vae_scale = 4
        # copy from wanmodel in multitalk
        first_frame_audio_emb_s = audio_embed[:, :1, ...] 
        latter_frame_audio_emb = audio_embed[:, 1:, ...] 
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=vae_scale) 
        middle_index = audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 

        return (first_frame_audio_emb_s.squeeze(0), latter_frame_audio_emb_s.squeeze(0)) # Output dataloader will add batch dim automatically
