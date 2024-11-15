import gc
import hashlib
import os
import shlex
import subprocess
import librosa
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from src.rvc import Config, load_hubert, get_vc, rvc_infer
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RVC_MODELS_DIR = BASE_DIR / 'rvc_models'
OUTPUT_DIR = BASE_DIR / 'song_output'
RVC_other_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/'
RVC_hubert_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/'


def dl_model(link, model_name, dir_name):
    dir_name.mkdir(parents=True, exist_ok=True)
    with requests.get(f'{link}{model_name}', stream=True) as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_models():
    rvc_other_names = ['rmvpe.pt', 'fcpe.pt']
    rvc_hubert_names = ['hubert_base.pt']
    for model in rvc_other_names:
        print(f'Downloading {model}...')
        dl_model(RVC_other_DOWNLOAD_LINK, model, RVC_MODELS_DIR)
    for model in rvc_hubert_names:
        print(f'Downloading {model}...')
        dl_model(RVC_hubert_DOWNLOAD_LINK, model, RVC_MODELS_DIR)
    print('All models downloaded!')


def get_rvc_model(voice_model):
    model_dir = RVC_MODELS_DIR / voice_model
    rvc_model_path = next(model_dir.glob('*.pth'), None)
    rvc_index_path = next(model_dir.glob('*.index'), None)

    if rvc_model_path is None:
        raise FileNotFoundError(f'There is no model file in the {model_dir} directory.')

    return rvc_model_path, rvc_index_path

def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if not isinstance(wave[0], np.ndarray):
        stereo_path = 'Voice_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    return audio_path

def get_hash(filepath):
    file_hash = hashlib.blake2b()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]

def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, f0_min, f0_max):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, RVC_MODELS_DIR / 'hubert_base.pt')
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g,
              filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, f0_min, f0_max)
    
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()
