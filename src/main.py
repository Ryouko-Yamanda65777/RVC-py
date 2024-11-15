import os
import gc
import shlex
import hashlib
import subprocess
import librosa
import torch
import numpy as np
import gradio as gr
from pathlib import Path
from src.rvc import Config, load_hubert, get_vc, rvc_infer
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RVC_MODELS_DIR = BASE_DIR / 'rvc_models'
MODELS_DIR = BASE_DIR 
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
        dl_model(RVC_other_DOWNLOAD_LINK, model, MODELS_DIR)
    for model in rvc_hubert_names:
        print(f'Downloading {model}...')
        dl_model(RVC_hubert_DOWNLOAD_LINK, model, MODELS_DIR)
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

def song_cover_pipeline(uploaded_file, voice_model, pitch_change, index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, f0_method='rmvpe',
                        crepe_hop_length=128, protect=0.33, output_format='mp3',, f0_min=50, f0_max=1100):

    if not uploaded_file or not voice_model:
        raise ValueError('Make sure that the song input field and voice model field are filled in.')

    print('[~] Starting the AI cover generation pipeline...')
    if not os.path.exists(uploaded_file):
        raise FileNotFoundError(f'{uploaded_file} does not exist.')

    song_id = get_hash(uploaded_file)
    song_dir = os.path.join(OUTPUT_DIR, song_id)
    os.makedirs(song_dir, exist_ok=True)

    orig_song_path = convert_to_stereo(uploaded_file)
    ai_cover_path = os.path.join(song_dir, f'Converted_Voice.{output_format}')

    if os.path.exists(ai_cover_path):
        os.remove(ai_cover_path)

    print('[~] Converting vocals...')
    voice_change(voice_model, orig_song_path, ai_cover_path, pitch_change, f0_method, index_rate,
                 filter_radius, rms_mix_rate, protect, crepe_hop_length, f0_min, f0_max)

    print('[~] Conversion complete.')
    return ai_cover_path


def convert():
    import argparse

    parser = argparse.ArgumentParser(description="Convert a song using AI voice model")
    parser.add_argument('-i', '--uploaded_file', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('-rmod', '--voice_model', type=str, required=True, help='Name of the voice model to use')
    parser.add_argument('-p', '--pitch_change', type=float, default=0, help='Pitch change value')
    parser.add_argument('-opt', '--output_format', type=str, default='mp3', help='Output format (e.g., mp3, wav)')
    paser.add_argument('-ext', '--f0_method', type=str, default='rmvpe', help='F0 Method (e.g., rmvpe, fcpe')

    args = parser.parse_args()

    output = song_cover_pipeline(
        uploaded_file=args.uploaded_file,
        voice_model=args.voice_model,
        pitch_change=args.pitch_change,
        f0_method=args.f0_method,
        output_format=args.output_format
    )

    print(f'AI cover generated at: {output}')
