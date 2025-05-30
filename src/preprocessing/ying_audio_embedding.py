import json 
import argparse
import librosa
import torch
import os
import glob
import torch
import pandas as pd
import numpy as np 
from pathlib import Path
import pysptk
from joblib import Parallel, delayed
import nlpaug.augmenter.audio as naa
from transformers import Wav2Vec2Processor
 
def embed_audio(filepath, chunk_size, processor, output_dir):
    # Load the audio file
    sr = 16000
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    recording_name = filepath.split('/')[-1][:-4]

    N = audio.shape[0]
    K = 5
    max_len = 10*sr
    gap = (N-max_len)//K
    audio_data = torch.zeros([K, 160000])
    for i in range(K):
        temp_data = audio[gap*i:gap*i+max_len]
        ret = processor(
            temp_data,
            sampling_rate=sr,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        ret = ret.input_values.squeeze()
        audio_data[i] = ret


    # Save the features
    output_filename = os.path.join(output_dir, f"{recording_name}_{0:04}.pt")
    torch.save(audio_data, output_filename)


def get_audio_chunks(
        audio_dir, 
        chunk_size,
        output_dir 
        ):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")

    # Define a directory to save the features
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath=filepath, 
            chunk_size=chunk_size,
            processor=processor, 
            output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    chunk_size = 10
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_audio_chunks(
            audio_dir=dataset_config['audio_dir'], 
            chunk_size = chunk_size,
            output_dir=dataset_config['ying_audio_outdir'] 
            )
