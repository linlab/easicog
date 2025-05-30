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
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from joblib import Parallel, delayed
import opensmile

def embed_audio(filepath, output_dir, extractor):
    # Load the audio file
    sr = 44100
    audio, _ = librosa.load(filepath, sr=sr)

    recording_name = filepath.split('/')[-1][:-4]

    features = extractor.process_signal(audio, sampling_rate=sr)   
        
    # Save the features
    output_filename = os.path.join(output_dir, f"{recording_name}_{0:04}.npy")
    np.save(output_filename, features)



def get_compare_embeddings(audio_dir, compare_outdir):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    # Initialize the Wav2Vec2 processor and feature extractor
    smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.ComParE_2016,
      feature_level=opensmile.FeatureLevel.Functionals)

    # Define a directory to save the features
    output_dir = compare_outdir
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath, 
            output_dir, 
            smile
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])

        get_compare_embeddings(
            audio_dir=dataset_config['audio_dir'], 
            compare_outdir=dataset_config['compare_outdir']
            )
