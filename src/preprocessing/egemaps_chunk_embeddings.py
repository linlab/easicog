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

def embed_audio(filepath, output_dir, extractor, num_segments=10):
    sr = 44100
    chunk_size = 30
    overlap_duration = 0
    samples_per_chunk = chunk_size * sr
    overlap_samples = overlap_duration * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)

    recording_name = filepath.split('/')[-1][:-4]

    if len(audio) < chunk_size*sr:
        print(f"File {recording_name} has been padded to fill {chunk_size} seconds")
        audio = np.pad(audio, (0, chunk_size*sr - len(audio) + 1), 'constant')
    
    total_duration = len(audio) / sr
    total_chunks = int((total_duration - chunk_size) / (chunk_size - overlap_duration)) + 1

    # Iterate through each chunk 
    jj = 0
    for i in range(total_chunks):
        # Calculate start and end sample indices for the current chunk
        start = i * (samples_per_chunk - overlap_samples)
        end = start + samples_per_chunk
        
        # Extract the current chunk
        chunk = audio[start:end]

        # Extract features from the chunk
        _, _, features = extractor.process(chunk,sampling_rate=sr)
        features = np.nan_to_num(features)
        
        # Save the features
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, features)

    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()


def get_egemaps_embeddings(audio_dir, egemaps_chunk_outdir, num_segments=10):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    # Initialize the Wav2Vec2 processor and feature extractor
    smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.eGeMAPSv02,
      feature_level=opensmile.FeatureLevel.Functionals)

    # Define a directory to save the features
    output_dir = egemaps_chunk_outdir
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath, 
            output_dir, 
            smile, 
            num_segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
    # for dataset_config in [config["datasets"][1]]:
        print(dataset_config['setname'])

        get_egemaps_embeddings(
            dataset_config['audio_dir'], 
            dataset_config['egemaps_chunk_outdir'], 
            num_segments=10)
