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


def embed_audio(filepath, chunk_size, hop_size, output_dir):
    # Load the audio file
    sr = 16000
    samples_per_chunk = chunk_size * sr
    overlap_samples = hop_size * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    recording_name = filepath.split('/')[-1][:-4]

    aug0 = naa.MaskAug(sampling_rate=sr, zone=(0,1), coverage=0.3)
    aug1 = naa.LoudnessAug(zone=(0,1), factor=(0.3, 3))
    aug2 = naa.PitchAug(sampling_rate=sr, zone=(0,1))
    aug3 = naa.VtlpAug(zone=(0,1), sampling_rate=sr, factor=(0.5,3), coverage=1) # needs padding after!
    all_augs = [aug0, aug1, aug2, aug3]
    repeat_augs = 3


    if len(audio) < chunk_size*sr:
        print(f"File {recording_name} has been padded to fill {chunk_size} seconds")
        audio = np.pad(audio, (0, chunk_size*sr - len(audio) + 1), 'constant')
    
    total_duration = len(audio) / sr
    total_chunks = int((total_duration - chunk_size) / (chunk_size - hop_size)) + 1

    # Iterate through each chunk 
    jj = 0
    for i in range(total_chunks):
        # Calculate start and end sample indices for the current chunk
        start = i * (samples_per_chunk - overlap_samples)
        end = start + samples_per_chunk
        
        # Extract the current chunk
        chunk = audio[start:end]

        # Extract features from the chunk
        # features = np.nan_to_num(extractor.my_extract_static_features(chunk, sr))
        
        # Save the features
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, chunk)

        for kk in range(repeat_augs):
            for aug in all_augs:
                aug_segment = aug.augment(chunk)[0]
                if aug_segment.shape != chunk.shape:
                    pad_val = chunk.shape[0]-aug_segment.shape[0]
                    aug_segment = np.pad(aug_segment, (0,pad_val), 'constant', constant_values=0.0)

                output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
                jj+=1
                np.save(output_filename, aug_segment)



    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()


def get_audio_chunks(
        audio_dir, 
        chunk_size,
        hop_size,
        output_dir 
        ):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()
    # filepaths = filepaths[:2]

    # Define a directory to save the features
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath=filepath, 
            chunk_size=chunk_size, 
            hop_size=hop_size, 
            output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    chunk_size = config['hubert_chunk_size']
    hop_size = config['hubert_hop_size']
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_audio_chunks(
            audio_dir=dataset_config['audio_dir'], 
            chunk_size = chunk_size,
            hop_size = hop_size,
            output_dir=dataset_config['hubert_chunk_outdir'] 
            )
