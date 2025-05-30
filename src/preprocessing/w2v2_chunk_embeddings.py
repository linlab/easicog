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
from src.utils.utils import denoise_audio, loudnorm_audio 
import pyloudnorm as pyln 


class w2v2FeatureExtractor:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
    
    def extract_features(self, audio_chunk, downsample_factor=0):
        features = self.feature_extractor(audio_chunk, return_tensors="pt",sampling_rate=16000)
        with torch.no_grad():
            w2v2_feats = self.model(features.input_values)
        flattened_tensor = w2v2_feats.extract_features.numpy()

        # Downsample the features
        if downsample_factor > 1:
            flattened_tensor = flattened_tensor[:,::downsample_factor,:]

        return flattened_tensor

def embed_audio(filepath, output_dir, extractor, noisereduce_audio, normalize_audio, chunk_size=5, overlap_duration=1, downsample_factor=1):
    # Load the audio file
    sr = 16000
    meter = pyln.Meter(sr)
    samples_per_chunk = chunk_size * sr
    overlap_samples = overlap_duration * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    if noisereduce_audio:
        audio = denoise_audio(audio, sr)
    if normalize_audio:
        audio = loudnorm_audio(audio, meter)
    recording_name = filepath.split('/')[-1][:-4]
    if len(audio) < chunk_size*sr:
        print(f"File {recording_name} has been padded to fill {chunk_size} seconds")
        audio = np.pad(audio, (0, chunk_size*sr - len(audio) + 1), 'constant')
    
    # Calculate the total number of chunks
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
        features = extractor.extract_features(chunk, downsample_factor)
        mean_features = features.mean(axis=1)
        
        # Save the features
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, mean_features)


    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()


def get_w2v2_embeddings(audio_dir, w2v2_chunk_outdir, w2v2_model, w2v2_chunk_size, downsample_factor, noisereduce_audio, normalize_audio):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    # Initialize the Wav2Vec2 processor and feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2Model.from_pretrained(w2v2_model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v2_model)
    extractor = w2v2FeatureExtractor(model=model,feature_extractor=feature_extractor)

    # Define the chunk size in seconds
    chunk_size = w2v2_chunk_size
    downsample_factor = downsample_factor
    overlap_duration = 1

    # Define a directory to save the features
    output_dir = w2v2_chunk_outdir
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath, 
            output_dir, 
            extractor, 
            noisereduce_audio,
            normalize_audio,
            chunk_size, 
            overlap_duration, 
            downsample_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    w2v2_model = config['w2v2_model']
    w2v2_chunk_size = 30
    downsample_factor = 1
    noisereduce_audio = False
    normalize_audio = False
    # for dataset_config in config["datasets"]:
    for dataset_config in [config["datasets"][1]]:
        print(dataset_config['setname'])
        get_w2v2_embeddings(
            dataset_config['audio_dir'], 
            dataset_config['w2v2_chunk_outdir'], 
            w2v2_model, 
            w2v2_chunk_size,
            downsample_factor,
            noisereduce_audio,
            normalize_audio)
