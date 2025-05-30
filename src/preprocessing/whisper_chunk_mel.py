import pandas as pd
import os
import glob
# import whisper_timestamped as whisper
import whisper
import torch
from pathlib import Path
import json 
import argparse
from joblib import Parallel, delayed
import librosa
import numpy as np 


def embed_file(filepath, audio_dir, outdir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    language_holder = {}
    language_holder['Audio File'] = filepath.split('/')[-1]
    language_holder['Language'] = 'English'
    participant = filepath.split('/')[-1][:-4]
    print(participant)

    sr = 16000
    chunk_size = 30
    overlap_duration = 0
    samples_per_chunk = chunk_size * sr
    overlap_samples = overlap_duration * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)


    if len(audio) < chunk_size*sr:
        print(f"File {participant} has been padded to fill {chunk_size} seconds")
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
        chunk_tensor = torch.tensor(chunk[None,:]).to(device)
        chunk_mel = whisper.audio.log_mel_spectrogram(chunk_tensor).to('cpu').detach().numpy()
        
        # Save the features
        output_filename = os.path.join(outdir, f"{participant}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, chunk_mel)


    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()

    return language_holder

def get_embeddings(audio_dir, outdir, language_csv_name, model_size, compute_type, initial_prompt="", n_jobs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepaths = glob.glob(audio_dir + '/*.wav')
    filepaths.sort()
    # filepaths = filepaths[0:2]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for filepath in filepaths:
        language = embed_file(
            filepath, 
            audio_dir, 
            outdir, 
            )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_size = config['whisper_model_size']
    compute_type = config['whisper_compute_type']
    initial_prompt = config['initial_prompt']
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_embeddings(
            dataset_config['audio_dir'], 
            dataset_config['whisper_chunk_outdir'], 
            dataset_config['language_csv_name'], 
            model_size, 
            compute_type,
            )
