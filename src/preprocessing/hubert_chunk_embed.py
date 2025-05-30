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
from transformers import HubertModel, AutoProcessor

def embed_file(filepath, audio_dir, outdir, model, processor):
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

        data = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        
        outputs = model(input_values=data['input_values']).last_hidden_state

        chunk_embedding = outputs.mean(axis=1).detach().cpu().numpy()
      
        # Save the features
        output_filename = os.path.join(outdir, f"{participant}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, chunk_embedding)


    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()

    return language_holder

def get_embeddings(audio_dir, outdir, language_csv_name, n_jobs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepaths = glob.glob(audio_dir + '/*.wav')
    filepaths.sort()
    # filepaths = filepaths[0:2]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    for parameter in model.parameters():
            parameter.requires_grad = False
    for filepath in filepaths:
        language = embed_file(
            filepath, 
            audio_dir, 
            outdir, 
            model,
            processor)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=3, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    
    for dataset_config in config["datasets"]:
    # for dataset_config in [config["datasets"][1]]:
        print(dataset_config['setname'])
        get_embeddings(
            dataset_config['audio_dir'], 
            dataset_config['hubert_chunk_outdir'], 
            dataset_config['language_csv_name']
            )
