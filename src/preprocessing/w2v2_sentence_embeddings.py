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
    
    def extract_features(self, audio_chunk):
        features = self.feature_extractor(audio_chunk, return_tensors="pt",sampling_rate=16000)
        with torch.no_grad():
            w2v2_feats = self.model(features.input_values)
        flattened_tensor = w2v2_feats.extract_features.numpy()

        return flattened_tensor

def embed_audio(filepath, timestamp_json, output_dir, extractor, noisereduce_audio, normalize_audio):
    # Load the audio file
    sr = 16000
    meter = pyln.Meter(sr)
    chunk_size = 5
    overlap_duration = 1
    samples_per_chunk = chunk_size * sr
    overlap_samples = overlap_duration * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    if noisereduce_audio:
        audio = denoise_audio(audio, sr)
    if normalize_audio:
        audio = loudnorm_audio(audio, meter)
    recording_name = filepath.split('/')[-1][:-4]
    
    timestamps = json.load(open(timestamp_json))


    for segment in timestamps['segments']:
        start_samp = int(sr*segment['start'])
        end_samp = int(sr*segment['end'])
        seg_num = segment['id']
        chunk = audio[start_samp:end_samp]

        if len(chunk) < 512:
            print(f"File {recording_name} has been padded to fill {512} samples")
            chunk = np.pad(chunk, (0, 512 - len(chunk) + 1), 'constant')

        features = extractor.extract_features(chunk)
        mean_features = features.mean(axis=1)
        output_filename = os.path.join(output_dir, f"{recording_name}_{seg_num:04}.npy")
        np.save(output_filename, mean_features)



def get_w2v2_sentence_embeddings(
        timestamped_transcription_dir, 
        audio_dir, 
        w2v2_sentence_outdir, 
        w2v2_model,
        noisereduce_audio, 
        normalize_audio 
        ):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    timestamp_jsons = glob.glob(os.path.join(timestamped_transcription_dir,'*.json'))
    timestamp_jsons.sort()

    for filepath, timestamp_json in zip(filepaths,timestamp_jsons):
        print(filepath, timestamp_json, '\n')
        assert os.path.basename(filepath)[:-4] == os.path.basename(timestamp_json)[:-5]

    # Initialize the Wav2Vec2 processor and feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Vec2Model.from_pretrained(w2v2_model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v2_model)
    extractor = w2v2FeatureExtractor(model=model,feature_extractor=feature_extractor)

    # Define a directory to save the features
    output_dir = w2v2_sentence_outdir
    os.makedirs(output_dir, exist_ok=True)

    for filepath, timestamp_json in zip(filepaths, timestamp_jsons):
        print(f'Embedding {filepath}')
        embed_audio(
            filepath, 
            timestamp_json,
            output_dir, 
            extractor,
            noisereduce_audio, 
            normalize_audio
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    w2v2_model = config['w2v2_model']
    noisereduce_audio = config['noisereduce_audio']
    normalize_audio = config['normalize_audio']
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_w2v2_sentence_embeddings(
            timestamped_transcription_dir=dataset_config['fixed_timestamped_transcription_outdir'],
            audio_dir=dataset_config['audio_dir'], 
            w2v2_sentence_outdir=dataset_config['w2v2_sentence_outdir'], 
            w2v2_model=w2v2_model,
            noisereduce_audio=noisereduce_audio,
            normalize_audio=normalize_audio 
            )
