import json 
import argparse
import librosa
import torch
import os
import glob
import torch
import pandas as pd
import numpy as np 
import scipy
from pathlib import Path
from disvoice.prosody.prosody import Prosody 
from disvoice.prosody.prosody_functions import V_UV, F0feat, energy_cont_segm, polyf0, energy_feat, dur_seg, duration_feat, get_energy_segment
import pysptk
from joblib import Parallel, delayed
from src.utils.utils import denoise_audio, loudnorm_audio 
import pyloudnorm as pyln 

def embed_covarep_mat(timestamp_json, covarep_filepath, output_dir):
    # Load the audio file
    covarep_mat = scipy.io.loadmat(covarep_filepath)
    covarep_features = covarep_mat['features']
    recording_name = covarep_filepath.split('/')[-1][:-4]
    
    timestamps = json.load(open(timestamp_json))

    for segment in timestamps['segments']:
        all_feats = []
        seg_num = segment['id']
        for word in segment['words']:
            start_ind = int(100 * word['start'])
            end_ind = int(100 * word['end'])
            covarep_chunk = covarep_features[start_ind:end_ind]
            mean_covarep_chunk = covarep_chunk.mean(axis=0)
            all_feats.append(mean_covarep_chunk)

        stacked_feats = np.stack(all_feats)
        output_filename = os.path.join(output_dir, f"{recording_name}_{seg_num:04}.npy")
        np.save(output_filename, stacked_feats)

# def legacy_embed_covarep_mat(timestamp_json, covarep_filepath, output_dir):
#     # Load the audio file
#     truncation_length = 32 #Mean 12 words per sentence, std 10 words
#     covarep_mat = scipy.io.loadmat(covarep_filepath)
#     covarep_features = covarep_mat['features']
#     recording_name = covarep_filepath.split('/')[-1][:-4]
    
#     timestamps = json.load(open(timestamp_json))

#     for segment in timestamps['segments']:
#         all_feats = []
#         seg_num = segment['id']
#         for word in segment['words']:
#             start_ind = int(100*word['start'])
#             end_ind = int(100*word['end'])
#             seg_num = segment['id']
#             covarep_chunk = covarep_features[start_ind:end_ind]
#             mean_covarep_chunk = covarep_chunk.mean(axis=0)
#             all_feats.append(mean_covarep_chunk)

#         stacked_feats = np.stack(all_feats)
#         if stacked_feats.shape[0] < truncation_length:
#             padding = truncation_length - stacked_feats.shape[0]
#             stacked_feats = np.pad(stacked_feats, ((0, padding), (0, 0)), mode='constant')
#         elif stacked_feats.shape[0] > truncation_length:
#             stacked_feats = stacked_feats[:truncation_length, :]

#         output_filename = os.path.join(output_dir, f"{recording_name}_{seg_num:04}.npy")
#         np.save(output_filename, stacked_feats)

def get_misa_acoustic_sentence_embeddings(
        timestamped_transcription_dir, 
        audio_dir, 
        covarep_dir,
        misa_acoustic_outdir
        ):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    covarep_mats = glob.glob(os.path.join(covarep_dir,'*.mat'))
    covarep_mats.sort()

    timestamp_jsons = glob.glob(os.path.join(timestamped_transcription_dir,'*.json'))
    timestamp_jsons.sort()

    for filepath, covarep_mat, timestamp_json in zip(filepaths,covarep_mats,timestamp_jsons):
        assert os.path.basename(filepath)[:-4] == os.path.basename(covarep_mat)[:-4] == os.path.basename(timestamp_json)[:-5] 

    # Define a directory to save the features
    output_dir = misa_acoustic_outdir
    os.makedirs(output_dir, exist_ok=True)


    for covarep_mat, timestamp_json in zip(covarep_mats, timestamp_jsons):
        print(f'Embedding {covarep_mat}')
        embed_covarep_mat(
            timestamp_json,
            covarep_mat,
            output_dir
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    noisereduce_audio = config['noisereduce_audio']
    normalize_audio = config['normalize_audio']

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_misa_acoustic_sentence_embeddings(
            timestamped_transcription_dir=dataset_config['fixed_timestamped_transcription_outdir'],
            audio_dir=dataset_config['audio_dir'], 
            covarep_dir=dataset_config['covarep_dir'],
            misa_acoustic_outdir=dataset_config['misa_acoustic_sentence_outdir']
            )
