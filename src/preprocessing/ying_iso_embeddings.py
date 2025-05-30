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
import csv


def get_iso_embeddings(audio_dir, output_dir):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    # Initialize the Wav2Vec2 processor and feature extractor
    smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.eGeMAPSv02,
      feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    # Define a directory to save the features
    os.makedirs(output_dir, exist_ok=True)

    opensmile_path = '/sc/arion/projects/EASI-COG/jt_workspace/naxing_container/opensmile/build/progsrc/smilextract/SMILExtract'
    config_path = '/sc/arion/projects/EASI-COG/jt_workspace/naxing_container/opensmile/config/is09-13/IS10_paraling.conf'
    command_head = opensmile_path + ' -C ' + config_path + ' -I '
    csv_path = 'temp.csv' 
    command_tail = ' -O ' + csv_path + ' -instname ' + csv_path 

    for filepath in filepaths:
        recording_name = filepath.split('/')[-1][:-4]
        safe_filepath = filepath.replace(" ", "\\ ")
        print(recording_name)
        print(safe_filepath)
        output_filename = os.path.join(output_dir, f"{recording_name}_{0:04}.npy")
        command = command_head + safe_filepath + command_tail
        os.system(command)
        f = open(csv_path, 'r')
        df = list(csv.reader(f))[-1]
        feature_vec = np.array(df[1:-1], np.double)
        np.save(output_filename, feature_vec)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])

        get_iso_embeddings(
            dataset_config['audio_dir'], 
            dataset_config['ying_iso_outdir']
            )
