import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModel
import os
import glob
import json 
import argparse
import torch
from pathlib import Path
from joblib import Parallel, delayed
import gensim.downloader

def tokenize_sentences(file_path, word2vec_embedder, output_dir):
    recording_name = file_path.split('/')[-1][:-5]
    data = json.load(open(file_path))
    sentences = [segment['text'] for segment in data['segments']]

    jj = 0
    for sentence in sentences:
        all_words = []
        for word in sentence.split(' '):
            try:
                embedding_vector = word2vec_embedder[word]
            except:
                embedding_vector = np.zeros(300)
            all_words.append(embedding_vector)
        sent_embedding = np.stack(all_words)
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        np.save(output_filename, sent_embedding)
        jj+=1


def process_directory_bert(transcript_path, output_dir, model_name):

    word2vec_embedder = gensim.downloader.load(model_name)
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      tokenize_sentences(filepath, word2vec_embedder, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_name = 'word2vec-google-news-300'
 
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['bclstm_word2vec_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_dir=output_path, 
            model_name=model_name)

