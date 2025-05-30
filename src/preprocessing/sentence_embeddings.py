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

def tokenize_sentences(file_path, tokenizer, model, output_dir):
    recording_name = file_path.split('/')[-1][:-5]
    data = json.load(open(file_path))
    text = data['text']
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     text = file.read()

    sentences = nltk.tokenize.sent_tokenize(text)
    jj = 0
    for sentence in sentences:
        encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        output = model(**encoded_input)
        sent_embedding = output[1].detach().numpy()
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        np.save(output_filename, sent_embedding)
        jj+=1


def process_directory_bert(transcript_path, output_dir, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    bert_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      tokenize_sentences(filepath, tokenizer, model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_name = config['sentence_model']
 
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['sentence_embedding_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_dir=output_path, 
            model_name=model_name)

