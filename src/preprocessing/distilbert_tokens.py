import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import os
import glob
import json 
import argparse
import torch
from pathlib import Path
from joblib import Parallel, delayed

def tokenize_text_file_bert(file_path, bert_tokenizer):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text using BERT tokenizer and encode using BERT bare transformer
    input_ids = bert_tokenizer.encode(
        text=text, 
        add_special_tokens=True, 
        truncation=True, 
        padding='max_length',
        return_tensors='np')

    return input_ids

def process_directory_bert(transcript_path, output_path, model_name):

    bert_tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    bert_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.txt'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-4]
      print(participant)
      # Tokenize the text file using BERT tokenizer
      sequence_embedding = tokenize_text_file_bert(filepath, bert_tokenizer)
      np.save(os.path.join(output_path,participant+"_00.npy"),sequence_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    model_name = config['distilbert_model']
 
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['transcription_outdir']
        output_path = dataset_config['distilbert_token_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_path=output_path, 
            model_name=model_name)

