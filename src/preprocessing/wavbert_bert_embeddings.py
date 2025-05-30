import numpy as np
import tensorflow as tf
import keras
import pandas as pd
# from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer
import os
import glob
import json 
import argparse
import torch
from pathlib import Path
from joblib import Parallel, delayed

def tokenize_text_file_bert(file_path, bert_tokenizer):
    data = json.load(open(file_path))
    text = data['text']
    text = text.replace(',', '')

    encoded_dict = bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
    
    del encoded_dict['token_type_ids'] #we don't need to save this so we delete it

    return encoded_dict

def process_directory_bert(transcript_path, output_path, model_name):

    bert_tokenizer = BertTokenizer.from_pretrained(model_name)

    bert_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      # Tokenize the text file using BERT tokenizer
      sequence_embedding = tokenize_text_file_bert(filepath, bert_tokenizer)
      torch.save(sequence_embedding,os.path.join(output_path,participant+"_0000.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_name = 'bert-base-uncased'
 
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['wavbert_nlp_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_path=output_path, 
            model_name=model_name)

