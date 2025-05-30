import numpy as np
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import os
import glob
import json 
import argparse
import torch
from pathlib import Path
from joblib import Parallel, delayed

def tokenize_text_file_bert(file_path, bert_tokenizer, bert_model):
    data = json.load(open(file_path))
    text = data['text']
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     text = file.read()

    # Tokenize the text using BERT tokenizer and encode using BERT bare transformer
    input_ids = bert_tokenizer.encode(text=text, add_special_tokens=True, truncation=True, return_tensors='tf')
    bert_output = bert_model(input_ids)
    sequence_embedding = bert_output[1].numpy()

    return sequence_embedding

def process_directory_bert(transcript_path, output_path, model_name):

    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = TFBertModel.from_pretrained(model_name)

    bert_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      # Tokenize the text file using BERT tokenizer
      sequence_embedding = tokenize_text_file_bert(filepath, bert_tokenizer, bert_model)
      np.save(os.path.join(output_path,participant+"_0000.npy"),sequence_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_name = config['bert_model']
 
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['bert_embedding_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_path=output_path, 
            model_name=model_name)

