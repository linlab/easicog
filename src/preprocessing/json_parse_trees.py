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
import spacy
import lftk

def embed_file(filepath, nlp, feature_set):
    data = json.load(open(filepath))
    text = data['text']
    # with open(filepath, 'r', encoding='utf-8') as file:
    #     text = file.read()

    # Tokenize the text using BERT tokenizer and encode using BERT bare transformer
    doc = nlp(text)
    LFTK = lftk.Extractor(docs=doc)

    # optionally, you can customize how LFTK extractor calculates handcrafted linguistic features
    # for example, include stop word? include puncutaion? maximum decimal digits?
    LFTK.customize(stop_words=True, punctuations=True, round_decimal=3)

    # now, extract the handcrafted linguistic features that you need
    # refer to them as feature keys
    extracted_features = LFTK.extract(features=feature_set)
    sequence_embedding = pd.DataFrame(extracted_features, index=[0]).values

    return sequence_embedding

def process_directory_bert(transcript_path, output_path):

    nlp = spacy.load("en_core_web_sm")
    feature_set = lftk.search_features(domain="surface", return_format='list_key')
    feature_set += lftk.search_features(domain="syntactic", return_format='list_key')
    feature_set += lftk.search_features(domain="lexico-semantics", return_format='list_key')



    lftk_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      sequence_embedding = embed_file(filepath, nlp, feature_set)
      np.save(os.path.join(output_path,participant+"_0000.npy"),sequence_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['parse_tree_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_path=output_path)

