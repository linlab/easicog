import pandas as pd
import os
import glob
import whisper_timestamped as whisper
import torch
from pathlib import Path
import json 
import argparse
from joblib import Parallel, delayed


def transcribe_file(filepath, audio_dir, outdir, model, initial_prompt):
    language_holder = {}
    language_holder['Audio File'] = filepath.split('/')[-1]
    language_holder['Language'] = 'English'
    participant = filepath.split('/')[-1][:-4]
    print(participant)
    if False:
        result = whisper.transcribe(
            model,
            filepath,
            language='en',
            initial_prompt=initial_prompt,
            beam_size=10,
            vad=True,
            condition_on_previous_text=False #Added 2025-01-26 
            ) #'Please separate utterances by period. A dog in the yard. Um, a girl sitting. Mom is washing the dishes.'

        with open(os.path.join(outdir,participant+'.json'), 'w', encoding='utf-8') as fp:
            json.dump(result, fp, ensure_ascii=False, indent=4)

    return language_holder

def get_transcriptions(audio_dir, outdir, language_csv_name, model_size, compute_type, initial_prompt="", n_jobs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepaths = glob.glob(audio_dir + '/*.wav')
    filepaths.sort()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_size, device=device.type)
    all_languages = []
    for filepath in filepaths:
        language = transcribe_file(
            filepath, 
            audio_dir, 
            outdir, 
            model,
            initial_prompt)
        all_languages.append(language)
    
    all_languages = pd.DataFrame(all_languages)
    all_languages.to_csv(os.path.join(outdir,language_csv_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=3, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    model_size = config['whisper_model_size']
    compute_type = config['whisper_compute_type']
    initial_prompt = config['initial_prompt']
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_transcriptions(
            dataset_config['audio_dir'], 
            dataset_config['timestamped_transcription_outdir'], 
            dataset_config['language_csv_name'], 
            model_size, 
            compute_type,
            initial_prompt)
