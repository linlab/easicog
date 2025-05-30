import pandas as pd
import os
import glob
import whisper_timestamped as whisper
import torch
from pathlib import Path
import json 
import argparse
import noisereduce as nr
from joblib import Parallel, delayed


def transcribe_file(filepath, audio_dir, outdir, model):
    # filepath = '/sc/arion/projects/EASI-COG/jt_migration/zipped_data/new_nu_ct/training/CT_LL_5.18.23.wav'
    # filepath = '/sc/arion/projects/EASI-COG/jt_migration/dementiabank/adresso21/training/adrso257.wav'
    # filepath = '/sc/arion/projects/EASI-COG/jt_migration/unzipped_data/separated_audio/training_set/Patient AC 6.4.2021.wav'
    
    language_holder = {}
    language_holder['Audio File'] = filepath.split('/')[-1]
    language_holder['Language'] = 'English'
    participant = filepath.split('/')[-1][:-4]
    print(participant)
    audio = whisper.load_audio(filepath)
    result = whisper.transcribe(
        model,
        audio,
        language='en',
        initial_prompt='Please separate utterances by period. A dog in the yard. Um, a girl sitting. Mom is washing the dishes.',
        beam_size=10,
        vad=True,
        condition_on_previous_text=False
        # vad='silero:5.0',
        #vad='auditok'
        )

    with open(os.path.join(outdir,participant+'_FIXING.json'), 'w', encoding='utf-8') as fp:
        json.dump(result, fp, ensure_ascii=False, indent=4)

    return language_holder


def get_transcriptions(audio_dir, outdir, language_csv_name, model_size, compute_type, n_jobs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepaths = glob.glob(audio_dir + '/*.wav')
    filepaths.sort()
    # filepaths = filepaths[:10]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(model_size, device=device.type)
    all_languages = []
    for filepath in filepaths:
        language = transcribe_file(
            filepath, 
            audio_dir, 
            outdir, 
            model)
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
    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        get_transcriptions(
            '/sc/arion/projects/EASI-COG/jt_migration/unzipped_data/problem_audio', 
            '/sc/arion/projects/EASI-COG/jt_migration/unzipped_data/problem_audio/problem_audio_transcripts', 
            dataset_config['language_csv_name'], 
            model_size, 
            compute_type,
            n_jobs=args.n_jobs)
        exit()
