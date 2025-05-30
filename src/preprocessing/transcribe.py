import pandas as pd
import os
import glob
from faster_whisper import WhisperModel
import torch
from pathlib import Path
import json 
import argparse
from joblib import Parallel, delayed


def transcribe_file(filepath, audio_dir, outdir, model):
    language_holder = {}
    language_holder['Audio File'] = filepath.split('/')[-1]
    participant = filepath.split('/')[-1][:-4]
    print(participant)
    segments, info = model.transcribe(filepath, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    if info.language != 'en':
        print("Forced Chinese transcription")
        language_holder['Language'] = 'Chinese'
        segments, info = model.transcribe(
            filepath, 
            language='zh', 
            beam_size=5)
    else:
        language_holder['Language'] = 'English'
        segments, info = model.transcribe(
            filepath, 
            initial_prompt='Please, use punctuation where appropriate.',
            language='en', 
            beam_size=5)
    print('\n')
    df_holder = []
    all_text = []
    for segment in segments:
        temp_df = {}
        temp_df["Start"] = segment.start
        temp_df["End"] = segment.end
        temp_df["Transcription"] = segment.text
        df_holder.append(temp_df)
        all_text.append(segment.text)

    transcription_df = pd.DataFrame(df_holder)
    transcription_df.to_csv(os.path.join(outdir,participant+'.csv'))

    concat_text = "".join(all_text)
    with open(os.path.join(outdir,participant+'.txt'), 'w') as f:
        f.write(concat_text)

    return language_holder

def get_transcriptions(audio_dir, outdir, language_csv_name, model_size, compute_type, n_jobs=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepaths = glob.glob(audio_dir + '/*.wav')
    filepaths.sort()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = WhisperModel(model_size, device=device.type, compute_type=compute_type)
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
            dataset_config['audio_dir'], 
            dataset_config['transcription_outdir'], 
            dataset_config['language_csv_name'], 
            model_size, 
            compute_type,
            n_jobs=args.n_jobs)
