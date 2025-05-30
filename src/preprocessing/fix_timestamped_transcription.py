import os
import nltk
import json
import glob
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
import argparse

def fix_transcriptions(origin_dir, destination_dir):
      
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    filepaths = glob.glob(os.path.join(origin_dir,'*.json'))
    filepaths.sort()
    # filepaths = filepaths[55:]

    for ii, filepath in enumerate(filepaths):
        temp_df = {}
        basename = os.path.basename(filepath)
        print(f"Filepath {ii}: {basename}")
        data = json.load(open(filepath))
        rows = []
        text = ""
        for segment in data['segments']:
            end = segment['end']
            start = segment['start']

            if end - start > 0.2:# and segment['compression_ratio'] < 2.0:
                text += segment['text']
                for word in segment['words']:
                    if len(word['text'].split()) == 1:
                        rows.append({
                            'text': word['text'],
                            'start': word['start'],
                            'end': word['end'],
                            'confidence': word['confidence']
                        })
                    else:
                        for part_word in word['text'].split():
                            rows.append({
                                'text': part_word,
                                'start': word['start'],
                                'end': word['end'],
                                'confidence': word['confidence']
                            })



        df = pd.DataFrame(rows)
        df['sentence_id'] = -1  # Initialize with -1 to indicate unassigned sentences

        # text = data['text']
        sentences = nltk.sent_tokenize(text.strip())
        num_sents = len(sentences)
        print(f'Number of sentences: {num_sents}')
        search_start_index = 0

        
        start_row = 0
        for sentence_number, sentence in enumerate(sentences):
            sentence_words = sentence.split()
            
            end_row = start_row + len(sentence_words)
            df.loc[start_row:end_row, 'sentence_id'] = sentence_number
            start_row += len(sentence_words)

            
        # Constructing the JSON structure
        # breakpoint()
        json_output = {
            "text": " ".join(df['text']),  # Join all text
            "segments": []
        }

        # Group by sentence_id
        for sentence_id, group in df.groupby('sentence_id'):
            if sentence_id == -1:  # Skip unassigned words
                continue

            segment = {
                "id": sentence_id,
                "seek": 0,
                "start": group['start'].min(),
                "end": group['end'].max(),
                "text": " ".join(group['text']),
                "tokens": [],  # Add your tokens if available
                "temperature": 0.0,  # Placeholder, adjust as needed
                "avg_logprob": 0.0,  # Placeholder, adjust as needed
                "compression_ratio": 0.0,  # Placeholder, adjust as needed
                "no_speech_prob": 0.0,  # Placeholder, adjust as needed
                "confidence": group['confidence'].mean(),  # Average confidence
                "words": []
            }
            
            # Append word details
            for _, row in group.iterrows():
                segment["words"].append({
                    "text": row['text'],
                    "start": row['start'],
                    "end": row['end'],
                    "confidence": row['confidence']
                })
            
            json_output["segments"].append(segment)

        with open(os.path.join(destination_dir,basename), 'w', encoding='utf-8') as fp:
                json.dump(json_output, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=3, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        fix_transcriptions(
            dataset_config['timestamped_transcription_outdir'], 
            dataset_config['fixed_timestamped_transcription_outdir']
        )
