import pandas as pd
import os
import glob
import argparse
import json

def create_associations_dataframe(groundtruth_df, language_df, features_dir, output_csv):
    df = pd.read_csv(groundtruth_df)
    language_df = pd.read_csv(language_df, index_col=[0])
    language_df.rename(columns={'Audio File': 'tkdname'}, inplace=True)
    merged_df = df.merge(language_df, on='tkdname', how='inner')

    files_list = glob.glob(os.path.join(features_dir, '*.npy'))
    files_list.sort()

    associations = {}
    for filename in files_list:
        subject_pattern = os.path.basename(filename)[:-7]  # Assuming filename format is consistent
        matching_row = merged_df[merged_df['tkdname'].str.contains(subject_pattern)]

        if not matching_row.empty:
            index = matching_row.index[0]
            associations[filename] = merged_df.loc[index].to_dict()

    new_df = pd.DataFrame.from_dict(associations, orient='index')
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'index': 'Filename'}, inplace=True)
    new_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--config", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)


    for dataset in config["datasets"]:
        print(dataset['setname'])
        create_associations_dataframe(
            groundtruth_df = dataset['groundtruth_csv'], 
            language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
            features_dir = dataset['w2v2_chunk_outdir'], 
            output_csv = dataset['w2v2_chunk_df'])

        create_associations_dataframe(
            groundtruth_df = dataset['groundtruth_csv'], 
            language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
            features_dir = dataset['bert_embedding_outdir'], 
            output_csv = dataset['bert_df'])
