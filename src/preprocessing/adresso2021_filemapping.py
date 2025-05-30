import pandas as pd
import os
import glob
import argparse
import json

def create_associations_dataframe(groundtruth_df, language_df, features_dir, output_csv):
    df = pd.read_csv(groundtruth_df)
    df['tkdname'] = df['tkdname'] + '.wav'
    language_df = pd.read_csv(language_df, index_col=[0])
    language_df.rename(columns={'Audio File': 'tkdname'}, inplace=True)
    merged_df = df.merge(language_df, on='tkdname', how='inner')

    files_list = glob.glob(os.path.join(features_dir, '*.npy'))
    files_list.sort()

    associations = {}
    for filename in files_list:
        subject_pattern = os.path.basename(filename)[:-9]  # Assuming filename format is consistent
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
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)


    for dataset in config["datasets"]:
        print(dataset['setname'])
        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['w2v2_chunk_outdir'], 
        #     output_csv = dataset['w2v2_chunk_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['w2v2_single_outdir'], 
        #     output_csv = dataset['w2v2_single_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['bert_embedding_outdir'], 
        #     output_csv = dataset['bert_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['egemaps_chunk_outdir'], 
        #     output_csv = dataset['egemaps_chunk_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['distilbert_embedding_outdir'], 
        #     output_csv = dataset['distilbert_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['distilbert_token_outdir'], 
        #     output_csv = dataset['distilbert_token_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['sentence_embedding_outdir'], 
        #     output_csv = dataset['sentence_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['w2v2_sentence_outdir'], 
        #     output_csv = dataset['w2v2_sentence_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['prosody_sentence_outdir'], 
        #     output_csv = dataset['prosody_sentence_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['prosody_chunk_outdir'], 
        #     output_csv = dataset['prosody_chunk_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['prosody_rec_outdir'], 
        #     output_csv = dataset['prosody_rec_df'])

        # create_associations_dataframe(
        #     groundtruth_df = dataset['groundtruth_csv'], 
        #     language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
        #     features_dir = dataset['whisper_embedding_outdir'], 
        #     output_csv = dataset['whisper_embedding_df'])

        create_associations_dataframe(
            groundtruth_df = dataset['groundtruth_csv'], 
            language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
            features_dir = dataset['lftk_embedding_outdir'], 
            output_csv = dataset['lftk_df'])
