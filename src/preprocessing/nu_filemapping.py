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

def create_nu_associations_dataframe(groundtruth_df, language_df, features_dir, output_csv):
    df = pd.read_csv(groundtruth_df)
    df['tkdname'] = df.Subject.str[:] + '.wav'
    
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

def create_pytorch_nu_associations_dataframe(groundtruth_df, language_df, features_dir, output_csv):
    df = pd.read_csv(groundtruth_df)
    df['tkdname'] = df.Subject.str[:] + '.wav'
    
    language_df = pd.read_csv(language_df, index_col=[0])
    language_df.rename(columns={'Audio File': 'tkdname'}, inplace=True)
    merged_df = df.merge(language_df, on='tkdname', how='inner')

    files_list = glob.glob(os.path.join(features_dir, '*.pt'))
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

    mapping_function = create_nu_associations_dataframe
    pytorch_mapping_function = create_pytorch_nu_associations_dataframe


    for dataset in config["datasets"]:
        print(dataset['setname'])

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['promaps_embedding_outdir'], 
                output_csv = dataset['promaps_embedding_df'])
        except:
            print('No promaps embeddings found')

        exit()
        
        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['whisper_embedding_outdir'], 
                output_csv = dataset['whisper_embedding_df'])
        except:
            print('No whisper embeddings found')



        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['w2v2_chunk_outdir'], 
                output_csv = dataset['w2v2_chunk_df'])
        except:
            print('No W2V2 Chunk features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['w2v2_single_outdir'], 
                output_csv = dataset['w2v2_single_df'])
        except:
            print('No W2V2 Single features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['bert_embedding_outdir'], 
                output_csv = dataset['bert_df'])
        except:
            print('No BERT features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['egemaps_chunk_outdir'], 
                output_csv = dataset['egemaps_chunk_df'])
        except:
            print('No egemaps features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['compare_outdir'], 
                output_csv = dataset['compare_df'])
        except:
            print('No egemaps features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['distilbert_embedding_outdir'], 
                output_csv = dataset['distilbert_df'])
        except:
            print('No DistilBERT features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['distilbert_token_outdir'], 
                output_csv = dataset['distilbert_token_df'])
        except:
            print('No DistilBERT Token features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['sentence_embedding_outdir'], 
                output_csv = dataset['sentence_df'])
        except:
            print('No MPNet Sentence features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['w2v2_sentence_outdir'], 
                output_csv = dataset['w2v2_sentence_df'])
        except:
            print('No W2V2 Sentence features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['prosody_sentence_outdir'], 
                output_csv = dataset['prosody_sentence_df'])
        except:
            print('No Prosody Sentence features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['prosody_chunk_outdir'], 
                output_csv = dataset['prosody_chunk_df'])
        except:
            print('No Prosody Chunk features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['prosody_rec_outdir'], 
                output_csv = dataset['prosody_rec_df'])
        except:
            print('No Prosody Chunk features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['lftk_embedding_outdir'], 
                output_csv = dataset['lftk_df'])
        except:
            print('No LFTK features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['hubert_chunk_outdir'], 
                output_csv = dataset['hubert_chunk_df'])
        except:
            print('No DA4AD features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['significant_nltk_outdir'], 
                output_csv = dataset['significant_nltk_df'])
        except:
            print('No Significant NLTK features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['misa_acoustic_sentence_outdir'], 
                output_csv = dataset['misa_acoustic_sentence_df'])
        except:
            print('No MISA acoustic features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['misa_bert_sentence_outdir'], 
                output_csv = dataset['misa_bert_sentence_df'])
        except:
            print('No MISA BERT features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['mfn_glove_word_outdir'], 
                output_csv = dataset['mfn_glove_word_df'])
        except:
            print('No MFN Glove features found')

        try:
            pytorch_mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['ying_bert_outdir'], 
                output_csv = dataset['ying_bert_df'])
        except:
            print('No Ying Bert features found')

        try:
            pytorch_mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['ying_audio_outdir'], 
                output_csv = dataset['ying_audio_df'])
        except:
            print('No Ying audio found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['ying_iso_outdir'], 
                output_csv = dataset['ying_iso_df'])
        except:
            print('No MFN Glove features found')

        try:
            mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['farzana_nlp_outdir'], 
                output_csv = dataset['farzana_nlp_df'])
        except:
            print('No Farzana nlp features found')

        try:
            pytorch_mapping_function(
                groundtruth_df = dataset['groundtruth_csv'], 
                language_df = os.path.join(dataset['transcription_outdir'],dataset['language_csv_name']),
                features_dir = dataset['wavbert_nlp_outdir'], 
                output_csv = dataset['wavbert_nlp_df'])
        except:
            print('No wavbert nlp features found')

