import sys
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Adam as torchAdam
from torch.optim import AdamW as torchAdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import keras
import pandas as pd
import argparse
import json
import os
import pickle
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA 
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from keras.models import Model
from keras.optimizers import Adam, AdamW
from keras.models import load_model
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint
from eipy.ei import EnsembleIntegration

from src.models.models import DistilBERTSeqClassifier, BERTClassifier, W2V2Classifier, B2AIW2V2Classifier, GauderClassifier, Autoencoder, EIPredictors, FastEIPredictors
from src.models.models import Autoencoder, EIPredictors, FastEIPredictors, BasePredictors, KUModel, DA4ADClassifier, MISAClassifier, MFNClassifier, SigNLTKPredictors
from src.models.models import BCLSTMClassifier, BCLSTMUnimodalClassifier, BCLSTMTextCNN 
import src.models.models as models
from src.data.dataloaders import StandardDataGenerator, AutoencoderDataGenerator, DA4ADDataset, MISADataset, MFNDataset, BCLSTMDataset, WhisperDataset, WhisperRNNDataset
import src.data.dataloaders as dataloaders
from src.utils import utils 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", type=str, help="Path to the JSON configuration file.")
    parser.add_argument("--training_config", type=str, help="Which embeddings to train on")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of train/test splits to generate")
    parser.add_argument("--model_name", type=str, default="model", help="N")
    parser.add_argument("--offset", type=int, default=-1, help="Offset for fold random seed")
    parser.add_argument('--clf_name', type=str, default=None)
    parser.add_argument('--make_val_set', action='store_true')

    args = parser.parse_args()

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    with open(args.training_config) as f:
        training_config = json.load(f)

    print('Configs parsed', file=sys.stderr)

    train_dataset = [i for i in dataset_config['datasets'] if i['setname']=='training'][0]
    train_groundtruth_df, train_data_df = utils.prepare_dataframes(train_dataset, training_config, cohort=dataset_config['cohort'])

    test_dataset = [i for i in dataset_config['datasets'] if i['setname']=='testing'][0]
    test_groundtruth_df, test_data_df = utils.prepare_dataframes(test_dataset, training_config, cohort=dataset_config['cohort'])

    groundtruth_df = pd.concat((train_groundtruth_df,test_groundtruth_df))
    data_df = pd.concat((train_data_df,test_data_df))

    # test_dataset = [i for i in dataset_config['datasets'] if i['setname']=='testing'][0]
    # test_groundtruth_df, test_data_df = utils.prepare_dataframes(test_dataset, training_config, cohort=dataset_config['cohort'])

    # groundtruth_df = test_groundtruth_df
    # data_df = test_data_df

    print('Groundtruth df generated', file=sys.stderr)

    all_scores = []
    all_preds = []
    offset = args.offset
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    for ii, (train_index, test_index) in enumerate(skf.split(groundtruth_df, groundtruth_df['dx'].astype(int))):
        if args.offset > -1:
            if ii != offset:
                print(f'skipping {ii}')
                continue
        
        print(f'Fold {ii} of {args.n_folds}')
        fold_num = ii
        mci_preds = []
        mmse_scores = []
        train_df = groundtruth_df.iloc[train_index]
        test_df = groundtruth_df.iloc[test_index]

        if not args.make_val_set:
            resamp_train_df = utils.random_oversample_dataframe(train_df)

            # Get the resampled subjects
            resamp_train_subjects = resamp_train_df['Subject']
            # Create a dataframe to count occurrences of each subject
            subject_counts = resamp_train_subjects.value_counts().reset_index()
            subject_counts.columns = ['Subject', 'Count']
            # Merge the subject counts with data_df to get the repeated instances
            train_data_df = data_df.merge(subject_counts, on='Subject', how='inner')
            # Repeat each row according to its count
            train_data_df = train_data_df.loc[train_data_df.index.repeat(train_data_df['Count'])].reset_index(drop=True)
            # Drop the count column as it's no longer needed
            train_data_df = train_data_df.drop(columns=['Count'])

            mean_age = train_data_df['age'].mean()
            std_age = train_data_df['age'].std() + 1e-7
            train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
            train_tkdnames = train_data_df['tkdname'].drop_duplicates()

            test_subjects = test_df['Subject']
            test_data_df = data_df[data_df['Subject'].isin(test_subjects)]
            test_data_df['age'] = (test_data_df['age'] - mean_age) / (std_age)
            test_tkdnames = test_data_df['tkdname'].drop_duplicates()
        else:
            print('Making Validation Data')
            train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['dx'], random_state=ii)

            resamp_train_df = utils.random_oversample_dataframe(train_df)
            # Get the resampled subjects
            resamp_train_subjects = resamp_train_df['Subject']
            # Create a dataframe to count occurrences of each subject
            subject_counts = resamp_train_subjects.value_counts().reset_index()
            subject_counts.columns = ['Subject', 'Count']
            # Merge the subject counts with data_df to get the repeated instances
            train_data_df = data_df.merge(subject_counts, on='Subject', how='inner')
            # Repeat each row according to its count
            train_data_df = train_data_df.loc[train_data_df.index.repeat(train_data_df['Count'])].reset_index(drop=True)
            # Drop the count column as it's no longer needed
            train_data_df = train_data_df.drop(columns=['Count'])

            mean_age = train_data_df['age'].mean()
            std_age = train_data_df['age'].std() + 1e-7
            train_data_df['age'] = (train_data_df['age'] - mean_age) / (std_age)
            train_tkdnames = train_data_df['tkdname'].drop_duplicates()

            val_subjects = val_df['Subject']
            val_data_df = data_df[data_df['Subject'].isin(val_subjects)]
            val_data_df['age'] = (val_data_df['age'] - mean_age) / (std_age)
            val_tkdnames = val_data_df['tkdname'].drop_duplicates()

            test_subjects = test_df['Subject']
            test_data_df = data_df[data_df['Subject'].isin(test_subjects)]
            test_data_df['age'] = (test_data_df['age'] - mean_age) / (std_age)
            test_tkdnames = test_data_df['tkdname'].drop_duplicates()

            print('done making validation data')


        if training_config['architecture']=='BP':
            if training_config['target'] == "MCI":
                if "Audio_Filename" in train_data_df.columns:
                    modality = 'MULTIMODAL'
                    train_lang_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Language_Filename']]
                        )
                    test_lang_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Language_Filename']]
                        )
                    train_aud_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Audio_Filename']]
                        )
                    test_aud_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Audio_Filename']]
                        )
                    train_embeddings = np.concatenate((train_lang_embeddings,train_aud_embeddings), axis=-1)
                    test_embeddings = np.concatenate((test_lang_embeddings,test_aud_embeddings), axis=-1)
                else: 
                    modality = None   
                    train_embeddings = np.vstack(
                        [np.load(filepath) for filepath in train_data_df['Filename']]
                        )
                    test_embeddings = np.vstack(
                        [np.load(filepath) for filepath in test_data_df['Filename']]
                        )

                train_mean = train_embeddings.mean(axis=0)
                train_std = train_embeddings.std(axis=0)
                train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7)
                train_y_true = train_data_df['dx'].values.astype(int)
                train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
                
                test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
                test_y_true = test_data_df['dx'].values.astype(int)
                test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
        
            data_train = {training_config['modality']: train_embeddings}
            data_test = {training_config['modality']: test_embeddings}

            # base_predictors = models.BasePredictors()
            # base_predictors = models.FastBasePredictors()
            # base_predictors = SigNLTKPredictors()
            base_predictors = models.XGBoostGridSearch()
            # print(base_predictors)

            # clf_name = 'XGB_n500_d6_lr0.1_csbt1.0_sub1.0'
            
            print('\n\n\nBeginning Training!!\n\n\n')
            for predictor_name, base_predictor in base_predictors.base_predictors.items():
                print(f'Fitting {predictor_name}')
                if True:
                # if predictor_name == clf_name:
                    base_predictor.fit(train_embeddings,train_y_true)
                    eval_df = []
                    eval_df = utils.get_predictions(
                        test_tkdnames, 
                        test_data_df, 
                        base_predictor, 
                        EI=True, 
                        modality=modality,
                        data_mean=train_mean, 
                        data_std=train_std
                    )
                    eval_df['Fold'] = fold_num
                    eval_df['Ensemble Model'] = predictor_name
                    all_preds.append(eval_df)

        elif training_config['architecture']=='WHISPER_AUDIO':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device, file=sys.stderr)
            if training_config['target'] == "MCI":
                train_dataset = WhisperDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=32, 
                    shuffle=True,
                    drop_last=True)
                val_dataset = WhisperDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=32,
                    shuffle=True,
                    drop_last=True)
                test_dataset = WhisperDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False)

                
                # Define the model
                model = models.WhisperClassifier()               
                optimizer = torchAdam(model.parameters(), lr=1e-7)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Training loop
                # print('Begin training', file=sys.stderr)
                best_val_loss = float('inf')
                min_patience = 2
                patience = 10
                early_stop_counter = 0
                for epoch in range(100):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        inputs = [x.to(device) for x in inputs]
                        mci_targets = [t.to(device) for t in targets]

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets[0].float())
                        loss = loss_mci 

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)} \n", file=sys.stderr)
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            inputs = [x.to(device) for x in val_inputs]
                            mci_targets = [t.to(device) for t in val_targets]
                            val_outputs = model(inputs)
                            #Classification loss
                            loss_mci = criterion_mci(val_outputs.squeeze(), mci_targets[0].float())
                            val_losses.append(loss_mci.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f} \n', file=sys.stderr)
                    if avg_val_loss < best_val_loss and epoch >= min_patience:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    elif epoch+1 > min_patience:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}', file=sys.stderr)
                        break
                model.load_state_dict(best_model)

            model.eval()
            torch.cuda.empty_cache()

            with torch.no_grad():
                print(f"Getting training embeddings \n", file=sys.stderr)
                train_embeddings = []
                train_y_true = []
                for inputs, targets in train_loader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    inputs = [x.to(device) for x in inputs]
                    mci_targets = [t.to('cpu') for t in targets]

                    temp_embeddings = model.encoder(inputs[0]).to('cpu').detach().numpy().mean(axis=1)

                    train_embeddings.append(temp_embeddings)
                    train_y_true.append(mci_targets)
                print(f"Getting val embeddings \n", file=sys.stderr)
                for inputs, targets in val_loader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    inputs = [x.to(device) for x in inputs]
                    mci_targets = [t.to('cpu') for t in targets]

                    temp_embeddings = model.encoder(inputs[0]).to('cpu').detach().numpy().mean(axis=1)

                    train_embeddings.append(temp_embeddings)
                    train_y_true.append(mci_targets) 

                print(f"Stacking embeddings \n", file=sys.stderr)
                train_embeddings = np.vstack(train_embeddings)
                train_y_true = np.squeeze(np.concatenate(train_y_true, axis=1))
                print(f"Embeddings stacked \n", file=sys.stderr)

                print(np.shape(train_embeddings), file=sys.stderr)
                print(np.shape(train_y_true), file=sys.stderr)



                print(f"Getting testset embeddings \n", file=sys.stderr)
                tkdname_indices = {}
                all_tkdnames = []
                test_embeddings = []
                index = 0
                for tkdname in test_tkdnames:
                    filtered_rows = test_data_df[test_data_df['tkdname'] == tkdname]
                    if not filtered_rows.empty:
                        # Get the inputs for the neural network
                        inputs = filtered_rows.drop(columns=['tkdname'])
                        npy_filenames = inputs['Filename'].tolist()
                        X = []
                        for filename in npy_filenames:
                            all_tkdnames.append(tkdname)
                            data = np.load(filename)
                            # Append data and demographics to the lists
                            X.append(data)
                        tkdname_indices[tkdname] = (index, index + len(X))
                        index += len(X)
                        stacked_X = torch.tensor(np.vstack(X), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            temp_embeddings = model.encoder(stacked_X).to('cpu').detach().numpy().mean(axis=1)
                        test_embeddings.append(temp_embeddings)

                test_embeddings = np.vstack(test_embeddings)
                # eval_df = pd.DataFrame({'tkdname': all_tkdnames, 'output':outputs_mci})
                print(f"Predictions calculated \n", file=sys.stderr)

            train_mean = train_embeddings.mean(axis=0)
            train_std = train_embeddings.std(axis=0)
            train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7)
            # train_y_true = train_data_df['dx'].values.astype(int)
            # train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
            
            test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
            # test_y_true = test_data_df['dx'].values.astype(int)
            # test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
    
            data_train = {training_config['modality']: train_embeddings}
            data_test = {training_config['modality']: test_embeddings}

            # base_predictors = models.BasePredictors()
            # base_predictors = models.FastBasePredictors()
            # base_predictors = SigNLTKPredictors()
            base_predictors = models.XGBoostGridSearch()
            # print(base_predictors)
            
            print('\n\n\nBeginning Training!!\n\n\n', file=sys.stderr)
            for predictor_name, base_predictor in base_predictors.base_predictors.items():
                print(f'Fitting {predictor_name}', file=sys.stderr)
                base_predictor.fit(train_embeddings,train_y_true)

                eval_df = []
                outputs = base_predictor.predict_proba(test_embeddings)
                outputs = outputs[:,1]

                eval_df = pd.DataFrame({'tkdname': all_tkdnames})
                eval_df['output'] = outputs
                eval_df['Fold'] = fold_num
                eval_df['Ensemble Model'] = f'Whisper_Audio_Finetune_{predictor_name}'
                all_preds.append(eval_df)

                # exit()
                # eval_df['Fold'] = fold_num
                # eval_df['Ensemble Model'] = 'Whisper_Audio_Finetune'
                # all_preds.append(eval_df)

            del model


        elif training_config['architecture']=='WHISPER_RNN':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device, file=sys.stderr)
            if training_config['target'] == "MCI":
                train_tkdnames = train_data_df['tkdname'].drop_duplicates()
                train_dataset = WhisperRNNDataset(dataframe=train_data_df, tkdnames=train_tkdnames)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, 
                    shuffle=True,
                    drop_last=True,
                    collate_fn=utils.whisperrnn_collate_fn)
                
                val_tkdnames = val_data_df['tkdname'].drop_duplicates()
                val_dataset = WhisperRNNDataset(dataframe=val_data_df, tkdnames=val_tkdnames)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=16,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=utils.whisperrnn_collate_fn)
                test_tkdnames = test_data_df['tkdname'].drop_duplicates()
                test_dataset = WhisperRNNDataset(dataframe=test_data_df, tkdnames=test_tkdnames)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.whisperrnn_collate_fn)

                # Define the model
                sample_aud = np.load(train_data_df['Filename'][0])
                model = models.WhisperRNNClassifier(input_dim=sample_aud.shape[1], hidden_dim=50)               
                optimizer = torchAdam(model.parameters(), lr=1e-4)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Training loop
                # print('Begin training', file=sys.stderr)
                best_val_loss = float('inf')
                min_patience = 2
                patience = 10
                early_stop_counter = 0
                for epoch in range(100):
                    model.train()
                    epoch_loss = 0
                    for batch in train_loader:
                        
                        X_batch, lengths, y_batch = batch  # Get batch
                        X_batch, lengths, y_batch = X_batch.to(device), lengths.to('cpu'), y_batch.to(device)

                        optimizer.zero_grad()
                        outputs = model(X_batch, lengths)  # Pass lengths to model

                        loss = criterion_mci(outputs.squeeze(), y_batch.squeeze())
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)} \n", file=sys.stderr)
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_batch in val_loader:
                        
                            X_batch, lengths, y_batch = val_batch  # Get batch
                            X_batch, lengths, y_batch = X_batch.to(device), lengths.to('cpu'), y_batch.to(device)

                            outputs = model(X_batch, lengths)  # Pass lengths to model

                            loss = criterion_mci(outputs.squeeze(), y_batch.squeeze())
                            val_losses.append(loss.item())
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f} \n', file=sys.stderr)
                    if avg_val_loss < best_val_loss and epoch >= min_patience:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    elif epoch+1 > min_patience:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}', file=sys.stderr)
                        break
                model.load_state_dict(best_model)

            model.eval()
            torch.cuda.empty_cache()

            print(f"Getting testset embeddings \n", file=sys.stderr)
            with torch.no_grad():
                for test_batch in test_loader:
                
                    X_batch, lengths, y_batch = test_batch  # Get batch
                    X_batch, lengths, y_batch = X_batch.to(device), lengths.to('cpu'), y_batch.to(device)

                    test_outputs = model(X_batch, lengths)  # Pass lengths to model

            eval_df = []
            eval_df = pd.DataFrame({'tkdname': test_tkdnames})
            eval_df['output'] = torch.nn.functional.sigmoid(test_outputs.detach().to('cpu')).numpy()
            eval_df['Fold'] = fold_num
            eval_df['Ensemble Model'] = 'Whisper_RNN'
            all_preds.append(eval_df)


            del model


        elif training_config['architecture']=='WHISPER_AUDIO_MLPCLF':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device, file=sys.stderr)
            if training_config['target'] == "MCI":
                train_dataset = WhisperDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=32, 
                    shuffle=True,
                    drop_last=True)
                val_dataset = WhisperDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=32,
                    shuffle=True,
                    drop_last=True)
                test_dataset = WhisperDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False)

                
                # Define the model
                model = models.WhisperClassifier()               
                optimizer = torchAdam(model.parameters(), lr=1e-7)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Training loop
                # print('Begin training', file=sys.stderr)
                best_val_loss = float('inf')
                min_patience = 2
                patience = 10
                for epoch in range(100):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        inputs = [x.to(device) for x in inputs]
                        mci_targets = [t.to(device) for t in targets]

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets[0].float())
                        loss = loss_mci 

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)} \n", file=sys.stderr)
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            inputs = [x.to(device) for x in val_inputs]
                            mci_targets = [t.to(device) for t in val_targets]
                            val_outputs = model(inputs)
                            #Classification loss
                            loss_mci = criterion_mci(val_outputs.squeeze(), mci_targets[0].float())
                            
                            val_losses.append(loss_mci.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f} \n', file=sys.stderr)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    elif epoch+1 > min_patience:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                        break
                model.load_state_dict(best_model)

            model.eval()

            with torch.no_grad():
                print(f"Getting predictions \n", file=sys.stderr)
                eval_df = utils.get_predictions(
                        test_tkdnames, 
                        test_data_df, 
                        model, 
                        pytorch=True 
                    )
                print(f"Predictions calculated \n", file=sys.stderr)
                eval_df['Fold'] = fold_num
                eval_df['Ensemble Model'] = 'Whisper_Audio_Finetune'
                all_preds.append(eval_df)

            del model

        elif training_config['architecture']=='MULTIGPU_WHISPER_AUDIO':
            a = 1 
            def setup_ddp(rank, world_size):
                """Initialize DDP if using multiple GPUs"""
                if world_size > 1:
                    dist.init_process_group("nccl", rank=rank, world_size=world_size)
                    torch.cuda.set_device(rank)

            def cleanup_ddp():
                """Clean up DDP process group"""
                if dist.is_initialized():
                    dist.destroy_process_group()

            def train(rank, world_size, training_config, train_data_df, val_data_df, test_data_df, test_tkdnames, fold_num, all_preds):
                """Training loop with DDP support"""
                setup_ddp(rank, world_size)

                device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

                # Load Dataset
                train_dataset = WhisperDataset(dataframe=train_data_df)
                val_dataset = WhisperDataset(dataframe=val_data_df)
                test_dataset = WhisperDataset(dataframe=test_data_df)

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

                # Initialize Model
                model = models.WhisperClassifier().to(device)

                # Wrap with DDP if using multiple GPUs
                if world_size > 1:
                    model = DDP(model, device_ids=[rank])

                optimizer = optim.Adam(model.parameters(), lr=1e-7)
                criterion_mci = nn.BCEWithLogitsLoss()

                # Training loop
                best_val_loss = float('inf')
                min_patience = 2
                patience = 10
                early_stop_counter = 0

                for epoch in range(100):
                    model.train()
                    epoch_loss = 0

                    for inputs, targets in train_loader:
                        inputs = [x.to(device) for x in inputs]
                        mci_targets = [t.to(device) for t in targets]

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets[0].float())
                        loss = loss_mci

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()

                    # Average loss across processes if using DDP
                    avg_epoch_loss = epoch_loss / len(train_loader)
                    if world_size > 1:
                        loss_tensor = torch.tensor(avg_epoch_loss, device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        avg_epoch_loss = loss_tensor.item() / world_size

                    if rank == 0:  # Only let rank 0 print
                        print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {avg_epoch_loss:.4f} \n", file=sys.stderr)

                    # Validation
                    model.eval()
                    val_losses = []

                    with torch.no_grad():
                        for val_inputs, val_targets in val_loader:
                            val_inputs = [x.to(device) for x in val_inputs]
                            mci_targets = [t.to(device) for t in val_targets]
                            val_outputs = model(val_inputs)
                            loss_mci = criterion_mci(val_outputs.squeeze(), mci_targets[0].float())
                            val_losses.append(loss_mci.item())

                    avg_val_loss = np.mean(val_losses)

                    if world_size > 1:
                        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                        avg_val_loss = val_loss_tensor.item() / world_size

                    if rank == 0:
                        print(f'Validation Loss: {avg_val_loss:.4f} \n', file=sys.stderr)

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            best_model = model.state_dict()
                            early_stop_counter = 0
                        elif epoch + 1 > min_patience:
                            early_stop_counter += 1

                        if early_stop_counter >= patience:
                            print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                            break

                # Load best model
                if rank == 0:
                    model.load_state_dict(best_model)

                # Evaluation
                if rank == 0:
                    model.eval()
                    with torch.no_grad():
                        print(f"Getting predictions \n", file=sys.stderr)
                        eval_df = utils.get_predictions(
                            test_tkdnames,
                            test_data_df,
                            model,
                            pytorch=True
                        )
                        print(f"Predictions calculated \n", file=sys.stderr)
                        eval_df['Fold'] = fold_num
                        eval_df['Ensemble Model'] = 'Whisper_Audio_Finetune'
                        all_preds.append(eval_df)

                cleanup_ddp()

            #Actually run training with a switch case for using multiple GPUs
            world_size = torch.cuda.device_count()
            if world_size > 1:
                mp.spawn(train, args=(world_size, training_config, train_data_df, val_data_df, test_data_df, test_tkdnames, fold_num, all_preds), nprocs=world_size)
            else:
                train(rank=0, world_size=1, training_config=training_config, train_data_df=train_data_df, val_data_df=val_data_df, test_data_df=test_data_df, test_tkdnames=test_tkdnames, fold_num=fold_num, all_preds=all_preds)


        elif training_config['architecture']=='DA4AD':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_dataset = DA4ADDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=32, 
                    shuffle=True,
                    drop_last=True)
                test_dataset = DA4ADDataset(dataframe=test_data_df)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                
                # Define the model
                model = DA4ADClassifier()
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
                criterion_mci = nn.BCEWithLogitsLoss()
                criterion_gender = nn.BCEWithLogitsLoss()
                
                # Training loop
                for epoch in range(training_config['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for inputs, targets in train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        inputs = [x.to(device) for x in inputs]
                        mci_targets, gender_targets = [t.to(device) for t in targets]

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        # Compute loss
                        loss_mci = criterion_mci(outputs[0].squeeze(), mci_targets.float())
                        loss_gender = criterion_gender(outputs[1].squeeze(), gender_targets.float().squeeze())
                        loss = 0.9*loss_mci + 0.1*loss_gender
                        # loss = loss_mci 

                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")
                    

            model.to('cpu')
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True)
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model 

        elif training_config['architecture']=='Ying':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Iso_Filename']]
                    )
                test_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Iso_Filename']]
                    )
                train_dataset = dataloaders.YingDataset(dataframe=train_data_df)
                bert_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, #should be 16 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                w2v2_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=8, #should be 8 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                full_train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset),  
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )
                test_dataset = dataloaders.YingDataset(dataframe=test_data_df)
                full_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )

                # Define the model
                base_predictors = models.YingPredictors()
                wav_model = models.YingWavClassifier()
                wav_model.to(device)
                bert_model = models.YingBertClassifier()
                bert_model.to(device)
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                bert_optimizer = torchAdamW(
                    bert_model.parameters(), 
                    lr=3e-5)

                w2v2_optimizer = torchAdamW(
                    wav_model.parameters(), 
                    lr=1e-5)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Finetune BERT
                for epoch in range(3): #should be 3 epochs
                    bert_model.train()
                    epoch_loss = 0
                    for inputs, targets in bert_train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, _ = inputs
                        X_lang = [x.to(device) for x in X_lang]
                        input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                        attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)
                        
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                        bert_optimizer.zero_grad()
                        outputs = bert_model(input_ids, attention_masks)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                        loss_mci.backward()
                        bert_optimizer.step()
                        
                        epoch_loss += loss_mci.item()

                        
                    print(f"Epoch {epoch+1}/{3}, Loss: {epoch_loss/len(bert_train_loader)}")
                      
                    
                # Freeze BERT model parameters
                for param in bert_model.parameters():
                    param.requires_grad = False

                # Finetune W2V2
                for epoch in range(32): #should be 32 epochs
                    wav_model.train()
                    epoch_loss = 0
                    for inputs, targets in w2v2_train_loader:
                        # Move data to the appropriate device (e.g., GPU if available)
                        _, X_aud = inputs                       
                        audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                        
                        mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                        w2v2_optimizer.zero_grad()
                        outputs = wav_model(audio)

                        # Compute loss
                        loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                        loss_mci.backward()
                        w2v2_optimizer.step()
                        
                        epoch_loss += loss_mci.item()

                        
                    print(f"Epoch {epoch+1}/{32}, Loss: {epoch_loss/len(w2v2_train_loader)}")
                    
                    
                # Freeze W2V2 model parameters
                for param in wav_model.parameters():
                    param.requires_grad = False
                
                print('Models have been finetuned')

                bert_model.eval()
                wav_model.eval()

                bert_flag = 0.0
                w2v2_flag = 1.0
                iso_flag = 0.0

                #Get finetuned embeddings and fit SVM
                for inputs, targets in full_train_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    train_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model.extract_embeding(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model.extract_embeding(audio).cpu().detach().numpy()
                    train_iso_embeddings = iso_flag * train_iso_embeddings

                    train_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings,train_iso_embeddings),axis=1)
                    train_mean = train_embeddings.mean(axis=0)
                    train_std = train_embeddings.std(axis=0)
                    train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        print(f'Fitting {predictor_name}')
                        base_predictor.fit(train_embeddings,train_y_true)
                        print('Base predictor has been fit')
                
                #Test on testing split
                for inputs, targets in full_test_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    test_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model.extract_embeding(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model.extract_embeding(audio).cpu().detach().numpy()
                    test_iso_embeddings = iso_flag * test_iso_embeddings

                    test_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings, test_iso_embeddings),axis=1)
                    test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        y_pred = base_predictor.predict_proba(test_embeddings)
                        y_pred = y_pred[:,1]
                        print('Test Proba Predicted')
                    
                    test_scores = utils.scores(test_y_true, y_pred)
                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = args.model_name
                    for key, val in test_scores.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)

                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = 'Random'
                    rand_df = []
                    class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
                    for jj in range(100):
                        y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                        pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                        pred_score['Accuracy'] = accuracy_score(
                            test_data_df['dx'].values.astype(int),
                            (y_pred>class_balance_thresh).astype(int))
                        rand_df.append(pred_score)
                    cat_rand_df = pd.DataFrame(rand_df)
                    pred_score = cat_rand_df.mean()
                    for key, val in pred_score.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)           

                del bert_model
                del wav_model 

        elif training_config['architecture']=='EASIYing':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if training_config['target'] == "MCI":
                train_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Iso_Filename']]
                    )
                test_iso_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Iso_Filename']]
                    )
                train_dataset = dataloaders.YingDataset(dataframe=train_data_df)
                bert_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=16, #should be 16 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                w2v2_train_loader = DataLoader(
                    train_dataset, 
                    batch_size=8, #should be 8 
                    shuffle=True,
                    collate_fn=utils.ying_collate_fn
                    )
                full_train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=len(train_dataset),  
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )
                test_dataset = dataloaders.YingDataset(dataframe=test_data_df)
                full_test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    shuffle=False,
                    collate_fn=utils.ying_collate_fn
                    )

                # Define the model
                base_predictors = models.YingPredictors()
                wav_model = models.YingWavClassifier()
                wav_model.to(device)
                bert_model = models.YingBertClassifier()
                bert_model.to(device)
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                bert_optimizer = torchAdamW(
                    bert_model.parameters(), 
                    lr=3e-5)

                w2v2_optimizer = torchAdamW(
                    wav_model.parameters(), 
                    lr=1e-5)
                criterion_mci = nn.BCEWithLogitsLoss()
                
                # Finetune BERT
                # for epoch in range(3): #should be 3 epochs
                #     bert_model.train()
                #     epoch_loss = 0
                #     for inputs, targets in bert_train_loader:
                #         # Move data to the appropriate device (e.g., GPU if available)
                #         X_lang, _ = inputs
                #         X_lang = [x.to(device) for x in X_lang]
                #         input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                #         attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)
                        
                #         mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                #         bert_optimizer.zero_grad()
                #         outputs = bert_model(input_ids, attention_masks)

                #         # Compute loss
                #         loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                #         loss_mci.backward()
                #         bert_optimizer.step()
                        
                #         epoch_loss += loss_mci.item()

                        
                #     print(f"Epoch {epoch+1}/{3}, Loss: {epoch_loss/len(bert_train_loader)}")
                      
                    
                # Freeze BERT model parameters
                for param in bert_model.parameters():
                    param.requires_grad = False

                # Finetune W2V2
                # for epoch in range(32): #should be 32 epochs
                #     wav_model.train()
                #     epoch_loss = 0
                #     for inputs, targets in w2v2_train_loader:
                #         # Move data to the appropriate device (e.g., GPU if available)
                #         _, X_aud = inputs                       
                #         audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                        
                #         mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)

                #         w2v2_optimizer.zero_grad()
                #         outputs = wav_model(audio)

                #         # Compute loss
                #         loss_mci = criterion_mci(outputs.squeeze(), mci_targets.float())

                #         loss_mci.backward()
                #         w2v2_optimizer.step()
                        
                #         epoch_loss += loss_mci.item()

                        
                #     print(f"Epoch {epoch+1}/{32}, Loss: {epoch_loss/len(w2v2_train_loader)}")
                    
                    
                # Freeze W2V2 model parameters
                for param in wav_model.parameters():
                    param.requires_grad = False
                
                print('Models have been finetuned')

                bert_model.eval()
                wav_model.eval()

                bert_flag = 0.0
                w2v2_flag = 0.0
                iso_flag = 1.0

                #Get finetuned embeddings and fit SVM
                for inputs, targets in full_train_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    train_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model(audio).cpu().detach().numpy()
                    train_iso_embeddings = iso_flag * train_iso_embeddings

                    train_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings,train_iso_embeddings),axis=1)
                    train_mean = train_embeddings.mean(axis=0)
                    train_std = train_embeddings.std(axis=0)
                    train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7) 

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        print(f'Fitting {predictor_name}')
                        base_predictor.fit(train_embeddings,train_y_true)
                        print('Base predictor has been fit')
                
                #Test on testing split
                for inputs, targets in full_test_dataloader:
                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang, X_aud = inputs 
                    
                    X_lang = [x.to(device) for x in X_lang]
                    input_ids = torch.vstack([item.input_ids for item in X_lang]).to(device)
                    attention_masks = torch.vstack([item.attention_mask for item in X_lang]).to(device)

                    audio = torch.stack([x.to(device) for x in X_aud]).to(device)
                    
                    mci_targets = torch.tensor([t.to(device) for t in targets]).to(device)
                    test_y_true = mci_targets.cpu().detach().numpy()

                    bert_embeddings = bert_flag * bert_model(input_ids, attention_masks).cpu().detach().numpy()
                    w2v2_embeddings = w2v2_flag * wav_model(audio).cpu().detach().numpy()
                    test_iso_embeddings = iso_flag * test_iso_embeddings

                    test_embeddings = np.concatenate((bert_embeddings,w2v2_embeddings, test_iso_embeddings),axis=1)
                    test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)

                    for predictor_name, base_predictor in base_predictors.base_predictors.items():
                        y_pred = base_predictor.predict_proba(test_embeddings)
                        y_pred = y_pred[:,1]
                        print('Test Proba Predicted')
                    
                    test_scores = utils.scores(test_y_true, y_pred)
                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = args.model_name
                    for key, val in test_scores.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)

                    temp_df = {}
                    temp_df['Outer Fold'] = ii
                    temp_df['Ensemble Model'] = 'Random'
                    rand_df = []
                    class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
                    for jj in range(100):
                        y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                        pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                        pred_score['Accuracy'] = accuracy_score(
                            test_data_df['dx'].values.astype(int),
                            (y_pred>class_balance_thresh).astype(int))
                        rand_df.append(pred_score)
                    cat_rand_df = pd.DataFrame(rand_df)
                    pred_score = cat_rand_df.mean()
                    for key, val in pred_score.items():
                        temp_df[key] = val
                    all_scores.append(temp_df)           

                del bert_model
                del wav_model

        elif training_config['architecture']=='MISA':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if training_config['target'] == "MCI":
                train_dataset = MISADataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.misa_collate_fn)
                val_dataset = MISADataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.misa_collate_fn)
                test_dataset = MISADataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.misa_collate_fn)
                
                # Define the model
                sample_lang = np.load(train_data_df['Language_Filename'][0])
                sample_aud = np.load(train_data_df['Audio_Filename'][0])
                model = MISAClassifier(
                    lang_dim = sample_lang.shape[1],
                    aud_dim = sample_aud.shape[1]
                )
                
                # Define the loss functions and optimizer
                class_weights = compute_class_weight(
                    class_weight='balanced', 
                    classes=np.unique(train_data_df['dx']), 
                    y=train_data_df['dx'].values
                )
                weights_dict = {index: value for index, value in enumerate(class_weights)}
                
                optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
                criterion_mci = nn.BCEWithLogitsLoss()
                criterion_mse = utils.MISAMSE()
                criterion_diff = utils.MISADiffLoss()
                criterion_cmd = utils.MISACMD()
                
                # Training loop
                best_val_loss = float('inf')
                patience = 30
                for epoch in range(training_config['n_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch in train_loader:
                        # Unpack the batch
                        [X_lang, X_aud_padded], mci_targets, [X_lang_lengths, X_aud_lengths] = batch

                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                        X_aud_lengths = X_aud_lengths.to('cpu')
                        mci_targets = mci_targets.to(device)

                        # Pack the padded sequences
                        X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths.cpu(), batch_first=True, enforce_sorted=False).to(device)

                        # Zero the gradients
                        optimizer.zero_grad()

                        # Prepare inputs for the model
                        inputs = [X_lang, X_aud_packed]
                        
                        # Forward pass
                        outputs = model(inputs)
                        #Classification loss
                        loss_mci = criterion_mci(outputs['mci_output'].squeeze(), mci_targets.float())
                        #Reconstruction loss
                        loss_recon = 0.5*criterion_mse(outputs['lang_reconstruction'],outputs['lang_proj'])
                        loss_recon += 0.5*criterion_mse(outputs['aud_reconstruction'],outputs['aud_proj'])
                        #Similarity (CMD) loss
                        loss_sim = criterion_cmd(outputs['lang_public'],outputs['aud_public'],5)
                        #Difference loss
                        loss_diff = 0.33*criterion_diff(outputs['lang_public'],outputs['lang_private'])
                        loss_diff += 0.33*criterion_diff(outputs['aud_public'],outputs['aud_private'])
                        loss_diff += 0.33*criterion_diff(outputs['lang_private'],outputs['aud_private'])

                        loss = loss_mci + 0.7*loss_sim + loss_diff + loss_recon 
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")
                    print(loss_mci)

                    model.eval()
                    val_losses = []
                    for val_batch in val_loader:
                        # Unpack the val_batch
                        [X_lang, X_aud_padded], val_mci_targets, [X_lang_lengths, X_aud_lengths] = val_batch

                        # Move data to the appropriate device (e.g., GPU if available)
                        X_lang, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                        X_aud_lengths = X_aud_lengths.to('cpu')
                        val_mci_targets = val_mci_targets.to(device)

                        # Pack the padded sequences
                        X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(device)
                        val_inputs = [X_lang, X_aud_packed]

                        with torch.no_grad():
                            val_outputs = model(val_inputs)
                            #Classification loss
                            loss_mci = criterion_mci(val_outputs['mci_output'].squeeze(), val_mci_targets.float())
                            #Reconstruction loss
                            loss_recon = 0.5*criterion_mse(val_outputs['lang_reconstruction'],val_outputs['lang_proj'])
                            loss_recon += 0.5*criterion_mse(val_outputs['aud_reconstruction'],val_outputs['aud_proj'])
                            #Similarity (CMD) loss
                            loss_sim = criterion_cmd(val_outputs['lang_public'],val_outputs['aud_public'],5)
                            #Difference loss
                            loss_diff = 0.33*criterion_diff(val_outputs['lang_public'],val_outputs['lang_private'])
                            loss_diff += 0.33*criterion_diff(val_outputs['aud_public'],val_outputs['aud_private'])
                            loss_diff += 0.33*criterion_diff(val_outputs['lang_private'],val_outputs['aud_private'])

                            loss = loss_mci + 0.7*loss_sim + loss_diff + loss_recon
                            val_losses.append(loss_mci.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    print(f'Validation Loss: {avg_val_loss:.4f}')
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model = model.state_dict()
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                        break
                model.load_state_dict(best_model)
                        
            model.to('cpu')
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True, modality='MISA')
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True, modality='MISA')

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model 

        elif training_config['architecture']=='MFN':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if training_config['modality'] == "MFN":
                train_dataset = MFNDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.mfn_collate_fn)
                val_dataset = MFNDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                test_dataset = MFNDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                
                # Define the model
                model = MFNClassifier()
            elif training_config['modality'] == "MFN EASI-COG":
                train_dataset = MFNDataset(dataframe=train_data_df)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=training_config['batch_size'], 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn = utils.mfn_collate_fn)
                val_dataset = MFNDataset(dataframe=val_data_df)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                test_dataset = MFNDataset(dataframe=test_data_df)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset),
                    shuffle=True, 
                    collate_fn = utils.mfn_collate_fn)
                
                # Define the model
                model = models.EASICOGMFNClassifier()
            
            # Define the loss functions and optimizer
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(train_data_df['dx']), 
                y=train_data_df['dx'].values
            )
            weights_dict = {index: value for index, value in enumerate(class_weights)}
            
            optimizer = torchAdam(model.parameters(), lr=training_config['learning_rate'])
            criterion_mci = nn.BCEWithLogitsLoss()
            criterion_mse = utils.MISAMSE()
            criterion_diff = utils.MISADiffLoss()
            criterion_cmd = utils.MISACMD()
            
            # Training loop
            best_val_loss = float('inf')
            patience = 30
            for epoch in range(training_config['n_epochs']):
                model.train()
                epoch_loss = 0
                for batch in train_loader:
                    # Unpack the batch
                    [X_lang_padded, X_aud_padded], mci_targets, [X_lang_lengths, X_aud_lengths] = batch

                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang_padded, X_aud_padded = X_lang_padded.to(device), X_aud_padded.to(device)
                    X_lang_lengths, X_aud_lengths = X_lang_lengths.to('cpu'), X_aud_lengths.to('cpu')
                    mci_targets = mci_targets.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Prepare inputs for the model
                    inputs = [X_lang_padded, X_aud_padded]
                    lengths = [X_lang_lengths, X_aud_lengths]

                    #Forward pass
                    outputs = model([inputs, lengths])

                    #Classification loss
                    loss = criterion_mci(outputs.squeeze(), mci_targets.float())
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                print(f"Epoch {epoch+1}/{training_config['n_epochs']}, Loss: {epoch_loss/len(train_loader)}")

                model.eval()
                val_losses = []
                for val_batch in val_loader:
                    # Unpack the val_batch
                    [X_lang, X_aud_padded], val_mci_targets, [X_lang_lengths, X_aud_lengths] = val_batch

                    # Move data to the appropriate device (e.g., GPU if available)
                    X_lang_padded, X_aud_padded = X_lang.to(device), X_aud_padded.to(device)
                    X_lang_lengths, X_aud_lengths = X_lang_lengths.to('cpu'), X_aud_lengths.to('cpu')
                    val_mci_targets = val_mci_targets.to(device)

                    # Pack the padded sequences
                    val_inputs = [X_lang_padded, X_aud_padded]
                    val_lengths = [X_lang_lengths, X_aud_lengths]

                    with torch.no_grad():
                        val_outputs = model([val_inputs, val_lengths])
                        #Classification loss
                        loss = criterion_mci(val_outputs.squeeze(), val_mci_targets.float())
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                print(f'Validation Loss: {avg_val_loss:.4f}')
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    print(f'Early stopping after epoch {epoch+1} with validation loss {best_val_loss:.4f}')
                    break
            model.load_state_dict(best_model)
                        
            model.to('cpu')
            model.device='cpu'
            train_scores = utils.get_scores(train_tkdnames, train_data_df, model, pytorch=True, modality='MFN')
            test_scores = utils.get_scores(test_tkdnames, test_data_df, model, pytorch=True, modality='MFN')

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = args.model_name
            for key, val in test_scores.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

            del model

        elif training_config['architecture']=='FARZANA':
            language_flag = 0.0
            prosody_flag = 1.0
            if training_config['target'] == "MCI":
                lang_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Language_Filename']]
                    )*language_flag
                aud_train_embeddings = np.vstack(
                    [np.load(filepath) for filepath in train_data_df['Audio_Filename']]
                    )*prosody_flag
                train_embeddings = np.concatenate((lang_train_embeddings,aud_train_embeddings), axis=-1)
                train_mean = train_embeddings.mean(axis=0)
                train_std = train_embeddings.std(axis=0)
                train_embeddings = (train_embeddings - train_mean) / (train_std + 1e-7)
                train_y_true = train_data_df['dx'].values.astype(int)
                train_y_true_eval = train_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
                
                lang_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Language_Filename']]
                    )*language_flag
                aud_test_embeddings = np.vstack(
                    [np.load(filepath) for filepath in test_data_df['Audio_Filename']]
                    )*prosody_flag
                test_embeddings = np.concatenate((lang_test_embeddings,aud_test_embeddings), axis=-1)
                test_embeddings = (test_embeddings - train_mean) / (train_std + 1e-7)
                test_y_true = test_data_df['dx'].values.astype(int)
                test_y_true_eval = test_data_df[['tkdname','dx']].drop_duplicates()['dx'].values.astype(int)
            
            data_train = {training_config['modality']: train_embeddings}
            data_test = {training_config['modality']: test_embeddings}

            base_predictors = {
                    'LR': LogisticRegression(),
                    'SVM': SVC(probability=True),
                    }

            for predictor_name, base_predictor in base_predictors.items():
                print(f'Fitting {predictor_name}')
                base_predictor.fit(train_embeddings,train_y_true)
                pred_score = utils.get_scores(
                    test_tkdnames, 
                    test_data_df, 
                    base_predictor, 
                    EI=True, 
                    modality=training_config['modality'],
                    data_mean=train_mean, 
                    data_std=train_std)

                temp_df = {}
                temp_df['Outer Fold'] = ii
                temp_df['Ensemble Model'] = predictor_name
                for key, val in pred_score.items():
                    temp_df[key] = val
                all_scores.append(temp_df)

            _, tkdname_indices = utils.average_ei_model_output_per_tkdname(
                test_tkdnames, 
                test_data_df, 
                base_predictor, 
                modality=training_config['modality'])

            temp_df = {}
            temp_df['Outer Fold'] = ii
            temp_df['Ensemble Model'] = 'Random'
            rand_df = []
            class_balance_thresh = (1-(train_data_df['dx'].values.sum()/train_data_df['dx'].values.shape[0]))
            for jj in range(100):
                y_pred = np.random.uniform(size=test_data_df['dx'].values.shape)
                pred_score = utils.scores(test_data_df['dx'].values.astype(int), y_pred)
                pred_score['Accuracy'] = accuracy_score(
                    test_data_df['dx'].values.astype(int),
                    (y_pred>class_balance_thresh).astype(int))
                rand_df.append(pred_score)
            cat_rand_df = pd.DataFrame(rand_df)
            pred_score = cat_rand_df.mean()
            for key, val in pred_score.items():
                temp_df[key] = val
            all_scores.append(temp_df)

    pred_df = pd.concat(all_preds)
    pred_path = training_config['scores_outdir'].replace('reports','predictions')
    os.makedirs(pred_path, exist_ok=True)
    preds_outpath = os.path.join(pred_path,args.model_name+'.csv')
    pred_df.to_csv(preds_outpath)

