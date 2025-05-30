import sys
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import noisereduce as nr 
import pyloudnorm as pyln 
import tensorflow as tf
import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Concatenate, Conv1D
from keras.layers import MaxPooling1D, Reshape, BatchNormalization, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten, LayerNormalization
from keras.optimizers import Adam, AdamW
from keras.models import load_model
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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

def pad_and_truncate_bclstm(input_tensor, max_len=50, embedding_dim=300):
    input_tensor = torch.tensor(input_tensor)
    # Truncate if longer than max_len
    if input_tensor.shape[0] > max_len:
        input_tensor = input_tensor[:max_len, :]
    
    # Pad with zeros if shorter than max_len
    if input_tensor.shape[0] < max_len:
        padding = torch.zeros((max_len - input_tensor.shape[0], embedding_dim))
        input_tensor = torch.cat((input_tensor, padding), dim=0)
    
    return input_tensor

def ying_collate_fn(batch):
    # Separate the data and labels
    X_lang = [item[0][0] for item in batch]
    X_aud = [item[0][1] for item in batch]
    y_batch = [torch.tensor(item[1][0]) for item in batch]

    # Convert labels to tensor
    y_tensor = torch.tensor(y_batch)

    return [X_lang, X_aud], y_tensor

def bclstm_collate_lang_fn(batch):
    # Separate the data and labels
    X_lang_batch = [item[0][0] for item in batch] #[batch, num_sentences, num_words, feat_size]
    y_batch = [torch.tensor(item[1][0]) for item in batch]

    # Convert labels to tensor
    y_tensor = torch.tensor(y_batch)

    return X_lang_batch, y_tensor

def bclstm_collate_fn(batch):
    # Separate the data and labels
    X_lang_batch = [item[0][0] for item in batch] #[batch, num_sentences, num_words, feat_size]
    X_aud_batch = [item[0][1] for item in batch]
    y_batch = [torch.tensor(item[1][0]) for item in batch]

    # Calculate lengths of the sequences before padding
    sentence_counts = torch.tensor([len(x) for x in X_lang_batch])

    # Convert labels to tensor
    y_tensor = torch.tensor(y_batch)

    return [X_lang_batch, X_aud_batch], y_tensor, sentence_counts

def whisperrnn_collate_fn(batch):
    X_aud_batch, y_batch = zip(*batch)

    # Ensure X_aud_batch is a list of tensors, each of shape [seq_len, 1024]
    lengths = torch.tensor([x.shape[0] for x in X_aud_batch])  # Sequence lengths

    # Pad sequences
    X_aud_padded = pad_sequence(X_aud_batch, batch_first=True, padding_value=0)  # Shape: [batch, max_seq_len, 1024]

    # Convert labels to tensor
    y_batch = torch.tensor(y_batch, dtype=torch.float32)

    return X_aud_padded, lengths, y_batch

def mfn_collate_fn(batch):
    # Separate the data and labels
    lang_flag = 1.0
    aud_flag = 0.0
    X_lang_batch = [torch.tensor(lang_flag*item[0][0]) for item in batch]
    X_aud_batch = [torch.tensor(aud_flag*item[0][1]) for item in batch]
    y_batch = [torch.tensor(item[1][0]) for item in batch]

    # Calculate lengths of the sequences before padding
    X_lang_lengths = torch.tensor([len(x) for x in X_lang_batch])
    X_aud_lengths = torch.tensor([len(x) for x in X_aud_batch])

    # Pad sequences for X_lang and X_aud
    X_lang_padded = pad_sequence(X_lang_batch, batch_first=True)
    X_aud_padded = pad_sequence(X_aud_batch, batch_first=True)

    # Convert labels to tensor
    y_tensor = torch.tensor(y_batch)

    return [X_lang_padded, X_aud_padded], y_tensor, [X_lang_lengths, X_aud_lengths]


def misa_collate_fn(batch):
    # Separate the data and labels
    X_lang_batch = [torch.tensor(item[0][0]) for item in batch]
    X_aud_batch = [torch.tensor(item[0][1]) for item in batch]
    y_batch = [torch.tensor(item[1][0]) for item in batch]

    # Calculate lengths of the sequences before padding
    X_lang_lengths = torch.tensor([len(x) for x in X_lang_batch])
    X_aud_lengths = torch.tensor([len(x) for x in X_aud_batch])

    # Pad sequences for X_lang and X_aud
    X_lang_padded = pad_sequence(X_lang_batch, batch_first=True)
    X_aud_padded = pad_sequence(X_aud_batch, batch_first=True)

    # Convert labels to tensor
    y_tensor = torch.tensor(y_batch)

    return [X_lang_padded, X_aud_padded], y_tensor, [X_lang_lengths, X_aud_lengths]

def denoise_audio(audio, sr, stationary=True, peak_normalize=True):
    if peak_normalize:
        audio /= np.max(np.abs(audio))
    filtered_audio = nr.reduce_noise(y=audio,sr=sr,stationary=stationary)

    return filtered_audio

def loudnorm_audio(audio, meter, target_loudness=-23.0):
    input_loundess = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, input_loundess, target_loudness)

    return normalized_audio

def scores(y_true, y_pred, beta=1, metric_to_maximise="fscore", verbose=0):

    # Get Fmax
    f_measure_minor = fmeasure_score(y_true, y_pred, pos_label=1)

    # Get F on majority class using the threshold of Fmax on the minority class
    f_measure_major = fmeasure_score(y_true, y_pred, pos_label=0, thres=1-f_measure_minor['thres'])

    # AUC
    auc = roc_auc_score(y_true, y_pred)

    # Store all scores
    scores_dict = {
      "Fmax (minority)": f_measure_minor['F'],
      "F (majority)": f_measure_major['F'],
      "AUC": auc,
      "Threshold": f_measure_minor['thres'], 
      "Precision (minority)": f_measure_minor['P'],
      "Recall (minority)": f_measure_minor['R']
                             }  # dictionary of (score, threshold)

    # Print all scores (if you want)
    if verbose > 0:
        for metric_name, score in scores_dict.items():
            print(metric_name + ": ", score[0])

    return scores_dict

def fmeasure_score(labels, predictions, thres=None,
                    beta = 1.0, pos_label = 1, thres_same_cls = False):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if pos_label == 0:
        labels = 1-np.array(labels)
        predictions = 1-np.array(predictions)

    if thres is None:  # calculate Fmax here
        np.seterr(divide='ignore', invalid='ignore')
        precision, recall, threshold = precision_recall_curve(labels, predictions,)

        fs = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        fmax_point = np.where(fs==np.nanmax(fs))[0]
        p_maxes = precision[fmax_point]
        r_maxes = recall[fmax_point]
        pr_diff = np.abs(p_maxes - r_maxes)
        balance_fmax_point = np.where(pr_diff == min(pr_diff))[0]
        p_max = p_maxes[balance_fmax_point[0]]
        r_max = r_maxes[balance_fmax_point[0]]
        opt_threshold = threshold[fmax_point][balance_fmax_point[0]]

        return {'F':np.nanmax(fs), 'thres':opt_threshold, 'P':p_max, 'R':r_max, 'PR-curve': [precision, recall]}

    else:  # calculate fmeasure for specific threshold
        binary_predictions = np.array(predictions)
        if thres_same_cls:
            binary_predictions[binary_predictions >= thres] = 1.0
            binary_predictions[binary_predictions < thres] = 0.0
        else:
            binary_predictions[binary_predictions > thres] = 1.0
            binary_predictions[binary_predictions <= thres] = 0.0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(labels,
                                                                        binary_predictions,
                                                                        average='binary')
        return {'P':precision, 'R':recall, 'F':fmeasure}

def sample(X, y, strategy, random_state):
    if strategy is None:
        X_resampled, y_resampled = X, y
    elif strategy == "undersampling":  # define sampler
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "oversampling":
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == 'hybrid':
        y_pos = float(sum(y==1))
        y_total = y.shape[0]
        if (y_pos/y_total) < 0.5:
            y_min_count = y_pos
            y_maj_count = (y_total - y_pos)
            maj_class = 0
        else:
            y_maj_count = y_pos
            y_min_count = (y_total - y_pos)
            maj_class = 1
        rus = RandomUnderSampler(random_state=random_state,
                                sampling_strategy=y_min_count/(y_total/2))
        ros = RandomOverSampler(random_state=random_state,
                                sampling_strategy=(y_total/2)/y_maj_count)
        X_maj, y_maj = rus.fit_resample(X=X, y=y)
        X_maj = X_maj[y_maj==maj_class]
        y_maj = y_maj[y_maj==maj_class]
        X_min, y_min = ros.fit_resample(X=X, y=y)
        X_min = X_min[y_min!=maj_class]
        y_min = y_min[y_min!=maj_class]
        X_resampled = np.concatenate([X_maj, X_min])
        y_resampled = np.concatenate([y_maj, y_min])

    if (strategy == "undersampling") or (strategy == "oversampling"):
        X_resampled, y_resampled = sampler.fit_resample(X=X, y=y)
    return X_resampled, y_resampled

def uar_score(y_pred,y_test):
  cm = confusion_matrix(y_test, y_pred)
  tp, tn = cm[0, 0], cm[1, 1]
  fn, fp = cm[1, 0], cm[0, 1]

  sensitivity = tp / (tp + fn + 1e-7)
  specificity = tn / (fp + tn + 1e-7)

  uar = (specificity + sensitivity)/2.0
  return uar

def average_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            npy_filenames = inputs['Filename'].tolist()

            X = []
            demographics = []
            for filename in npy_filenames:
                data = np.load(filename)

                # Extract demographics from the dataframe
                demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                # Append data and demographics to the lists
                X.append(data)
                demographics.append(demographics_row)

            all_X.extend(X)
            all_demographics.extend(demographics)
            tkdname_indices[tkdname] = (index, index + len(X))
            index += len(X)

    all_X = np.vstack(all_X)
    all_demographics = np.vstack(all_demographics)

    # Run all inputs through the neural network model
    outputs = model.predict([all_X, all_demographics], verbose=0)

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs = outputs[start:end]
        average_output = np.mean(tkd_outputs, axis=0)
        output_per_tkdname[tkdname] = average_output

    return output_per_tkdname

def average_pytorch_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.eval()
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    outputs_mci = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            npy_filenames = inputs['Filename'].tolist()

            X = []
            for filename in npy_filenames:
                data = np.load(filename)

                # Extract demographics from the dataframe

                # Append data and demographics to the lists
                X.append(data)

            # all_X.extend(X)
            tkdname_indices[tkdname] = (index, index + len(X))
            index += len(X)

            stacked_X = torch.tensor(np.vstack(X), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model([stacked_X])
            outputs_mci.extend(torch.sigmoid(outputs[0].cpu()).numpy())
            

    # all_X = np.vstack(all_X)
    # all_X_tensor = torch.tensor(all_X, dtype=torch.float32).to(device)

    # Run all inputs through the neural network model
    # with torch.no_grad():
    #     outputs = model([all_X_tensor, all_demographics_tensor])

    # outputs_mci = torch.sigmoid(outputs[0].cpu()).numpy()

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs_mci = outputs_mci[start:end]

        # Average the outputs
        average_output_mci = np.mean(tkd_outputs_mci, axis=0)
        
        output_per_tkdname[tkdname] = average_output_mci

    return output_per_tkdname, tkdname_indices

def pytorch_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    outputs_mci = []
    all_tkdnames = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]
        

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])
            npy_filenames = inputs['Filename'].tolist()
            X = []
            for filename in npy_filenames:
                all_tkdnames.append(tkdname)
                
                data = np.load(filename)

                # Extract demographics from the dataframe

                # Append data and demographics to the lists
                X.append(data)

            # all_X.extend(X)
            tkdname_indices[tkdname] = (index, index + len(X))
            index += len(X)

            stacked_X = torch.tensor(np.vstack(X), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model([stacked_X])
            outputs_mci.append(torch.sigmoid(outputs.cpu()).numpy())


    outputs_mci = np.squeeze(np.vstack(outputs_mci))
    eval_df = pd.DataFrame({'tkdname': all_tkdnames, 'output':outputs_mci})
    # eval_df['output'] = outputs_mci

    return eval_df

def average_BCLSTM_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_device = 'cpu'
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    outputs_mci = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            lang_filenames = inputs['Language_Filename'].tolist()
            aud_filenames = inputs['Audio_Filename'].tolist()

            lang_X = []
            aud_X = []
            for lang_filename, aud_filename in zip(lang_filenames,aud_filenames):
                lang_data = np.load(lang_filename)
                aud_data = np.load(aud_filename)

                # Extract demographics from the dataframe

                # Append data and demographics to the lists
                lang_X.append(lang_data)
                aud_X.append(aud_data)

            # all_X.extend(X)
            tkdname_indices[tkdname] = (index, index + 1)
            index += 1
       
            # lang_lengths = torch.tensor([len(x) for x in lang_X])
            # aud_lengths = torch.tensor([len(x) for x in aud_X])
            # lang_X = [torch.tensor(item, dtype=torch.float32) for item in lang_X]
            # aud_X = [torch.tensor(item, dtype=torch.float32) for item in aud_X]

            X_lang = [torch.tensor(item, dtype=torch.float) for item in lang_X]
            X_aud = [torch.tensor(item, dtype=torch.float) for item in aud_X]

            # Calculate lengths of the sequences before padding
            X_lang_lengths = torch.tensor([len(x) for x in X_lang]).to('cpu')
            X_aud_lengths = torch.tensor([len(x) for x in X_aud]).to('cpu')

            # Pad sequences for X_lang and X_aud
            X_lang_padded = nn.utils.rnn.pad_sequence(X_lang, batch_first=True).to(eval_device)
            X_aud_padded = nn.utils.rnn.pad_sequence(X_aud, batch_first=True).to(eval_device)
            X_lang_packed = nn.utils.rnn.pack_padded_sequence(X_lang_padded, X_lang_lengths, batch_first=True, enforce_sorted=False).to(eval_device)
            X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False).to(eval_device)

            # Prepare inputs for the model
            inputs = [X_lang_packed, X_aud_packed]
            lengths = [X_lang_lengths, X_aud_lengths]

            with torch.no_grad():
                outputs = model([inputs, lengths])
            outputs_mci.extend(torch.sigmoid(outputs.cpu()).numpy())
            

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs_mci = outputs_mci[start:end]

        # Average the outputs
        average_output_mci = np.mean(tkd_outputs_mci, axis=0)
        
        output_per_tkdname[tkdname] = average_output_mci

    return output_per_tkdname, tkdname_indices

def average_MFN_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_device = 'cpu'
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    outputs_mci = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            lang_filenames = inputs['Language_Filename'].tolist()
            aud_filenames = inputs['Audio_Filename'].tolist()

            lang_X = []
            aud_X = []
            for lang_filename, aud_filename in zip(lang_filenames,aud_filenames):
                lang_data = np.load(lang_filename)
                aud_data = np.load(aud_filename)

                # Extract demographics from the dataframe

                # Append data and demographics to the lists
                lang_X.append(lang_data)
                aud_X.append(aud_data)

            # all_X.extend(X)
            tkdname_indices[tkdname] = (index, index + len(lang_X))
            index += len(lang_X)
       
            lang_lengths = torch.tensor([len(x) for x in lang_X])
            aud_lengths = torch.tensor([len(x) for x in aud_X])
            lang_X = [torch.tensor(item, dtype=torch.float32) for item in lang_X]
            aud_X = [torch.tensor(item, dtype=torch.float32) for item in aud_X]
            # Pad sequences for X_lang and X_aud
            X_lang_padded = pad_sequence(lang_X, batch_first=True)
            X_aud_padded = pad_sequence(aud_X, batch_first=True)
            inputs = [X_lang_padded.to(eval_device), X_aud_padded.to(eval_device)]
            lengths = [lang_lengths.to(eval_device), aud_lengths.to(eval_device)]
            # X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False)
            # stacked_aud_X = torch.tensor(np.stack(aud_X), dtype=torch.float32).to(eval_device)
            with torch.no_grad():
                outputs = model([inputs, lengths])
            outputs_mci.extend(torch.sigmoid(outputs.cpu()).numpy())
            

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs_mci = outputs_mci[start:end]

        # Average the outputs
        average_output_mci = np.mean(tkd_outputs_mci, axis=0)
        
        output_per_tkdname[tkdname] = average_output_mci

    return output_per_tkdname, tkdname_indices

def average_MISA_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_device = 'cpu'
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    outputs_mci = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            lang_filenames = inputs['Language_Filename'].tolist()
            aud_filenames = inputs['Audio_Filename'].tolist()

            lang_X = []
            aud_X = []
            for lang_filename, aud_filename in zip(lang_filenames,aud_filenames):
                lang_data = np.load(lang_filename)
                aud_data = np.load(aud_filename)

                # Extract demographics from the dataframe

                # Append data and demographics to the lists
                lang_X.append(lang_data)
                aud_X.append(aud_data)

            # all_X.extend(X)
            tkdname_indices[tkdname] = (index, index + len(lang_X))
            index += len(lang_X)

            stacked_lang_X = torch.tensor(np.stack(lang_X), dtype=torch.float32).to(eval_device)
            
            X_aud_lengths = torch.tensor([len(x) for x in aud_X])
            aud_X = [torch.tensor(item, dtype=torch.float32) for item in aud_X]
            # Pad sequences for X_lang and X_aud
            X_aud_padded = pad_sequence(aud_X, batch_first=True)
            X_aud_packed = nn.utils.rnn.pack_padded_sequence(X_aud_padded, X_aud_lengths, batch_first=True, enforce_sorted=False)
            # stacked_aud_X = torch.tensor(np.stack(aud_X), dtype=torch.float32).to(eval_device)
            with torch.no_grad():
                outputs = model([stacked_lang_X,X_aud_packed])
            outputs_mci.extend(torch.sigmoid(outputs['mci_output'].cpu()).numpy())
            

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs_mci = outputs_mci[start:end]

        # Average the outputs
        average_output_mci = np.mean(tkd_outputs_mci, axis=0)
        
        output_per_tkdname[tkdname] = average_output_mci

    return output_per_tkdname, tkdname_indices

def average_ei_model_output_per_tkdname(tkdnames, w2v2_df, model, modality=None, data_mean=None, data_std=None):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            

            X = []
            demographics = []
            if modality == 'MULTIMODAL' or modality == 'FARZANA':
                language_flag = 1.0
                prosody_flag = 1.0
                lang_filenames = inputs['Language_Filename'].tolist()
                aud_filenames = inputs['Audio_Filename'].tolist()
                for lang_filename, aud_filename in zip(lang_filenames, aud_filenames):
                    lang_data = np.load(lang_filename)*language_flag
                    aud_data = np.load(aud_filename)*prosody_flag
                    data = np.concatenate((lang_data,aud_data), axis=-1)

                    # Extract demographics from the dataframe
                    demographics_row = inputs[inputs['Language_Filename'] == lang_filename][['age', 'sex', 'Language']].values[0]

                    # Append data and demographics to the lists
                    X.append(data)
                    demographics.append(demographics_row)
            else:
                npy_filenames = inputs['Filename'].tolist()
                for filename in npy_filenames:
                    data = np.load(filename)

                    # Extract demographics from the dataframe
                    demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                    # Append data and demographics to the lists
                    X.append(data)
                    demographics.append(demographics_row)

            all_X.extend(X)
            all_demographics.extend(demographics)
            tkdname_indices[tkdname] = (index, index + len(X))
            index += len(X)
    all_X = np.vstack(all_X)
    if data_mean is not None and data_std is not None:
        all_X = (all_X-data_mean) / (data_std + 1e-7)

    # Run all inputs through the neural network model
    outputs = model.predict_proba(all_X)
    outputs = outputs[:,1]

    # Distribute outputs to corresponding tkdnames
    for tkdname, (start, end) in tkdname_indices.items():
        tkd_outputs = outputs[start:end]
        average_output = np.mean(tkd_outputs, axis=0)
        output_per_tkdname[tkdname] = average_output

    return output_per_tkdname, tkdname_indices

def ei_model_output_per_tkdname(tkdnames, w2v2_df, model, modality=None, data_mean=None, data_std=None):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}

    # Collect inputs and demographics for all tkdnames
    all_X = []
    all_demographics = []
    all_tkdnames = []
    tkdname_indices = {}

    # Iterate through unique tkdnames
    index = 0
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]
        all_tkdnames.append(filtered_rows['tkdname'])

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            X = []
            demographics = []
            if modality == 'MULTIMODAL' or modality == 'FARZANA':
                language_flag = 1.0
                prosody_flag = 1.0
                lang_filenames = inputs['Language_Filename'].tolist()
                aud_filenames = inputs['Audio_Filename'].tolist()
                for lang_filename, aud_filename in zip(lang_filenames, aud_filenames):
                    lang_data = np.load(lang_filename)*language_flag
                    aud_data = np.load(aud_filename)*prosody_flag
                    data = np.concatenate((lang_data,aud_data), axis=-1)

                    # Extract demographics from the dataframe
                    demographics_row = inputs[inputs['Language_Filename'] == lang_filename][['age', 'sex', 'Language']].values[0]

                    # Append data and demographics to the lists
                    X.append(data)
                    demographics.append(demographics_row)
            else:
                npy_filenames = inputs['Filename'].tolist()
                for filename in npy_filenames:
                    data = np.load(filename)

                    # Extract demographics from the dataframe
                    demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                    # Append data and demographics to the lists
                    X.append(data)
                    demographics.append(demographics_row)

            all_X.extend(X)
            all_demographics.extend(demographics)
            tkdname_indices[tkdname] = (index, index + len(X))
            index += len(X)
    all_X = np.vstack(all_X)
    if data_mean is not None and data_std is not None:
        all_X = (all_X-data_mean) / (data_std + 1e-7)

    # Run all inputs through the neural network model
    outputs = model.predict_proba(all_X)
    outputs = outputs[:,1]

    eval_df = pd.DataFrame({'tkdname':pd.concat(all_tkdnames, ignore_index=True)})
    eval_df['output'] = outputs

    return eval_df


def legacy_average_model_output_per_tkdname(tkdnames, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}

    # Iterate through unique tkdnames
    for tkdname in tkdnames:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['tkdname'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            npy_filenames = inputs['Filename'].tolist()

            X = []
            demographics = []
            y = []
            for filename in npy_filenames:
                data = np.load(filename)

                # Extract demographics from the dataframe
                demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                # Extract label from the dataframe
                label = inputs[inputs['Filename'] == filename]['dx'].values[0]

                # Append data, demographics, and label to the lists
                X.append(data)
                demographics.append(demographics_row)
                y.append(label)

            X, demographics, y = np.vstack(X), np.vstack(demographics), np.vstack(y)

            # Run the inputs through the neural network model
            
            outputs = model.predict([X,demographics],verbose=0)

            # Calculate the average output for the current tkdname
            average_output = np.mean(outputs, axis=0)

            # Store the average output for the current tkdname
            output_per_tkdname[tkdname] = average_output

    return output_per_tkdname

def get_bert_ae_embeddings_per_tkdname(subjects, bert_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}
    all_outputs = []
    all_y = []
    all_mmse = []

    # Iterate through unique tkdnames
    for tkdname in subjects:
        # Filter bert_df for rows with the same tkdname
        filtered_rows = bert_df[bert_df['Subject'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            npy_filenames = inputs['Filename'].tolist()

            X = []
            demographics = []
            prosody = []
            y = [] 
            mmse = []
            for filename in npy_filenames:
                data = np.load(filename)

                # Extract demographics from the dataframe
                demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                # Extract label from the dataframe
                label = inputs[inputs['Filename'] == filename]['dx'].values[0]
                # mmse_val = inputs[inputs['Filename'] == filename]['mmse'].values[0]

                # Append data, demographics, and label to the lists
                X.append(data)
                demographics.append(demographics_row)
                y.append(label)
                # mmse.append(mmse_val)

            # X, demographics, prosody, y = np.stack(X,axis=1), np.stack(demographics,axis=1), np.stack(prosody,axis=1), np.stack(y)
            X = np.stack(X,axis=1)
            demographics = np.vstack(demographics)
            demographics = demographics[None,:,:]
            all_y.append(np.stack(y))
            # all_mmse.append(np.stack(mmse))

            # Run the inputs through the neural network model
            outputs = model.predict([X,demographics],verbose=0)
            all_outputs.append(outputs)

    return np.vstack(all_outputs), np.vstack(all_y) #, np.vstack(all_mmse)

def get_w2v2_ae_embeddings_per_tkdname(subjects, w2v2_df, model):
    # Initialize an empty dictionary to store aggregated outputs per tkdname
    output_per_tkdname = {}
    all_outputs = []
    all_y = []
    all_mmse = []

    # Iterate through unique tkdnames
    for tkdname in subjects:
        # Filter w2v2_df for rows with the same tkdname
        filtered_rows = w2v2_df[w2v2_df['Subject'] == tkdname]

        # Check if there are any rows for the current tkdname
        if not filtered_rows.empty:
            # Get the inputs for the neural network
            inputs = filtered_rows.drop(columns=['tkdname'])

            npy_filenames = inputs['Filename'].tolist()

            X = []
            demographics = []
            prosody = []
            y = [] 
            mmse = []
            for filename in npy_filenames:
                data = np.load(filename)
                data = np.tanh(data)
                data = np.mean(data, axis=1)

                # Extract demographics from the dataframe
                demographics_row = inputs[inputs['Filename'] == filename][['age', 'sex', 'Language']].values[0]

                # Extract label from the dataframe
                label = inputs[inputs['Filename'] == filename]['dx'].values[0]
                # mmse_val = inputs[inputs['Filename'] == filename]['mmse'].values[0]

                # Append data, demographics, and label to the lists
                X.append(data)
                demographics.append(demographics_row)
                y.append(label)
                # mmse.append(mmse_val)

            # X, demographics, prosody, y = np.stack(X,axis=1), np.stack(demographics,axis=1), np.stack(prosody,axis=1), np.stack(y)
            X = np.stack(X,axis=1)
            demographics = np.vstack(demographics)
            demographics = demographics[None,:,:]
            all_y.append(np.stack(y))
            # all_mmse.append(np.stack(mmse))

            # Run the inputs through the neural network model
            outputs = model.predict([X,demographics],verbose=0)
            all_outputs.append(outputs)

    return np.vstack(all_outputs), np.vstack(all_y) #, np.vstack(all_mmse)

def get_scores(tkdnames, data_df, model, EI=False, pytorch=False, modality=None, data_mean=None, data_std=None):

    if EI:
        eval, _ = average_ei_model_output_per_tkdname(tkdnames,data_df,model,modality,data_mean,data_std)
    elif pytorch:
        if modality == 'MISA':
            eval, _ = average_MISA_model_output_per_tkdname(tkdnames,data_df,model)
        elif modality == 'MFN':
            eval,_ = average_MFN_model_output_per_tkdname(tkdnames,data_df,model)
        elif modality == 'BC-LSTM':
            eval,_ = average_BCLSTM_model_output_per_tkdname(tkdnames,data_df,model)
        else:
            eval, _ = average_pytorch_model_output_per_tkdname(tkdnames,data_df,model)
    else:
        eval = average_model_output_per_tkdname(tkdnames,data_df,model)
    eval_df = pd.DataFrame(eval.items(), columns=['tkdname', 'output'])
    labels = data_df[['Subject','dx']].drop_duplicates()
    eval_df['Subject'] = eval_df.tkdname.str[:-4]
    eval_df.sort_values(by='tkdname', inplace=True)

    mean_ensemble = eval_df.groupby('Subject')['output'].mean().reset_index()
    y_pred = np.vstack(mean_ensemble['output'].values)[:,0]
    true_df = data_df.groupby('Subject')['dx'].mean().reset_index()
    y_train = true_df['dx'].values
    class_balance_thresh = 1-(true_df['dx'].values.sum()/true_df['dx'].values.shape[0])

    out_scores = scores(y_train.astype(int),y_pred)
    out_scores['Accuracy'] = accuracy_score(y_train.astype(int),(y_pred>class_balance_thresh).astype(int))

    return out_scores


def get_predictions(tkdnames, data_df, model, EI=False, pytorch=False, modality=None, data_mean=None, data_std=None):

    if EI:
        eval_df = ei_model_output_per_tkdname(tkdnames,data_df,model,modality,data_mean,data_std)
    elif pytorch:
        if modality == 'MISA':
            eval, _ = average_MISA_model_output_per_tkdname(tkdnames,data_df,model)
        elif modality == 'MFN':
            eval,_ = average_MFN_model_output_per_tkdname(tkdnames,data_df,model)
        elif modality == 'BC-LSTM':
            eval,_ = average_BCLSTM_model_output_per_tkdname(tkdnames,data_df,model)
        else:
            eval_df = pytorch_model_output_per_tkdname(tkdnames,data_df,model)
    else:
        eval_df = average_model_output_per_tkdname(tkdnames,data_df,model)
    # eval_df = pd.DataFrame(eval.items(), columns=['tkdname', 'output'])
    # labels = data_df[['Subject','dx']].drop_duplicates()
    # eval_df['Subject'] = eval_df.tkdname.str[:-4]
    # eval_df.sort_values(by='tkdname', inplace=True)

    return eval_df


def get_legacy_scores(tkdnames, data_df, model):
    eval = legacy_average_model_output_per_tkdname(tkdnames,data_df,model)
    eval_df = pd.DataFrame(eval.items(), columns=['tkdname', 'output'])
    labels = data_df[['Subject','dx']].drop_duplicates()
    eval_df['Subject'] = eval_df.tkdname.str[:-4]
    eval_df.sort_values(by='tkdname', inplace=True)

    mean_ensemble = eval_df.groupby('Subject')['output'].mean().reset_index()
    y_pred = np.vstack(mean_ensemble['output'].values)[:,0]
    true_df = data_df.groupby('Subject')['dx'].mean().reset_index()
    y_train = true_df['dx'].values

    out_scores = scores(y_train.astype(int),y_pred)
    out_scores['Accuracy'] = accuracy_score(y_train.astype(int),(y_pred>0.476).astype(int))

    return out_scores

class LinearWarmUpScheduler(keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_steps):
        super(LinearWarmUpScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = self.compute_lr()
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = self.compute_lr()
        print(f'Current learning rate: {lr}')

    def compute_lr(self):
        if self.global_step < self.warmup_steps:
            return self.initial_lr * (self.global_step / self.warmup_steps)
        else:
            return self.initial_lr

def random_oversample_dataframe(dataframe, random_state=0):
    ros = RandomOverSampler(random_state=random_state)
    x = dataframe.drop(columns='dx')
    y = dataframe['dx'].astype('Float32')

    resampled_x, resampled_y = ros.fit_resample(x,y)

    resampled_dataframe = resampled_x.copy()
    resampled_dataframe['dx'] = resampled_y

    return resampled_dataframe

def prepare_dataframes(dataset, training_config, cohort='AD'):
    column_rename_dict = {
        'CI status':'dx',
        # 'Severe CI status':'dx',
        'AGE':'age',
        'SEX':'sex'
        }

        
    if cohort == 'NU':
        groundtruth_df = pd.read_csv(dataset['groundtruth_csv'])
        groundtruth_df['tkdname'] = groundtruth_df.Subject.str[:] + '.wav'
        groundtruth_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Language'] = 'English'
        groundtruth_df = groundtruth_df[groundtruth_df['SEX'].isin(['Male','Female'])]
        groundtruth_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        groundtruth_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        drop_list = ['RACE_ETHNICITY_COMBINED','record_id','One Drive File Name:','Completed PCP Recording?','Cookie Thief','In Person Completed Date:','Issue']
        groundtruth_df.drop(columns=drop_list, inplace=True)
        groundtruth_df.dropna(axis=0, inplace=True)

        if training_config['modality']=='BERT':
            data_df = pd.read_csv(dataset['bert_df'])

        elif training_config['modality']=='DistilBERT':
            if training_config['architecture']=='EI':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='BP':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='SeqCls':
                data_df = pd.read_csv(dataset['distilbert_token_df'])

        elif training_config['modality']=='W2V2':
            if training_config['architecture'] in ['AE','EI']:
                data_df = pd.read_csv(dataset['w2v2_chunk_df']) ##Change back to single
            else:
                data_df = pd.read_csv(dataset['w2v2_chunk_df'])   

        elif training_config['modality']=='W2V2_SENT':
            data_df = pd.read_csv(dataset['w2v2_sentence_df'])

        elif training_config['modality']=='PROSODY':
            data_df = pd.read_csv(dataset['prosody_chunk_df'])

        elif training_config['modality']=='PROSODY_SENT':
            data_df = pd.read_csv(dataset['prosody_sentence_df'])
        
        elif training_config['modality']=='PROSODY_REC':
            data_df = pd.read_csv(dataset['prosody_rec_df'])

        elif training_config['modality']=='DNN_SENT':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='INTERP_REC':
            data_df_lang = pd.read_csv(dataset['lftk_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='EGEMAPS':
            data_df = pd.read_csv(dataset['egemaps_chunk_df'])

        elif training_config['modality']=='MPNET':
            data_df = pd.read_csv(dataset['sentence_df']) 

        elif training_config['modality']=='LFTK':
            data_df = pd.read_csv(dataset['lftk_df']) 

        elif training_config['modality']=='SIGNLTK':
            data_df = pd.read_csv(dataset['significant_nltk_df']) 

        elif training_config['modality']=='HUBERT':
            data_df = pd.read_csv(dataset['hubert_chunk_df']) 

        elif training_config['modality']=='SARAWGI':
            data_df = pd.read_csv(dataset['compare_df']) 

        elif training_config['modality']=='MISA':
            data_df_lang = pd.read_csv(dataset['misa_bert_sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN EASI-COG':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='BC-LSTM':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='FARZANA':
            data_df_lang = pd.read_csv(dataset['farzana_nlp_df'])  
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
        
        elif training_config['modality']=='Ying':
            data_df_lang = pd.read_csv(dataset['ying_bert_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)
            data_df_iso = pd.read_csv(dataset['ying_iso_df'])
            data_df_iso.rename(columns={"Filename":"Iso_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
            data_df['Iso_Filename'] = data_df_iso['Iso_Filename'].values

        elif training_config['modality']=='Wavbert':
            data_df_lang = pd.read_csv(dataset['wavbert_nlp_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df']) ###Leaving this in place bc I'm too lazy to reconfigure the collate_fn
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='WHISPER':
            data_df = pd.read_csv(dataset['whisper_embedding_df'])

        


        data_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df = data_df[data_df['SEX'].isin(['Male','Female'])]
        data_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        data_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        data_df['Subject'] = data_df.tkdname.str[:-4]
        data_df = data_df[data_df['RACE_ETHNICITY_COMBINED']=='BLACK OR AFRICAN-AMERICAN']
        data_df.drop(columns=drop_list,inplace=True)
        # data_testfile = data_df['Filename'][0]
        # data_input_size = np.load(data_testfile).shape

        groundtruth_df.rename(columns=column_rename_dict, inplace=True)
        data_df.rename(columns=column_rename_dict, inplace=True)

        unique_subjects = data_df['Subject'].unique()
        groundtruth_df = groundtruth_df[groundtruth_df['Subject'].isin(unique_subjects)]

    elif cohort == 'MSSM':
        groundtruth_df = pd.read_csv(dataset['groundtruth_csv'])
        groundtruth_df['tkdname'] = groundtruth_df.Subject.str[:] + '.wav'
        groundtruth_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Language'] = 'English'
        groundtruth_df = groundtruth_df[groundtruth_df['SEX'].isin(['Male','Female'])]
        groundtruth_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        groundtruth_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        drop_list = ['MRN','Notes ID','Survey ID','Audio Survey ID','In-person interview date','Status of Audio','T-MoCA','RACE_ETHNICITY_COMBINED']
        drop_list = ['MRN','Notes ID','Survey ID','Audio Survey ID','In-person interview date','Status of Audio','T-MoCA']
        groundtruth_df.drop(columns=drop_list, inplace=True)
        groundtruth_df.dropna(axis=0, inplace=True)

        if training_config['modality']=='BERT':
            data_df = pd.read_csv(dataset['bert_df'])

        elif training_config['modality']=='DistilBERT':
            if training_config['architecture']=='EI':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='BP':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='SeqCls':
                data_df = pd.read_csv(dataset['distilbert_token_df'])

        elif training_config['modality']=='W2V2':
            if training_config['architecture'] in ['AE','EI']:
                data_df = pd.read_csv(dataset['w2v2_chunk_df']) ##Change back to single
            else:
                data_df = pd.read_csv(dataset['w2v2_chunk_df'])   

        elif training_config['modality']=='W2V2_SENT':
            data_df = pd.read_csv(dataset['w2v2_sentence_df'])

        elif training_config['modality']=='PROSODY':
            data_df = pd.read_csv(dataset['prosody_chunk_df'])

        elif training_config['modality']=='PROSODY_SENT':
            data_df = pd.read_csv(dataset['prosody_sentence_df'])
        
        elif training_config['modality']=='PROSODY_REC':
            data_df = pd.read_csv(dataset['prosody_rec_df'])

        elif training_config['modality']=='DNN_SENT':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='INTERP_REC':
            data_df_lang = pd.read_csv(dataset['lftk_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='EGEMAPS':
            data_df = pd.read_csv(dataset['egemaps_chunk_df'])

        elif training_config['modality']=='MPNET':
            data_df = pd.read_csv(dataset['sentence_df']) 
        
        elif training_config['modality']=='JINA':
            data_df = pd.read_csv(dataset['jina_df']) 

        elif training_config['modality']=='LFTK':
            data_df = pd.read_csv(dataset['lftk_df']) 

        elif training_config['modality']=='SIGNLTK':
            data_df = pd.read_csv(dataset['significant_nltk_df']) 

        elif training_config['modality']=='HUBERT':
            data_df = pd.read_csv(dataset['hubert_chunk_df']) 

        elif training_config['modality']=='SARAWGI':
            data_df = pd.read_csv(dataset['compare_df']) 

        elif training_config['modality']=='MISA':
            data_df_lang = pd.read_csv(dataset['misa_bert_sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN EASI-COG':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='BC-LSTM':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='FARZANA':
            data_df_lang = pd.read_csv(dataset['farzana_nlp_df'])  
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='EASI-COG_MULTIMODAL':
            data_df_lang = pd.read_csv(dataset['lftk_df'])  
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
        
        elif training_config['modality']=='Ying':
            data_df_lang = pd.read_csv(dataset['ying_bert_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)
            data_df_iso = pd.read_csv(dataset['ying_iso_df'])
            data_df_iso.rename(columns={"Filename":"Iso_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
            data_df['Iso_Filename'] = data_df_iso['Iso_Filename'].values

        elif training_config['modality']=='Ying_IS10':
            data_df = pd.read_csv(dataset['ying_iso_df'])

        elif training_config['modality']=='WHISPER':
            data_df = pd.read_csv(dataset['whisper_embedding_df'])

        elif training_config['modality']=='HUBERT':
            data_df = pd.read_csv(dataset['hubert_embedding_df'])

        elif training_config['modality']=='WHISPER_FINETUNE':
            data_df = pd.read_csv(dataset['whisper_chunk_df'])

        elif training_config['modality']=='Wavbert':
            data_df_lang = pd.read_csv(dataset['wavbert_nlp_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df']) ###Leaving this in place bc I'm too lazy to reconfigure the collate_fn
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        data_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df = data_df[data_df['SEX'].isin(['Male','Female'])]
        data_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        data_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        data_df['Subject'] = data_df.tkdname.str[:-4]
        data_df.drop(columns=drop_list,inplace=True)
        # data_testfile = data_df['Filename'][0]
        # data_input_size = np.load(data_testfile).shape

        groundtruth_df.rename(columns=column_rename_dict, inplace=True)
        data_df.rename(columns=column_rename_dict, inplace=True)

        unique_subjects = data_df['Subject'].unique()
        groundtruth_df = groundtruth_df[groundtruth_df['Subject'].isin(unique_subjects)]

    elif cohort == 'NU_PCP':
        groundtruth_df = pd.read_csv(dataset['groundtruth_csv'])
        groundtruth_df['tkdname'] = groundtruth_df.Subject.str[:] + '.wav'
        groundtruth_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        groundtruth_df['Language'] = 'English'
        groundtruth_df = groundtruth_df[groundtruth_df['SEX'].isin(['Male','Female'])]
        groundtruth_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        groundtruth_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        drop_list = ['RACE_ETHNICITY_COMBINED','record_id','One Drive File Name:','Completed PCP Recording?','Cookie Thief','In Person Completed Date:','Issue']
        drop_list = ['record_id','One Drive File Name:','Completed PCP Recording?','Cookie Thief','In Person Completed Date:','Issue']
        groundtruth_df.drop(columns=drop_list, inplace=True)
        groundtruth_df.dropna(axis=0, inplace=True)

        if training_config['modality']=='BERT':
            data_df = pd.read_csv(dataset['bert_df'])

        elif training_config['modality']=='DistilBERT':
            if training_config['architecture']=='EI':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='BP':
                data_df = pd.read_csv(dataset['distilbert_df'])
            elif training_config['architecture']=='SeqCls':
                data_df = pd.read_csv(dataset['distilbert_token_df'])

        elif training_config['modality']=='W2V2':
            if training_config['architecture'] in ['AE','EI']:
                data_df = pd.read_csv(dataset['w2v2_chunk_df']) ##Change back to single
            else:
                data_df = pd.read_csv(dataset['w2v2_chunk_df'])   

        elif training_config['modality']=='W2V2_SENT':
            data_df = pd.read_csv(dataset['w2v2_sentence_df'])

        elif training_config['modality']=='PROSODY':
            data_df = pd.read_csv(dataset['prosody_chunk_df'])

        elif training_config['modality']=='PROSODY_SENT':
            data_df = pd.read_csv(dataset['prosody_sentence_df'])
        
        elif training_config['modality']=='PROSODY_REC':
            data_df = pd.read_csv(dataset['prosody_rec_df'])

        elif training_config['modality']=='DNN_SENT':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='INTERP_REC':
            data_df_lang = pd.read_csv(dataset['lftk_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='EGEMAPS':
            data_df = pd.read_csv(dataset['egemaps_chunk_df'])

        elif training_config['modality']=='MPNET':
            data_df = pd.read_csv(dataset['sentence_df']) 
        
        elif training_config['modality']=='JINA':
            data_df = pd.read_csv(dataset['jina_df']) 

        elif training_config['modality']=='LFTK':
            data_df = pd.read_csv(dataset['lftk_df']) 

        elif training_config['modality']=='SIGNLTK':
            data_df = pd.read_csv(dataset['significant_nltk_df']) 

        elif training_config['modality']=='HUBERT':
            data_df = pd.read_csv(dataset['hubert_chunk_df']) 

        elif training_config['modality']=='SARAWGI':
            data_df = pd.read_csv(dataset['compare_df']) 

        elif training_config['modality']=='MISA':
            data_df_lang = pd.read_csv(dataset['misa_bert_sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='MFN EASI-COG':
            data_df_lang = pd.read_csv(dataset['sentence_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='BC-LSTM':
            data_df_lang = pd.read_csv(dataset['mfn_glove_word_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['misa_acoustic_sentence_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='FARZANA':
            data_df_lang = pd.read_csv(dataset['farzana_nlp_df'])  
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        elif training_config['modality']=='EASI-COG_MULTIMODAL':
            data_df_lang = pd.read_csv(dataset['lftk_df'])  
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['prosody_rec_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
        
        elif training_config['modality']=='Ying':
            data_df_lang = pd.read_csv(dataset['ying_bert_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df'])
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)
            data_df_iso = pd.read_csv(dataset['ying_iso_df'])
            data_df_iso.rename(columns={"Filename":"Iso_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values
            data_df['Iso_Filename'] = data_df_iso['Iso_Filename'].values

        elif training_config['modality']=='Ying_IS10':
            data_df = pd.read_csv(dataset['ying_iso_df'])

        elif training_config['modality']=='WHISPER':
            data_df = pd.read_csv(dataset['whisper_embedding_df'])

        elif training_config['modality']=='HUBERT':
            data_df = pd.read_csv(dataset['hubert_embedding_df'])

        elif training_config['modality']=='WHISPER_FINETUNE':
            data_df = pd.read_csv(dataset['whisper_chunk_df'])

        elif training_config['modality']=='Wavbert':
            data_df_lang = pd.read_csv(dataset['wavbert_nlp_df'])
            data_df_lang.rename(columns={"Filename":"Language_Filename"}, inplace=True)
            data_df_aud = pd.read_csv(dataset['ying_audio_df']) ###Leaving this in place bc I'm too lazy to reconfigure the collate_fn
            data_df_aud.rename(columns={"Filename":"Audio_Filename"}, inplace=True)

            data_df = data_df_lang.copy()
            data_df['Audio_Filename'] = data_df_aud['Audio_Filename'].values

        data_df['CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df['Severe CI status'].replace({'No Impairment': 0, 'Cognitively Impaired': 1}, inplace=True)
        data_df = data_df[data_df['SEX'].isin(['Male','Female'])]
        data_df['SEX'].replace({'Female': 0, 'Male': 1}, inplace=True)
        data_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        data_df['Subject'] = data_df.tkdname.str[:-4]
        data_df.drop(columns=drop_list,inplace=True)
        # data_testfile = data_df['Filename'][0]
        # data_input_size = np.load(data_testfile).shape

        groundtruth_df.rename(columns=column_rename_dict, inplace=True)
        data_df.rename(columns=column_rename_dict, inplace=True)

        unique_subjects = data_df['Subject'].unique()
        groundtruth_df = groundtruth_df[groundtruth_df['Subject'].isin(unique_subjects)]

    elif cohort == 'AD':
        groundtruth_df = pd.read_csv(dataset['groundtruth_csv'])
        groundtruth_df['dx'].replace({'Control': 0, 'ProbableAD': 1}, inplace=True)
        groundtruth_df['Subject'] = groundtruth_df.tkdname
        groundtruth_df['tkdname'] = groundtruth_df.tkdname + '.wav'
        groundtruth_df['age'] = 1.0
        groundtruth_df['sex'] = 1
        # groundtruth_df.dropna(axis=0, inplace=True)

        if training_config['modality']=='BERT':
            data_df = pd.read_csv(dataset['bert_df'])
        elif training_config['modality']=='W2V2':
            if training_config['architecture'] in ['AE','EI']:
                data_df = pd.read_csv(dataset['w2v2_chunk_df']) ##Change back to single
            else:
                data_df = pd.read_csv(dataset['w2v2_chunk_df'])  

        elif training_config['modality']=='EGEMAPS':
            data_df = pd.read_csv(dataset['egemaps_chunk_df'])

        elif training_config['modality']=='PROSODY_REC':
            data_df = pd.read_csv(dataset['prosody_rec_df'])
        
        elif training_config['modality']=='WHISPER':
            data_df = pd.read_csv(dataset['whisper_embedding_df'])

        elif training_config['modality']=='WHISPER_FINETUNE':
            data_df = pd.read_csv(dataset['whisper_chunk_df'])
            
        elif training_config['modality']=='LFTK':
            data_df = pd.read_csv(dataset['lftk_df']) 

        data_df['dx'].replace({'Control': 0, 'ProbableAD': 1}, inplace=True)
        data_df['Language'].replace({'English': 0, 'Chinese': 1}, inplace=True)
        data_df['Subject'] = data_df.tkdname.str[:-4]
        data_df['age'] = 1.0
        data_df['sex'] = 1
        data_testfile = data_df['Filename'][0]
        data_input_size = np.load(data_testfile).shape

        # groundtruth_df.rename(columns=column_rename_dict, inplace=True)
        # data_df.rename(columns=column_rename_dict, inplace=True)

        unique_subjects = data_df['Subject'].unique()
        groundtruth_df = groundtruth_df[groundtruth_df['Subject'].isin(unique_subjects)]

    return groundtruth_df, data_df

def model_probability(ad_preds):
  model_probabilities = []

  neg_preds = ad_preds[ad_preds<=0.5]
  pos_preds = ad_preds[ad_preds>=0.5]

  model_probabilities.append(np.mean(np.abs(neg_preds-0.5)))
  model_probabilities.append(np.mean(np.abs(pos_preds-0.5)))

  return model_probabilities

def majority_vote(ad_preds):
  ad_decisions = np.round(ad_preds).astype(int)
  if len(ad_decisions[ad_decisions==0]) == len(ad_decisions[ad_decisions==1]): #tie
    return np.argmax(model_probability(ad_preds))
  else:
    return np.argmax(np.bincount(ad_decisions))

def SA(mmse_preds, ad_decision, threshold):
  if ad_decision==1:
    if len(mmse_preds[mmse_preds<=threshold]) == 0:
      return np.min(mmse_preds)
    else:
      return np.mean(mmse_preds[mmse_preds<=threshold])
  else:
    if len(mmse_preds[mmse_preds>=threshold])==0:
      return np.max(mmse_preds)
    else:
      return np.mean(mmse_preds[mmse_preds>=threshold])

def margin_of_victory(ad_preds):
  ad_decisions = np.round(ad_preds).astype(int)
  ad_decisions[ad_decisions==0] = -1
  return np.abs(np.sum(ad_decisions))

def all_mmses_predict_vote_loser(mmse_preds, ad_decision, threshold):
  mmse_decisions = np.where(mmse_preds <= threshold, 1, 0)
  return np.all(mmse_decisions==1-ad_decision)

def simple_CONSEN(ad_preds, mmse_preds, threshold):
  vote_winner = majority_vote(ad_preds)
  margin = margin_of_victory(ad_preds)
  vote_loser = 1-vote_winner

  if margin>=2:
      return vote_winner, SA(mmse_preds, vote_winner, threshold)

  elif margin==1:
    model_probabilities = model_probability(ad_preds)
    if model_probabilities[vote_winner] < model_probabilities[vote_loser] and all_mmses_predict_vote_loser(mmse_preds, vote_winner, threshold):
      return vote_loser, SA(mmse_preds, vote_loser, threshold)
    else:
      return vote_winner, SA(mmse_preds, vote_winner, threshold)

  else: #tie goes to class with higher model probability
    if all_mmses_predict_vote_loser(mmse_preds, vote_winner, threshold):
      return vote_loser, SA(mmse_preds, vote_loser, threshold)
    else:
      return vote_winner, SA(mmse_preds, vote_winner, threshold)


def group_encode(mmse_array, num_groups=15): #Function for one-hot encoding the MMSE values
    encoded_array = np.zeros((len(mmse_array), num_groups))
    for ii, num in enumerate(mmse_array):
      if num == 0: #avoiding negative indexing bug if MMSE=0
        num+=1
      encoded_array[ii,int(np.ceil((num)/2)-1)] = 1

    return encoded_array

def tk_scores(y_true, y_pred, beta=1, metric_to_maximise="fscore", verbose=0):

    # Get Fmax
    f_measure_hc = fmeasure_score(y_true, y_pred)

    # Get F on majority class using the threshold of Fmax on the minority class
    f_measure_ci = fmeasure_score(y_true, y_pred, thres=1-f_measure_hc['thres'])

    # AUC
    auc = roc_auc_score(y_true, y_pred)

    # Accuracy
    accuracy = accuracy_score(y_true, [y>0.5 for y in y_pred])
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred > 0.5)

    # UAR
    cm = confusion_matrix(y_true, [y>0.5 for y in y_pred])
    tp, tn = cm[0, 0], cm[1, 1]
    fn, fp = cm[1, 0], cm[0, 1]

    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (fp + tn + 1e-7)

    uar = (specificity + sensitivity)/2.0

    # Store all scores
    scores_dict = {"Accuracy" : accuracy,
                   "Sensitivity": sensitivity,
                   "Specificity": specificity,
                   "Precision (minority)": precision[1],
                   "Recall (minority)": recall[1],
                   "UAR" : uar,
                   "F HC": f_measure_hc['F'],
                   "F CI": f_measure_ci['F'],
                   "AUC": auc
                             }  # dictionary of (score, threshold)

    # Print all scores (if you want)
    if verbose > 0:
        for metric_name, score in scores_dict.items():
            print(metric_name + ": ", score[0])

    return scores_dict

class MISAMSE(nn.Module):
    def __init__(self):
        super(MISAMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class MISASIMSE(nn.Module):

    def __init__(self):
        super(MISASIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class MISADiffLoss(nn.Module):

    def __init__(self):
        super(MISADiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class MISACMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(MISACMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)