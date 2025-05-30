import numpy as np
import tensorflow as tf
import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import TFDistilBertForSequenceClassification, HubertModel, AutoProcessor, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model, BertModel
import whisper
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Concatenate, Conv1D, Attention
from keras.layers import MaxPooling1D, Reshape, BatchNormalization, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten, LayerNormalization
from keras.optimizers import Adam, AdamW
from keras.models import load_model
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from eipy.additional_ensembles import MeanAggregation, CES
from eipy.metrics import fmax_score
from pynvml import *
from itertools import product


class DistilBERTSeqClassifier(Model):
    def __init__(
        self,
        distilbert_model
    ):
        super(DistilBERTSeqClassifier, self).__init__()
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            distilbert_model
            )
        classifier = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="pre_classifier",
        )
        self.model.classifier = classifier

    def call(self, inputs):
        bert_tokens, demographics_input = inputs

        outputs = self.model(bert_tokens)
        outputs = outputs['logits']

        return outputs

class BERTClassifier(Model):
    def __init__(
            self, 
            input_size, 
            dense_units, 
            output_head, 
            dropout_rate=0.3, 
            activation='tanh', 
            use_demographics=False,
            num_mmse_groups=15):

        super(BERTClassifier, self).__init__()
        self.input_size = input_size
        self.dense_units = dense_units
        self.output_head = output_head
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_demographics = use_demographics
        self.num_mmse_groups = num_mmse_groups

        self.dense_layers = [Dense(dense_units[ii], activation=activation) for ii in range(len(dense_units))]
        self.dropout_layers = [Dropout(rate=dropout_rate) for ii in range(len(dense_units))]

        if self.output_head == "MCI":
            self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')
        elif self.output_head == "MMSE":
            self.output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        elif self.output_head == "joint":
            self.mci_output_layer = Dense(1, activation='sigmoid', name='MCI_output')
            self.mmse_output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        else:
            print("Improper output head specified, now exiting")
            exit()

    def call(self, inputs):
        bert_input, demographics_input = inputs

        if self.use_demographics:
            x = Concatenate()([bert_input, demographics_input])
        else:
            x = bert_input

        for ii in range(len(self.dense_units)):
            x = self.dense_layers[ii](x)
            x = self.dropout_layers[ii](x)

        if self.output_head == "MCI":
            output = self.output_layer(x)
        elif self.output_head == "MMSE":
            output = self.output_layer(x)
        elif self.output_head == "joint":
            mci_output = self.mci_output_layer(x)
            mmse_output = self.mmse_output_layer(x)
            output = [mci_output, mmse_output]
        else:
            print("Improper output head specified, now exiting")
            exit()

        return output

class W2V2Classifier(Model):
    def __init__(
            self, 
            input_size, 
            dense_units, 
            output_head, 
            dropout_rate=0.4, 
            activation='tanh', 
            use_demographics=False,
            num_mmse_groups=15):

        super(W2V2Classifier, self).__init__()
        self.input_size = input_size
        self.dense_units = dense_units
        self.output_head = output_head
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_demographics = use_demographics
        self.num_mmse_groups = num_mmse_groups

        self.pooling = GlobalAveragePooling1D()
        self.layernorm = LayerNormalization()
        self.concat = Concatenate()
        self.dense_layers = [Dense(dense_units[ii], activation=activation) for ii in range(len(dense_units))]
        self.dropout_layers = [Dropout(rate=dropout_rate) for ii in range(len(dense_units))]

        if self.output_head == "MCI":
            self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')
        elif self.output_head == "MMSE":
            self.output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        elif self.output_head == "joint":
            self.mci_output_layer = Dense(1, activation='sigmoid', name='MCI_output')
            self.mmse_output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        else:
            print("Improper output head specified, now exiting")
            exit()

    def call(self, inputs):
        w2v2_input, demographics_input = inputs

        pooled_w2v2 = self.pooling(w2v2_input)
        normed_w2v2 = self.layernorm(pooled_w2v2)

        if self.use_demographics:
            x = self.concat([normed_w2v2,demographics_input])
        else:
            x = normed_w2v2

        for ii in range(len(self.dense_units)):
            x = self.dense_layers[ii](x)
            x = self.dropout_layers[ii](x)

        if self.output_head == "MCI":
            output = self.output_layer(x)
        elif self.output_head == "MMSE":
            output = self.output_layer(x)
        elif self.output_head == "joint":
            mci_output = self.mci_output_layer(x)
            mmse_output = self.mmse_output_layer(x)
            output = [mci_output, mmse_output]
        else:
            print("Improper output head specified, now exiting")
            exit()

        return output

class oldB2AIW2V2Classifier(Model):
    def __init__(
            self, 
            input_size, 
            dense_units, 
            output_head, 
            dropout_rate=0.4, 
            activation='tanh', 
            use_demographics=False,
            num_mmse_groups=15):

        super(B2AIW2V2Classifier, self).__init__()
        self.input_size = input_size
        self.dense_units = dense_units
        self.output_head = output_head
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_demographics = use_demographics
        self.num_mmse_groups = num_mmse_groups

        self.pooling = GlobalAveragePooling1D(data_format='channels_first')
        self.layernorm = LayerNormalization()
        self.concat = Concatenate()
        self.dense_layers = [Dense(dense_units[ii], activation=activation) for ii in range(len(dense_units))]
        self.dropout_layers = [Dropout(rate=dropout_rate) for ii in range(len(dense_units))]

        if self.output_head == "MCI":
            self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')
        elif self.output_head == "MMSE":
            self.output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        elif self.output_head == "joint":
            self.mci_output_layer = Dense(1, activation='sigmoid', name='MCI_output')
            self.mmse_output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        else:
            print("Improper output head specified, now exiting")
            exit()

    def call(self, inputs):
        w2v2_input, demographics_input = inputs

        pooled_w2v2 = self.pooling(w2v2_input)
        # normed_w2v2 = self.layernorm(pooled_w2v2)

        if self.use_demographics:
            x = self.concat([normed_w2v2,demographics_input])
        else:
            x = pooled_w2v2

        for ii in range(len(self.dense_units)):
            x = self.dense_layers[ii](x)
            x = self.dropout_layers[ii](x)

        if self.output_head == "MCI":
            output = self.output_layer(x)
        elif self.output_head == "MMSE":
            output = self.output_layer(x)
        elif self.output_head == "joint":
            mci_output = self.mci_output_layer(x)
            mmse_output = self.mmse_output_layer(x)
            output = [mci_output, mmse_output]
        else:
            print("Improper output head specified, now exiting")
            exit()

        return output

class B2AIW2V2Classifier(Model):
    def __init__(
            self, 
            input_size, 
            dense_units, 
            output_head, 
            dropout_rate=0.4, 
            activation='tanh', 
            use_demographics=False,
            num_mmse_groups=15):

        super(B2AIW2V2Classifier, self).__init__()
        self.input_size = input_size
        self.dense_units = dense_units
        self.output_head = output_head
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.my_tanh = tf.math.tanh
        self.use_demographics = use_demographics
        self.num_mmse_groups = num_mmse_groups

        # self.pooling = GlobalAveragePooling1D(data_format='channels_first')
        self.layernorms = [LayerNormalization() for ii in range(len(dense_units))]
        self.conv_layer = Conv1D(
                    filters=1, 
                    kernel_size=1, 
                    strides=1, 
                    activation='tanh') 
        self.concat = Concatenate()
        self.dense_layers = [Dense(dense_units[ii], activation=activation) for ii in range(len(dense_units))]
        self.dropout_layers = [Dropout(rate=dropout_rate) for ii in range(len(dense_units))]

        if self.output_head == "MCI":
            self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')
        elif self.output_head == "MMSE":
            self.output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        elif self.output_head == "joint":
            self.mci_output_layer = Dense(1, activation='sigmoid', name='MCI_output')
            self.mmse_output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        else:
            print("Improper output head specified, now exiting")
            exit()

    def call(self, inputs):
        w2v2_input, demographics_input = inputs

        # pooled_w2v2 = self.pooling(w2v2_input)
        # normed_w2v2 = self.layernorm(pooled_w2v2)
        tanh_w2v2 = tf.math.tanh(w2v2_input)
        proj_w2v2 = self.conv_layer(tanh_w2v2)
        proj_w2v2 = tf.keras.backend.squeeze(proj_w2v2,axis=-1)

        if self.use_demographics:
            x = self.concat([proj_w2v2,demographics_input])
        else:
            x = proj_w2v2

        for ii in range(len(self.dense_units)):
            x = self.dense_layers[ii](x)
            x = self.layernorms[ii](x)
            x = self.dropout_layers[ii](x)

        if self.output_head == "MCI":
            output = self.output_layer(x)
        elif self.output_head == "MMSE":
            output = self.output_layer(x)
        elif self.output_head == "joint":
            mci_output = self.mci_output_layer(x)
            mmse_output = self.mmse_output_layer(x)
            output = [mci_output, mmse_output]
        else:
            print("Improper output head specified, now exiting")
            exit()

        return output

class GauderClassifier(Model):
    def __init__(
            self, 
            input_size, 
            filter_sizes,
            kernel_sizes,
            strides,
            output_head, 
            activation='relu', 
            use_demographics=False,
            num_mmse_groups=15):

        super(GauderClassifier, self).__init__()
        self.input_size = input_size
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.output_head = output_head
        self.activation = activation
        self.use_demographics = use_demographics
        self.num_mmse_groups = num_mmse_groups

        self.pooling = GlobalAveragePooling1D()
        self.layernorm = LayerNormalization()
        self.batchnorms = [BatchNormalization() for ii in range(len(filter_sizes))]
        self.concat = Concatenate()
        self.conv_layers = [
            Conv1D(
                filters=filter_sizes[ii], 
                kernel_size=kernel_sizes[ii], 
                strides=strides[ii], 
                activation='relu') for ii in range(len(kernel_sizes))]

        if self.output_head == "MCI":
            self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')
        elif self.output_head == "MMSE":
            self.output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        elif self.output_head == "joint":
            self.mci_output_layer = Dense(1, activation='sigmoid', name='MCI_output')
            self.mmse_output_layer = Dense(self.num_mmse_groups, activation='softmax', name='MMSE_output')
        else:
            print("Improper output head specified, now exiting")
            exit()

    def call(self, inputs):
        w2v2_input, demographics_input = inputs

        x = w2v2_input

        for ii in range(len(self.kernel_sizes)):
            x = self.conv_layers[ii](x)
            x = self.batchnorms[ii](x)

        x = self.pooling(x)

        if self.output_head == "MCI":
            output = self.output_layer(x)
        elif self.output_head == "MMSE":
            output = self.output_layer(x)
        elif self.output_head == "joint":
            mci_output = self.mci_output_layer(x)
            mmse_output = self.mmse_output_layer(x)
            output = [mci_output, mmse_output]
        else:
            print("Improper output head specified, now exiting")
            exit()

        return output

class Encoder(Model):
    def __init__(
            self, 
            encoder_hidden_units, 
            dropout_rate=0.0, 
            activation='tanh'):

        super(Encoder, self).__init__()
        self.encoder_hidden_units = encoder_hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.encoder_layers = [Dense(units, activation=activation) for units in encoder_hidden_units]

    def call(self, inputs):
        x, demographics_input = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class Decoder(Model):
    def __init__(
            self, 
            decoder_hidden_units, 
            dropout_rate=0.0, 
            activation='tanh'):

        super(Decoder, self).__init__()
        self.decoder_hidden_units = decoder_hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.decoder_layers = [Dense(units, activation=activation) for units in decoder_hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x

class Autoencoder(Model):
    def __init__(
            self, 
            input_size, 
            encoder_hidden_units, 
            decoder_hidden_units=[], 
            dropout_rate=0.0, 
            activation='tanh', 
            use_demographics=False,
            modality='BERT'):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_demographics = use_demographics
        self.modality = modality
        self.pooling = GlobalAveragePooling1D()

        self.encoder = Encoder(encoder_hidden_units, dropout_rate, activation)
        self.decoder = Decoder(decoder_hidden_units, dropout_rate, activation)

        self.output_layer = Dense(input_size[-1], activation='sigmoid')

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        output = self.output_layer(decoded)

        return output

class EIPredictors():
    def __init__(self):
        self.base_predictors = {
                    'ADAB': AdaBoostClassifier(),
                    'XGB': XGBClassifier(),
                    'DT': DecisionTreeClassifier(),
                    'RF': RandomForestClassifier(),
                    'GB': GradientBoostingClassifier(),
                    'KNN': KNeighborsClassifier(),
                    'LR': LogisticRegression(),
                    'NB': GaussianNB(),
                    'MLP': MLPClassifier(),
                    'SVM': SVC(probability=True,max_iter=int(1e5)),
                    }

        self.ensemble_predictors = {
                        'S.Mean' : MeanAggregation(),
                        'S.CES' : CES(scoring=lambda y_test, y_pred: fmax_score(y_test, y_pred)[0]),
                        # 'S.ADAB': AdaBoostClassifier(),
                        'S.XGB': XGBClassifier(),
                        # 'S.DT': DecisionTreeClassifier(),
                        "S.RF": RandomForestClassifier(),
                        # 'S.GB': GradientBoostingClassifier(),
                        # 'S.KNN': KNeighborsClassifier(),
                        'S.LR': LogisticRegression(),
                        # 'S.NB': GaussianNB(),
                        'S.MLP': MLPClassifier(),
                        # 'S.SVM': SVC(probability=True,max_iter=int(1e5)),
                    }

class FastEIPredictors():
    def __init__(self):
        self.base_predictors = {
                    'XGB': XGBClassifier(),
                    'DT': DecisionTreeClassifier(),
                    'RF': RandomForestClassifier(),
                    'GB': GradientBoostingClassifier(),
                    'LR': LogisticRegression(),
                    'NB': GaussianNB(),
                    'MLP': MLPClassifier(),
                    }

        self.ensemble_predictors = {
                        'S.Mean' : MeanAggregation(),
                        'S.XGB': XGBClassifier(),
                        'S.DT': DecisionTreeClassifier(),
                        "S.RF": RandomForestClassifier(),
                        'S.GB': GradientBoostingClassifier(),
                        'S.LR': LogisticRegression(),
                        'S.NB': GaussianNB(),
                        'S.MLP': MLPClassifier(),
                    }

class BasePredictors():
    def __init__(self):
        self.base_predictors = {
                    'ADAB': AdaBoostClassifier(),
                    'XGB': XGBClassifier(),
                    'DT': DecisionTreeClassifier(),
                    'RF': RandomForestClassifier(),
                    'GB': GradientBoostingClassifier(),
                    'KNN': KNeighborsClassifier(),
                    'LR': LogisticRegression(),
                    'NB': GaussianNB(),
                    'MLP': MLPClassifier(),
                    'SVM': SVC(probability=True,max_iter=int(1e5)),
                    }

class FastBasePredictors():
    def __init__(self):
        self.base_predictors = {
                    'ADAB': AdaBoostClassifier(),
                    'XGB': XGBClassifier(),
                    'DT': DecisionTreeClassifier(),
                    'RF': RandomForestClassifier(),
                    'GB': GradientBoostingClassifier(),
                    'LR': LogisticRegression(),
                    'NB': GaussianNB(),
                    'MLP': MLPClassifier(),
                    }

class XGBoostGridSearch():
    def __init__(self):
        # Define smart values for each parameter
        param_grid = {
            'n_estimators': [100, 300, 500],  # Controls the number of boosting rounds
            'max_depth': [3, 6, 9],  # Tree depth controls model complexity
            'learning_rate': [0.01, 0.1, 0.2],  # Step size in boosting
            'colsample_bytree': [0.6, 0.8, 1.0],  # Feature subsampling per tree
            'subsample': [0.7, 0.85, 1.0],  # Row sampling for boosting
        }
        
        # param_grid = {
        #     'n_estimators': [100, 300],  # Controls the number of boosting rounds
        #     'max_depth': [3, 6],  # Tree depth controls model complexity
        #     'learning_rate': [0.01, 0.1],  # Step size in boosting
        #     'colsample_bytree': [0.6, 0.8],  # Feature subsampling per tree
        #     'subsample': [0.7, 0.85],  # Row sampling for boosting
        # }

        # Generate all combinations of the chosen hyperparameter values (brute force approach)
        param_combinations = list(product(*param_grid.values()))
        
        # Store XGBoost classifiers with explicit parameter names
        self.base_predictors = {}
        
        # Add multiple XGBoost models with distinct parameter settings
        for i, params in enumerate(param_combinations):
            key = (f"XGB_n{params[0]}_d{params[1]}_lr{params[2]}_csbt{params[3]}_sub{params[4]}")
            self.base_predictors[key] = XGBClassifier(
                n_estimators=params[0],
                max_depth=params[1],
                learning_rate=params[2],
                colsample_bytree=params[3],
                subsample=params[4],
            )


class SigNLTKPredictors():
    def __init__(self):
        self.base_predictors = {
                    'RF': RandomForestClassifier(n_estimators=500),
                    }

class KUModel(tf.keras.Model):

    def __init__(
            self, 
            hidden_dim,  
            dropout_rate=0.2, 
            activation='relu', 
            use_demographics=False,
            num_mmse_groups=15):
        super(KUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Normalization.
        self.norm = tf.keras.layers.BatchNormalization(momentum=0.6)

        # Down-projection.
        self.feat_proj_1 = Conv1D(
                    filters=self.hidden_dim, 
                    kernel_size=1, 
                    strides=1, 
                    activation=self.activation)
        self.feat_proj_drop_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.feat_proj_2 = Conv1D(
                    filters=2*self.hidden_dim, 
                    kernel_size=1, 
                    strides=1, 
                    activation=self.activation)
        self.feat_proj_drop_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.feat_proj_3 = Conv1D(
                    filters=1, 
                    kernel_size=1, 
                    strides=1, 
                    activation=self.activation)
        self.feat_proj_drop_3 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.output_layer = Dense(1, activation='sigmoid', name='MCI_output')

        # Attention
        # self.attention = Attention()

    def call(self, x):

        egemaps_feats, demographics_feats = x
        x = self.norm(egemaps_feats)

        # Down-projection to hidden dim.
        x = self.feat_proj_1(x)
        x = self.feat_proj_drop_1(x)

        #Calculate Attention
        atten_x = self.feat_proj_2(x)
        atten_x = self.feat_proj_drop_2(atten_x)
        atten_x = self.feat_proj_3(atten_x)
        atten_x = self.feat_proj_drop_3(atten_x)

        att = tf.transpose(atten_x, perm=[0, 2, 1])
        att = tf.nn.softmax(att, axis=2)

        x_pooled = tf.matmul(att, x)
        x_pooled = tf.squeeze(x_pooled, axis=1)

        output = self.output_layer(x_pooled)

        return output

class PoolAttFF(tf.keras.layers.Layer):
    '''
    PoolAttFF: Attention-Pooling module with additional feed-forward network.
    '''         
    def __init__(
            self,
            hidden_dim,
            activation='relu',
            dropout_rate=0.2, 
            out_dim=1):
        super(PoolAttFF, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.out_dim = out_dim

        # Attention + mapping to output.
        self.linear1 = tf.keras.layers.Dense(2*self.hidden_dim, activation=self.activation)
        self.linear2 = tf.keras.layers.Dense(1, activation=None)
        self.linear3 = tf.keras.layers.Dense(self.out_dim, activation='sigmoid')
        
        self.activation = tf.nn.relu
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        
    def call(self, x):

        # Attention weights over sequence.
        att = self.linear2(self.dropout(self.linear1(x))) 
        att = tf.transpose(att, perm=[0, 2, 1])
        att = tf.nn.softmax(att, axis=2)

        # Pool sequence to vector using attention weights.
        x_pooled = tf.matmul(att, x)
        x_pooled = tf.squeeze(x_pooled, axis=1)

        # Map vector to output.
        out = self.linear3(x_pooled)
        
        return out

class oldKUModel(tf.keras.Model):

    def __init__(
            self, 
            hidden_dim,  
            dropout_rate=0.2, 
            activation='relu', 
            use_demographics=False,
            num_mmse_groups=15):
        super(KUModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.do_ad = True
        self.do_mmse = False

        # Normalization.
        self.norm = tf.keras.layers.BatchNormalization()

        # Down-projection.
        self.down_proj = tf.keras.layers.Dense(self.hidden_dim, activation=self.activation)
        self.down_proj_drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        # Attention pooling.
        assert self.do_ad or self.do_mmse
        if self.do_ad:
            self.pool_ad = PoolAttFF(
                            self.hidden_dim,
                            self.activation,
                            self.dropout_rate
                            )
        if self.do_mmse:
            self.pool_mmse = PoolAttFF(config, 1)
            self.sigmoid_msme = tf.keras.layers.Activation('sigmoid')

    def call(self, x):

        egemaps_feats, demographics_feats = x
        x = self.norm(egemaps_feats)

        # Down-projection to hidden dim.
        x = self.down_proj(x)
        x = self.down_proj_drop(x)

        ## AD
        if self.do_ad:
            out = self.pool_ad(x)
            return out
        
        ## MMSE
        if self.do_mmse:
            out = self.pool_mmse(x)
            out = tf.squeeze(out, axis=-1)
            out = self.sigmoid_msme(out)
            return out

class DA4ADClassifier(nn.Module):
    def __init__(self):
        super(DA4ADClassifier, self).__init__()

        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960", device_map = self.device)
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        for parameter in self.model.feature_extractor.parameters(): #Uncomment to return to original formulation
            parameter.requires_grad = False
        for parameter in self.model.feature_projection.parameters():
            parameter.requires_grad = False
        # for parameter in self.model.parameters():
        #     parameter.requires_grad = False
        self.model.to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        # self.processor.to(self.device)

        # Define output layers
        self.mci_output_layer = nn.Linear(self.model.config.hidden_size, 1, device=self.device)
        self.gender_output_layer = nn.Linear(self.model.config.hidden_size, 1, device=self.device)

    def forward(self, inputs):
        # Extract batch from inputs
        batch = inputs[0]

        if next(self.parameters()).is_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Prepare batch for processing
        processed_samples = []
        for sample in batch:
            # Convert numpy array to tensor
            # sample_tensor = torch.tensor(sample, dtype=torch.float32)
            sample_tensor = sample.clone().detach()
            data = self.processor(sample_tensor.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            processed_samples.append(data['input_values'])
            # processed_samples.append(data['input_values'].to(self.device))

        # Concatenate all processed samples
        batch_input_values = torch.cat(processed_samples, dim=0)
        
        # Pass data through HubertModel
        outputs = self.model(input_values=batch_input_values).last_hidden_state
        
        # Apply pooling (average over the sequence length dimension)
        mean_outputs = outputs.mean(dim=1)
        
        # Pass pooled outputs through the output layers
        mci_output = self.mci_output_layer(mean_outputs)
        gender_output = self.gender_output_layer(mean_outputs)

        return [mci_output, gender_output]

class MISAClassifier(nn.Module):
    def __init__(
        self, 
        lang_dim, 
        aud_dim):
        super(MISAClassifier, self).__init__()

        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = 128
        self.output_size = 1
        self.dropout_rate = 0.1
        self.arnn1 = nn.LSTM(74, 74, bidirectional=True, device=self.device)
        self.arnn1_layernorm = nn.LayerNorm(2*74, device=self.device)
        self.arnn2 = nn.LSTM(2*74, 74, bidirectional=True, device=self.device)

        #Project input features to dimension 128
        self.lang_proj_1 = nn.Linear(lang_dim, self.hidden_size, device=self.device)
        self.lang_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)
        self.aud_proj_1 = nn.Linear(aud_dim*4, self.hidden_size, device=self.device)
        self.aud_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)

        #Define public encoder that maps 128-->128
        self.public_enc = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define private encoders that map 128-->128
        self.private_lang = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
        self.private_aud = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define reconstruction network that maps 256-->128
        self.reconstruction = nn.Linear(self.hidden_size*2, self.hidden_size, device=self.device)

        #Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        #Define fusion network of transformer outputs
        self.fusion_proj_1 = nn.Linear(self.hidden_size*4, self.hidden_size*2, device=self.device)
        self.fusion_drop_1 = nn.Dropout(self.dropout_rate) #need to activate after here
        self.mci_output_layer = nn.Linear(self.hidden_size*2, self.output_size, device=self.device)

    def forward(self, inputs):

        lang_input, aud_input = inputs
        batch_size = lang_input.shape[0]
        outputs = {}

        lang_proj = self.lang_norm_1(self.tanh(self.lang_proj_1(lang_input)))
        outputs['lang_proj'] = lang_proj

        packed_h1, (final_h1, _) = self.arnn1(aud_input)
        padded_h1, lengths = nn.utils.rnn.pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = self.arnn1_layernorm(padded_h1)

        packed_normed_h1 = nn.utils.rnn.pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        _, (final_h2, _) = self.arnn2(packed_normed_h1) 

        audio_postlstm = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, 1, -1)

        aud_proj = self.aud_norm_1(self.tanh(self.aud_proj_1(audio_postlstm)))
        outputs['aud_proj'] = aud_proj

        lang_public = self.sigmoid(self.public_enc(lang_proj))
        outputs['lang_public'] = lang_public
        aud_public = self.sigmoid(self.public_enc(aud_proj))
        outputs['aud_public'] = aud_public 

        lang_private = self.sigmoid(self.private_lang(lang_proj))
        outputs['lang_private'] = lang_private
        aud_private = self.sigmoid(self.private_aud(aud_proj))
        outputs['aud_private'] = aud_private

        cat_lang = torch.cat((lang_public, lang_private), dim=2)
        cat_aud = torch.cat((aud_public, aud_private), dim=2)

        lang_reconstruction = self.reconstruction(cat_lang)
        outputs['lang_reconstruction'] = lang_reconstruction
        aud_reconstruction = self.reconstruction(cat_aud)
        outputs['aud_reconstruction'] = aud_reconstruction

        lang_public_enc = self.transformer_encoder(lang_public)
        lang_private_enc = self.transformer_encoder(lang_private)
        aud_public_enc = self.transformer_encoder(aud_public)
        aud_private_enc = self.transformer_encoder(aud_private)

        stacked_enc = torch.cat((lang_public_enc, lang_private_enc, aud_public_enc, aud_private_enc), dim=2)

        fusion_proj_1 = self.tanh(self.fusion_drop_1(self.fusion_proj_1(stacked_enc)))

        mci_output = self.mci_output_layer(fusion_proj_1)

        outputs['mci_output'] = mci_output

        return outputs

class MFNClassifier(nn.Module):
    def __init__(self):
        super(MFNClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        [self.d_l, self.d_a] = [300, 74]
        [self.dh_l, self.dh_a] = [64, 16]
        total_h_dim = self.dh_l + self.dh_a 
        self.mem_dim = 128
        window_dim = 2
        output_dim = 1
        attInShape = total_h_dim * window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        h_att1 = 256
        h_att2 = 64
        h_gamma1 = 64
        h_gamma2 = 64
        h_out = 64
        att1_dropout = 0.5
        att2_dropout = 0.5
        gamma1_dropout = 0.5
        gamma2_dropout = 0.5
        out_dropout = 0.5

        # self.lstm_l = nn.LSTM(self.d_l, self.dh_l, batch_first=True, device=self.device)
        # self.lstm_a = nn.LSTM(self.d_a, self.dh_a, batch_first=True, device=self.device)

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l, device=self.device)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a, device=self.device)

        self.att1_fc1 = nn.Linear(attInShape, h_att1, device=self.device)
        self.att1_fc2 = nn.Linear(h_att1, attInShape, device=self.device)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2, device=self.device)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim, device=self.device)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1, device=self.device)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim, device=self.device)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2, device=self.device)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim, device=self.device)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out, device=self.device)
        self.out_fc2 = nn.Linear(h_out, output_dim, device=self.device)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, input_list):
        inputs, lengths = input_list
        lang_input, aud_input = inputs
        lang_lengths, aud_lengths = lengths

        x_l = lang_input
        x_a = aud_input
        lengths = lang_lengths #all lengths are aligned to the number of words in a sentence
        batch_size = x_l.size(0)
        max_seq_len = x_l.size(1)

        self.h_l = torch.zeros(batch_size, self.dh_l).to(self.device)
        self.h_a = torch.zeros(batch_size, self.dh_a).to(self.device)
        self.c_l = torch.zeros(batch_size, self.dh_l).to(self.device)
        self.c_a = torch.zeros(batch_size, self.dh_a).to(self.device)
        self.mem = torch.zeros(batch_size, self.mem_dim).to(self.device)

        for i in range(max_seq_len):
            # Process only valid steps according to sequence length
            valid_batch_mask = i < lengths 
            batch_idx = torch.arange(batch_size)[valid_batch_mask]

            new_h_l, new_c_l = self.lstm_l(x_l[batch_idx, i], (self.h_l[batch_idx], self.c_l[batch_idx]))
            new_h_a, new_c_a = self.lstm_a(x_a[batch_idx, i], (self.h_a[batch_idx], self.c_a[batch_idx]))


            prev_cs = torch.cat([self.c_l[batch_idx], self.c_a[batch_idx]], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a], dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))), dim=1)
            attended = attention * cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem[batch_idx]], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem[batch_idx] = gamma1 * self.mem[batch_idx] + gamma2 * cHat

            # Update hidden and cell states for next timestep
            self.h_l[batch_idx], self.c_l[batch_idx] = new_h_l, new_c_l
            self.h_a[batch_idx], self.c_a[batch_idx] = new_h_a, new_c_a

        # Use the last hidden states and memory cell for final output
        last_hs = torch.cat([self.h_l, self.h_a, self.mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        
        return output

class EASICOGMFNClassifier(nn.Module):
    def __init__(self):
        super(EASICOGMFNClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        [self.d_l, self.d_a] = [768, 103]
        [self.dh_l, self.dh_a] = [32, 8]
        total_h_dim = self.dh_l + self.dh_a 
        self.mem_dim = 64
        window_dim = 2
        output_dim = 1
        attInShape = total_h_dim * window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        h_att1 = 128
        h_att2 = 32
        h_gamma1 = 32
        h_gamma2 = 32
        h_out = 32
        att1_dropout = 0.5
        att2_dropout = 0.5
        gamma1_dropout = 0.5
        gamma2_dropout = 0.5
        out_dropout = 0.5

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l, device=self.device)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a, device=self.device)

        self.att1_fc1 = nn.Linear(attInShape, h_att1, device=self.device)
        self.att1_fc2 = nn.Linear(h_att1, attInShape, device=self.device)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2, device=self.device)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim, device=self.device)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1, device=self.device)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim, device=self.device)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2, device=self.device)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim, device=self.device)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out, device=self.device)
        self.out_fc2 = nn.Linear(h_out, output_dim, device=self.device)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, input_list):
        inputs, lengths = input_list
        lang_input, aud_input = inputs
        lang_lengths, aud_lengths = lengths

        x_l = lang_input
        x_a = aud_input
        lengths = lang_lengths #all lengths are aligned to the number of sentneces in a recording
        batch_size = x_l.size(0)
        max_seq_len = x_l.size(1)

        self.h_l = torch.zeros(batch_size, self.dh_l).to(self.device)
        self.h_a = torch.zeros(batch_size, self.dh_a).to(self.device)
        self.c_l = torch.zeros(batch_size, self.dh_l).to(self.device)
        self.c_a = torch.zeros(batch_size, self.dh_a).to(self.device)
        self.mem = torch.zeros(batch_size, self.mem_dim).to(self.device)

        for i in range(max_seq_len):
            # Process only valid steps according to sequence length
            valid_batch_mask = i < lengths 
            batch_idx = torch.arange(batch_size)[valid_batch_mask]

            new_h_l, new_c_l = self.lstm_l(x_l[batch_idx, i], (self.h_l[batch_idx], self.c_l[batch_idx]))
            new_h_a, new_c_a = self.lstm_a(x_a[batch_idx, i], (self.h_a[batch_idx], self.c_a[batch_idx]))


            prev_cs = torch.cat([self.c_l[batch_idx], self.c_a[batch_idx]], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a], dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))), dim=1)
            attended = attention * cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem[batch_idx]], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem[batch_idx] = gamma1 * self.mem[batch_idx] + gamma2 * cHat

            # Update hidden and cell states for next timestep
            self.h_l[batch_idx], self.c_l[batch_idx] = new_h_l, new_c_l
            self.h_a[batch_idx], self.c_a[batch_idx] = new_h_a, new_c_a

        # Use the last hidden states and memory cell for final output
        last_hs = torch.cat([self.h_l, self.h_a, self.mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        
        return output

class BCLSTMClassifier(nn.Module):
    def __init__(
        self, 
        lang_dim, 
        aud_dim):
        super(BCLSTMClassifier, self).__init__()

        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = 128
        self.output_size = 1
        self.dropout_rate = 0.1
        self.lrnn1 = nn.LSTM(300, self.hidden_size, bidirectional=True, dropout=0.6, device=self.device)
        self.arnn1 = nn.LSTM(74, self.hidden_size, bidirectional=True, dropout=0.6, device=self.device)
        self.arnn1_layernorm = nn.LayerNorm(2*74, device=self.device)
        self.fusion_rnn = nn.LSTM(4*self.hidden_size, self.hidden_size, bidirectional=True, device=self.device)

        #Project input features to dimension 128
        self.lang_proj_1 = nn.Linear(lang_dim, self.hidden_size, device=self.device)
        self.lang_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)
        self.aud_proj_1 = nn.Linear(aud_dim*4, self.hidden_size, device=self.device)
        self.aud_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)

        #Define public encoder that maps 128-->128
        self.public_enc = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define private encoders that map 128-->128
        self.private_lang = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
        self.private_aud = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define reconstruction network that maps 256-->128
        self.reconstruction = nn.Linear(self.hidden_size*2, self.hidden_size, device=self.device)

        #Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        #Define fusion network of transformer outputs
        self.fusion_proj_1 = nn.Linear(self.hidden_size*4, self.hidden_size*2, device=self.device)
        self.fusion_drop_1 = nn.Dropout(self.dropout_rate) #need to activate after here
        self.mci_output_layer = nn.Linear(self.hidden_size*2, self.output_size, device=self.device)

    def forward(self, inputs):

        packed_inputs, lengths = inputs 
        [lang_input, aud_input] = packed_inputs
        [lang_lengths, aud_lengths] = lengths

        #unimodal bi-lstms
        _, (final_lang_h, _) = self.lrnn1(lang_input) #bi-direction lstm for language inputs
        _, (final_aud_h, _) = self.arnn1(aud_input) #bi-directional lstm for audio inputs
        
        #multimodal bi-lstm
        stacked_unimodal = torch.cat((final_lang_h, final_aud_h), dim=2).permute(1, 0, 2).contiguous().view(lang_lengths.shape[0], 1, -1)
        _, (fusion_out, _) = self.fusion_rnn(stacked_unimodal)

        flat_fusion_out = fusion_out.view(1,-1)

        mci_output = self.mci_output_layer(flat_fusion_out)

        return mci_output

class BCLSTMUnimodalClassifier(nn.Module):
    def __init__(
        self, 
        input_dim, 
        ):
        super(BCLSTMUnimodalClassifier, self).__init__()

        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = 300
        self.proj_size = 100
        self.output_size = 1
        self.dropout1 = nn.Dropout(p=0.9)
        self.dropout2 = nn.Dropout(p=0.9)
        self.rnn1 = nn.LSTM(300, self.hidden_size, bidirectional=True, dropout=0.6, device=self.device)
        self.proj_1 = nn.Linear(self.hidden_size*2, self.proj_size, device=self.device)

        self.mci_output_layer = nn.Linear(self.proj_size, self.output_size, device=self.device)

    def forward(self, inputs):

        packed_input, lengths = inputs 

        #unimodal bi-lstms
        _, (inter_h, _) = self.rnn1(packed_input) #bi-direction lstm for unimodal inputs
        stacked_inter_h = torch.cat((inter_h[0],inter_h[1]), dim=-1) #stack bidirectional outputs on the feature dimension
        drop_inter_h = self.dropout1(stacked_inter_h)
        proj_h = self.tanh(self.proj_1(drop_inter_h))
        drop_proj_h = self.dropout2(proj_h)

        mci_output = self.mci_output_layer(drop_proj_h)

        return mci_output, proj_h

class BCLSTMTextCNN(nn.Module):
    def __init__(self, embedding_dim=300, num_classes=1):
        super(BCLSTMTextCNN, self).__init__()
        
        # First convolutional layer: 2 kernels of size 3 and 4, with 50 feature maps each
        self.conv1_3 = nn.Conv2d(1, 50, (3, 3), padding='same')
        self.conv1_4 = nn.Conv2d(1, 50, (4, 4), padding='same')
        
        # Second convolutional layer: 1 kernel of size 2 with 100 feature maps
        self.conv2 = nn.Conv2d(50, 100, (2, 2), padding='same')
        
        # Max pooling layer
        self.pool = nn.MaxPool2d((2, 2))
        
        # Fully connected layer
        self.fc1 = nn.Linear(100 * 25 * 75, 500)  # Adjust input size depending on pooling and conv
        self.fc2 = nn.Linear(500, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        
        # First convolutional layer + pooling + ReLU
        x1 = self.relu(self.pool(self.conv1_3(x))).squeeze(3)
        x2 = self.relu(self.pool(self.conv1_4(x))).squeeze(3)
        
        # Concatenate the outputs of different kernels from the first conv layer
        x = torch.cat((x1, x2), 2)
        
        # Second convolutional layer + pooling + ReLU
        x = self.relu(self.pool(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer + ReLU
        x = self.relu(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x

class BCLSTMMultimodalClassifier(nn.Module):
    def __init__(
        self, 
        lang_dim, 
        aud_dim):
        super(BCLSTMMultimodalClassifier, self).__init__()

        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = 128
        self.output_size = 1
        self.dropout_rate = 0.1
        self.lrnn1 = nn.LSTM(300, self.hidden_size, bidirectional=True, device=self.device)
        self.arnn1 = nn.LSTM(74, self.hidden_size, bidirectional=True, device=self.device)
        self.arnn1_layernorm = nn.LayerNorm(2*74, device=self.device)
        self.fusion_rnn = nn.LSTM(4*self.hidden_size, self.hidden_size, bidirectional=True, device=self.device)

        #Project input features to dimension 128
        self.lang_proj_1 = nn.Linear(lang_dim, self.hidden_size, device=self.device)
        self.lang_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)
        self.aud_proj_1 = nn.Linear(aud_dim*4, self.hidden_size, device=self.device)
        self.aud_norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)

        #Define public encoder that maps 128-->128
        self.public_enc = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define private encoders that map 128-->128
        self.private_lang = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
        self.private_aud = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)

        #Define reconstruction network that maps 256-->128
        self.reconstruction = nn.Linear(self.hidden_size*2, self.hidden_size, device=self.device)

        #Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        #Define fusion network of transformer outputs
        self.fusion_proj_1 = nn.Linear(self.hidden_size*4, self.hidden_size*2, device=self.device)
        self.fusion_drop_1 = nn.Dropout(self.dropout_rate) #need to activate after here
        self.mci_output_layer = nn.Linear(self.hidden_size*2, self.output_size, device=self.device)

    def forward(self, inputs):

        packed_inputs, lengths = inputs 
        [lang_input, aud_input] = packed_inputs
        [lang_lengths, aud_lengths] = lengths

        #unimodal bi-lstms
        _, (final_lang_h, _) = self.lrnn1(lang_input) #bi-direction lstm for language inputs
        _, (final_aud_h, _) = self.arnn1(aud_input) #bi-directional lstm for audio inputs
        
        #multimodal bi-lstm
        stacked_unimodal = torch.cat((final_lang_h, final_aud_h), dim=2).permute(1, 0, 2).contiguous().view(lang_lengths.shape[0], 1, -1)
        _, (fusion_out, _) = self.fusion_rnn(stacked_unimodal)

        flat_fusion_out = fusion_out.view(1,-1)

        mci_output = self.mci_output_layer(flat_fusion_out)

        return mci_output
    
class EASIYingWavClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")

        self.embedding = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Sequential(
            nn.Linear(192, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        out = feature.mean(dim=1)
        # out = self.classifier(feature)

        return out
    
    def extract_embeding(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            # out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        feature = feature.mean(dim=1)
        return feature

class EASIYingBertClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = nn.Sequential(
            nn.Linear(768, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(in_features=192, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        # out = self.activation(self.embedding(out))
        # out = self.classifier(out)
        return out
    
    def extract_embeding(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        # out = self.activation(self.embedding(out))
        return out

class YingWavClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")

        self.embedding = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Sequential(
            nn.Linear(192, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        feature = feature.mean(dim=1)
        out = self.classifier(feature)

        return out
    
    def extract_embeding(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        feature = feature.mean(dim=1)
        return feature

class YingBertClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = nn.Sequential(
            nn.Linear(768, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(in_features=192, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.activation(self.embedding(out))
        out = self.classifier(out)
        return out
    
    def extract_embeding(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.activation(self.embedding(out))
        return out

class YingPredictors():
    def __init__(self):
        self.base_predictors = {
                    'SVM': SVC(probability=True,kernel='rbf'),
                    }

class SarawgiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 24)
        self.batch_norm1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class WhisperClassifier(nn.Module):
    def __init__(
        self,
        whisper_model='tiny'
    ):
        super(WhisperClassifier, self).__init__()
        # Load pre-trained HubertModel and processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tanh = nn.Tanh()
        self.hidden_size = 128
        self.output_size = 1
        self.dropout_rate = 0.1

        self.encoder = whisper.load_model(whisper_model).encoder.to(self.device)

        # Freeze positional embeddings
        self.encoder.positional_embedding.requires_grad = False

        # Freeze all but the last encoder layer
        for layer in self.encoder.blocks[:-1]:
            for param in layer.parameters():
                param.requires_grad = False


        #Project input features
        self.proj_1 = nn.Linear(self.encoder.ln_post.normalized_shape[0], self.hidden_size, device=self.device)
        self.norm_1 = nn.LayerNorm(self.hidden_size, device=self.device)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.proj_2 = nn.Linear(self.hidden_size, 1, device=self.device)


    def forward(self, inputs):

        encoded_inputs = self.encoder(inputs[0]).mean(dim=1) #average features across time

        proj_x = self.tanh(self.proj_1(encoded_inputs))
        norm_x = self.norm_1(proj_x)
        # drop_x = self.dropout(norm_x)
        outputs = self.proj_2(norm_x)

        return outputs

class WhisperRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, bidirectional=True, dropout=0.2):
        super(WhisperRNNClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_directions = 2 if bidirectional else 1

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout, device=self.device)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim, device=self.device)

    def forward(self, x, lengths):
        # Pack the padded sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM forward pass
        packed_output, (hidden, _) = self.lstm(x_packed)

        # Extract last valid hidden state
        if self.bidirectional:
            last_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            last_hidden_state = hidden[-1]

        # Fully connected layer (returns raw logits for BCEWithLogitsLoss)
        out = self.fc(last_hidden_state)

        return out