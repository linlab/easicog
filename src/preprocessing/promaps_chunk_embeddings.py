import json 
import argparse
import librosa
import torch
import os
import glob
import torch
import pandas as pd
import numpy as np 
from pathlib import Path
from disvoice.prosody.prosody import Prosody 
from disvoice.prosody.prosody_functions import V_UV, F0feat, energy_cont_segm, polyf0, energy_feat, dur_seg, duration_feat, get_energy_segment
import pysptk
from joblib import Parallel, delayed
from src.utils.utils import denoise_audio, loudnorm_audio 
import pyloudnorm as pyln 
import opensmile



class CustomProsody(Prosody):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pitch_method = "rapt"
        self.size_frame = 0.02
        self.step = 0.01
        self.thr_len = 0.14
        self.minf0 = 60
        self.maxf0 = 350
        self.voice_bias = -0.2
        self.P = 5
        self.namefeatf0 = ["F0avg", "F0std", "F0max", "F0min",
                           "F0skew", "F0kurt", "F0tiltavg", "F0mseavg",
                           "F0tiltstd", "F0msestd", "F0tiltmax", "F0msemax",
                           "F0tiltmin", "F0msemin", "F0tiltskw", "F0mseskw",
                           "F0tiltku", "F0mseku", "1F0mean", "1F0std",
                           "1F0max", "1F0min", "1F0skw", "1F0ku", "lastF0avg",
                           "lastF0std", "lastF0max", "lastF0min", "lastF0skw", "lastF0ku"]
        self.namefeatEv = ["avgEvoiced", "stdEvoiced", "skwEvoiced", "kurtosisEvoiced",
                           "avgtiltEvoiced", "stdtiltEvoiced", "skwtiltEvoiced", "kurtosistiltEvoiced",
                           "avgmseEvoiced", "stdmseEvoiced", "skwmseEvoiced", "kurtosismseEvoiced",
                           "avg1Evoiced", "std1Evoiced", "max1Evoiced", "min1Evoiced", "skw1Evoiced",
                           "kurtosis1Evoiced", "avglastEvoiced", "stdlastEvoiced", "maxlastEvoiced",
                           "minlastEvoiced", "skwlastEvoiced",  "kurtosislastEvoiced"]
        self.namefeatEu = ["avgEunvoiced", "stdEunvoiced", "skwEunvoiced", "kurtosisEunvoiced",
                           "avgtiltEunvoiced", "stdtiltEunvoiced", "skwtiltEunvoiced", "kurtosistiltEunvoiced",
                           "avgmseEunvoiced", "stdmseEunvoiced", "skwmseEunvoiced", "kurtosismseEunvoiced",
                           "avg1Eunvoiced", "std1Eunvoiced", "max1Eunvoiced", "min1Eunvoiced", "skw1Eunvoiced",
                           "kurtosis1Eunvoiced", "avglastEunvoiced", "stdlastEunvoiced", "maxlastEunvoiced",
                           "minlastEunvoiced", "skwlastEunvoiced",  "kurtosislastEunvoiced"]

        self.namefeatdur = ["Vrate", "avgdurvoiced", "stddurvoiced", "skwdurvoiced", "kurtosisdurvoiced", "maxdurvoiced", "mindurvoiced",
                            "avgdurunvoiced", "stddurunvoiced", "skwdurunvoiced", "kurtosisdurunvoiced", "maxdurunvoiced", "mindurunvoiced",
                            "avgdurpause", "stddurpause", "skwdurpause", "kurtosisdurpause", "maxdurpause", "mindurpause",
                            "PVU", "PU", "UVU", "VVU", "VP", "UP"]
        self.head_st = self.namefeatf0+self.namefeatEv+self.namefeatEu+self.namefeatdur

        self.namef0d = ["f0coef"+str(i) for i in range(6)]
        self.nameEd = ["Ecoef"+str(i) for i in range(6)]
        self.head_dyn = self.namef0d+self.nameEd+["Voiced duration"]

    def my_extract_static_features(self, audio, fs, plots=False, fmt='npy'):
        features = self.my_prosody_static(audio, fs, plots)
        if fmt in ("npy", "txt"):
            return features
        elif fmt in ("dataframe", "csv"):
            df = {}
            for e, k in enumerate(self.head_st):
                df[k] = [features[e]]
            return pd.DataFrame(df)
        elif fmt == "torch":
            feat_t = torch.from_numpy(features)
            return feat_t
        elif fmt == "kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")
        raise ValueError("format" + fmt+" is not supported")

    def my_prosody_static(self, audio, fs, plots=False):
        """Extract the static prosody features from an audio file

        :param audio: numpy array containing audio
        :param fs: sampling rate of audio
        :param plots: timeshift to extract the features
        :returns: array with the 103 prosody features


        """
        # fs, data_audio = read(audio)
        data_audio = audio
        fs = fs 

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)
        data_audio = data_audio-np.mean(data_audio)
        data_audio = data_audio/float(np.max(np.abs(data_audio)))
        size_frameS = self.size_frame*float(fs)
        size_stepS = self.step*float(fs)
        thr_len_pause = self.thr_len*float(fs)

        if self.pitch_method == 'praat':
            name_audio = audio.split('/')
            temp_uuid = 'prosody'+name_audio[-1][0:-4]
            if not os.path.exists(PATH+'/../tempfiles/'):
                os.makedirs(PATH+'/../tempfiles/')
            temp_filename_f0 = PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            temp_filename_vuv = PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv,
                                      time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)

            F0, _ = praat_functions.decodeF0(
                temp_filename_f0, len(data_audio)/float(fs), self.step)
            os.remove(temp_filename_f0)
            os.remove(temp_filename_vuv)
        elif self.pitch_method == 'rapt':
            data_audiof = np.asarray(data_audio*(2**15), dtype=np.float32)
            F0 = pysptk.sptk.rapt(
                data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

        segmentsV = V_UV(F0, data_audio, type_seg="Voiced",
                         size_stepS=size_stepS)
        segmentsUP = V_UV(F0, data_audio, type_seg="Unvoiced",
                          size_stepS=size_stepS)

        segmentsP = []
        segmentsU = []
        for k in range(len(segmentsUP)):
            if (len(segmentsUP[k]) > thr_len_pause):
                segmentsP.append(segmentsUP[k])
            else:
                segmentsU.append(segmentsUP[k])

        F0_features = F0feat(F0)
        energy_featuresV = energy_feat(segmentsV, fs, size_frameS, size_stepS)
        energy_featuresU = energy_feat(segmentsU, fs, size_frameS, size_stepS)
        duration_features = duration_feat(
            segmentsV, segmentsU, segmentsP, data_audio, fs)

        if plots:
            self.plot_pros(data_audio, fs, F0, segmentsV,
                           segmentsU, F0_features)

        features = np.hstack(
            (F0_features, energy_featuresV, energy_featuresU, duration_features))

        return features


def embed_audio(filepath, timestamp_json, output_dir, extractor, noisereduce_audio, normalize_audio):
    # Load the audio file
    smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.eGeMAPSv02,
      feature_level=opensmile.FeatureLevel.Functionals)
    sr = 44100
    meter = pyln.Meter(sr)
    chunk_size = 30
    overlap_duration = 0
    samples_per_chunk = chunk_size * sr
    overlap_samples = overlap_duration * sr
    audio, _ = librosa.load(filepath, sr=sr, mono=True)

    if noisereduce_audio:
        audio = denoise_audio(audio, sr)
    if normalize_audio:
        audio = loudnorm_audio(audio, meter)
    recording_name = filepath.split('/')[-1][:-4]

    if len(audio) < chunk_size*sr:
        print(f"File {recording_name} has been padded to fill {chunk_size} seconds")
        audio = np.pad(audio, (0, chunk_size*sr - len(audio) + 1), 'constant')
    
    total_duration = len(audio) / sr
    total_chunks = int((total_duration - chunk_size) / (chunk_size - overlap_duration)) + 1

    # Iterate through each chunk 
    jj = 0
    for i in range(total_chunks):
        # Calculate start and end sample indices for the current chunk
        start = i * (samples_per_chunk - overlap_samples)
        end = start + samples_per_chunk
        
        # Extract the current chunk
        chunk = audio[start:end]

        # Extract features from the chunk
        prosody_features = np.nan_to_num(extractor.my_extract_static_features(chunk, sr))[None,:]

        _, _, features = smile.process(chunk,sampling_rate=sr)
        egemaps_features = np.nan_to_num(features)

        out_features = np.hstack((prosody_features,egemaps_features))
      
        # Save the features
        output_filename = os.path.join(output_dir, f"{recording_name}_{jj:04}.npy")
        jj+=1
        np.save(output_filename, out_features)


    if len(chunk) != samples_per_chunk:
        print(f'Error processing file {filepath}')
        exit()


def get_prosody_sentence_embeddings(
        timestamped_transcription_dir, 
        audio_dir, 
        prosody_sentence_outdir,
        noisereduce_audio,
        normalize_audio 
        ):
    # Get filepaths
    filepaths = glob.glob(os.path.join(audio_dir,'*.wav'))
    filepaths.sort()

    # Initialize the Disvoice Prosody extractor
    extractor = CustomProsody()

    # Define a directory to save the features
    output_dir = prosody_sentence_outdir
    os.makedirs(output_dir, exist_ok=True)

    for filepath in filepaths:
        print(f'Embedding {filepath}')
        embed_audio(
            filepath, 
            None,
            output_dir, 
            extractor,
            noisereduce_audio,
            normalize_audio
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    noisereduce_audio = config['noisereduce_audio']
    normalize_audio = config['normalize_audio']

    for dataset_config in config["datasets"]:
    # for dataset_config in [config["datasets"][1]]:
        print(dataset_config['setname'])
        get_prosody_sentence_embeddings(
            timestamped_transcription_dir=None,
            audio_dir=dataset_config['audio_dir'], 
            prosody_sentence_outdir=dataset_config['promaps_chunk_outdir'],
            noisereduce_audio = False,
            normalize_audio = False 
            )
