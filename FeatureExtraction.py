from os import path, listdir
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import math
import statistics
import time
import re
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from pydub.playback import play
from scipy.io.wavfile import read
from convert_WAVtoMFCC import convert_WAVtoMFCC, create_mfcc



#List all files in a directory except hidden files
#This does not include subdirectories
def listdir_ignore_hidden(path):
    files = []
    for file in os.listdir(path):
        if not file.startswith('.') and not os.path.isdir(os.path.join(path,file)):
            files.append(file)
    return files







#INPUT: 1) Fully qualified name of an mp3 file 2) Fully qualified path where chunks will be stored as .wav files
#OUTPUT: Array of "chunks" of the audio in wav format saved in chunk_folder
def extract_chunks(mp3File):
    wavFolder = path.dirname(__file__) + "/Audio_Data/Wav_Data/"
    wavFile = convert_to_wav(mp3File, wavFolder) # <fileName>.mp3 --> <fileName>.wav
    avgDb = calculate_avg_db(wavFile)
    silence_thresh = -(0.8 * avgDb) #Threshold of decibels to count as silence
    segment = AudioSegment.from_wav(wavFile)
    chunks = split_on_silence(segment, min_silence_len = 100, silence_thresh = silence_thresh, keep_silence = 200)
    file_name_base = extract_file_name(mp3File)
    return chunks


#INPUT: 1) An array of "chunks" (audio segments created from extract_chunks)
#OUTPUT: One audio segment created by combining all the chunks
def combine_chunks(chunks):
    combined_chunk = chunks[0]
    for chunk in chunks:
        if chunk != chunks[0]:
                combined_chunk = combined_chunk.append(chunk, crossfade = 0)
    return combined_chunk


#INPUT: 1) An audio segment 2) The length of each window in milliseconds
#OUTPUT: An array of "windows"
def extract_windows(audio, window_size):
    windows = audio[::window_size]
    window_arr = []
    for window in windows:
        if(len(window) == window_size):
            window_arr.append(window)
    return window_arr


#INPUT: 1) An array of windows (from extract_windows) 2) Fully qualified path name to save windows in 3) Name of file windows were generated from
#OUTPUT: Each window will be saved as a .wav file in the window_folder
# Ex: (windows, "/files/audiofiles/windows/", "test_windows") --> test_windows_0, test_windows_1, etc. in /windows/ directory
def save_windows(windows, window_folder, file_name):
    if not os.path.exists(window_folder):
        os.makedirs(window_folder)
    window_count = 0
    for window in windows:
        output_file_name = os.path.join(window_folder,file_name + "_" + str(window_count) + ".wav")
        window.export(output_file_name, format="wav")
        window_count = window_count + 1
    return



#INPUT: 1) Fully qualified name of mp3 file, 2) Fully qualified name of folder in which wav file will be created
#OUTPUT: A wav version of the file is created in the given output folder, and the path to this file is returned
#i.e. hello.mp3 -> hello.wav
def convert_to_wav(mp3File, output_folder):
    file_name = extract_file_name(mp3File)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_name = os.path.join(output_folder,file_name + ".wav")
    AudioSegment.from_mp3(mp3File).export(output_file_name, format="wav")
    return output_file_name

#INPUT: 1) Fully qualified name of wav file
#OUTPUT: Average volume in dB of file
def calculate_avg_db(wavFile):
    wavdata = (read(wavFile))
    wavdata = wavdata[1]
    chunks = np.array_split(wavdata, 1)
    dbs = [20*math.log10( math.sqrt(statistics.mean(chunk**2)**2) ) for chunk in chunks]
    return dbs[0]

#INPUT: Fully Qualified File name
#OUTPUT: Name of file
#Ex: /files/audio_files/audio_file.mp3 --> audio_file
def extract_file_name(file_name):
    file_name = os.path.basename(file_name)
    file_name = file_name.split('.')[0]
    return file_name


#INPUT: 1)Fully qualified path to folder with MP3 Files 2) Fully qualified path to where .wav windows will be saved 3) Desired length of each window (in ms)
#OUTPUT: All mp3 files in mp3_folder will be converted to windows and saved as .wav
def convert_folder_to_windows(mp3_folder, destination_folder, window_size):
    directory = destination_folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = listdir_ignore_hidden(mp3_folder)
    for file in files:
        file_name = extract_file_name(file)
        chunks = extract_chunks(os.path.join(mp3_folder,file))
        combined = combine_chunks(chunks)
        windows = extract_windows(combined, window_size)
        count = 0
        save_windows(windows, destination_folder, file_name)
    return


#Input: Fully qualified path name to folder containing .wav files (window_folder), Fully qualified path name to new folder where MFCCs to be saved
#Output: Folder of mfcc data created from .wav files
def convert_folder_to_mfcc(wav_folder, mfcc_folder):
    files = listdir_ignore_hidden(wav_folder)
    for file in files:
        print(file)
        file_name = extract_file_name(file)
        convert_WAVtoMFCCWAVasMFCC(os.path.join(wav_folder,file), mfcc_folder)
    return


def create_mfcc_array(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    mfcc_arr = []
    count = 0
    for file in files:
        count = count+1
        mfcc_arr.append(create_mfcc(os.path.join(wav_folder, file)))
        if(count % 1000 == 0):
            print("Converting file " + str(count) + " out of " + str(len(files)))
    print("Converted :" + str(len(files)))
    return mfcc_arr



import librosa
from librosa.feature.spectral import chroma_stft

def create_chroma_array(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    chroma_arr = []
    count = 0
    pad_count = 0
    for file in files:
        count = count+1
        if(count % 1000 == 0):
            print("Converted " + str(count) + " out of " + str(len(files)))
        sound = AudioSegment.from_file(os.path.join(wav_folder, file))
        samples = sound.get_array_of_samples()
        padding = False
        while(len(samples) < 22050): #Add zero padding if wav not long enough
            padding = True
            samples.append(0)
        float_samples = []
        for i in range(len(samples)):
            float_samples.append(float(samples[i]))
        samples = np.asarray(float_samples)
        chroma = chroma_stft(samples)
        if(chroma.shape[0] != 12 or chroma.shape[1] != 44):
            chroma = chroma[0:12,0:44]
        chroma = np.swapaxes(chroma,0,1)
        chroma_arr.append(chroma)
    print("Converted :" + str(len(files)))
    return chroma_arr





#Creates a dictionary that contains an array of both the mfcc and chroma feature arrays
#Feature Dictionary looks like {'MFCC': <mfcc_feature_array>, 'Chroma': <chroma_feature_array>}
#Need to do this so that mfcc_feature_array[i] and chroma_feature_array[i] were generated -
#- from the same .wav file (Need this for cotraining)
def create_feature_dictionary(wav_folder):
    files = listdir_ignore_hidden(wav_folder)
    mfcc_arr = []
    chroma_arr = []
    print(len(files))
    count = 0
    for file in files:
        if(count%1000==0):
            print("Converted " + str(count) + " out of " + str(len(files)))
        count = count+1
        mfcc = create_mfcc(os.path.join(wav_folder, file))

        mfcc_arr.append(mfcc)

        #CHROMA
        sound = AudioSegment.from_file(os.path.join(wav_folder, file))
        samples = sound.get_array_of_samples()
        padding = False
        while(len(samples) < 22050): #Add zero padding if wav not long enough
            padding = True
            samples.append(0)
        float_samples = []
        for i in range(len(samples)):
            float_samples.append(float(samples[i]))
        samples = np.asarray(float_samples)
        chroma = chroma_stft(samples)

        if(chroma.shape[0] != 12 or chroma.shape[1] != 44):
            chroma = chroma[0:12,0:44]
        chroma = np.swapaxes(chroma,0,1)


        chroma_arr.append(chroma)


    print("Converted :" + str(len(files)))
    feature_dict = {'MFCC': mfcc_arr, 'Chroma': chroma_arr}
    return feature_dict
