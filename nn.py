from keras.models import Sequential
from keras.layers import Dense
from scipy import misc
import numpy
from pydub import AudioSegment
from os import path
import os
import imageio
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, RNN, SimpleRNNCell, Embedding
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from keras import backend as K
import copy
import time
import array

from FeatureExtraction import create_mfcc_array, create_chroma_array
from FeatureExtraction import listdir_ignore_hidden
import convert_WAVtoMFCC



pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

###########################__SAMPLING_METHODS_USED_BY_BOTH__#########################


def uniform_random_sampling(data, labels, sample_size):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    data_size = len(complement_data)
    print("In Uniform Random Sampling")
    sampled_data_shape = [sample_size]
    count = 0
    for s in complement_data.shape:
        if(count!=0):
            sampled_data_shape.append(s)
        count = count+1
    sampled_data_shape = tuple(sampled_data_shape)
    sampled_labels_shape = [sample_size]
    count = 0
    for s in complement_labels.shape:
        if(count!=0):
            sampled_labels_shape.append(s)
        count = count + 1
    sampled_labels_shape = tuple(sampled_labels_shape)
    sampled_data = numpy.zeros(shape=(sampled_data_shape))
    sampled_labels = numpy.zeros(shape=sampled_labels_shape)
    print("About to sample...")
    for i in range(0, sample_size):
        if(i % 50 == 0):
            print("i: " + str(i))
        arr_choice = numpy.random.randint(0,data_size-i)
        sampled_data[i] = complement_data[arr_choice]
        complement_data = numpy.delete(complement_data, arr_choice, 0)
        sampled_labels[i] = complement_labels[arr_choice]
        complement_labels = numpy.delete(complement_labels, arr_choice, 0)
    return (sampled_data, sampled_labels, complement_data, complement_labels)



def sample_for_k_folds(data, labels, num_folds):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    print("IN K FOLDS SAMPLE")
    print("LENGTH: " + str(len(data)))
    sample_size = len(data)//num_folds
    print("SAMPLE SIZE: " + str(sample_size))
    folds_arr = []
    for fold in range(num_folds):
        sampled_data, sampled_labels, complement_data, complement_labels = uniform_random_sampling(complement_data, complement_labels, sample_size)
        folds_arr.append([sampled_data, sampled_labels])
    return folds_arr




###########################__MFCC_ANALYIS_BELOW__#########################


def load_mfccs(mfcc_folder): #Currently Unused
    files = listdir_ignore_hidden(mfcc_folder)
    mfcc_arr = []
    for file in files:
        file_path = os.path.join(mfcc_folder, file)
        mfcc = imageio.imread(file_path)
        mfcc_arr.append(mfcc)
    return mfcc_arr



#MFCCs
na_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
ind_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
china_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))


#CHROMAs
#na_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
#ind_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
#china_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))

#0 Label Corresponds to North America
#1 Label Corresponds to India



def create_array_and_labels(*argv):
    num_classes = len(argv)
    non_empty_arrays = True
    data_array = []
    label_array = []
    count = 0
    while(non_empty_arrays):
        count = count + 1
        if(count % 1000 == 0):
            print("On iteration: " + str(count))
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        if(len(argv[arr_choice]) != 0):
            data_array.append(argv[arr_choice].pop())
            label_array.append(arr_choice)
        for arg in argv:
            if len(arg) != 0:
                non_empty_arrays = True
    data_array = numpy.asarray(data_array)



    if(len(data_array.shape) != len(data_array[0].shape)+1):
        new_shape = (data_array.shape[0],) + data_array[0].shape
        new_list = []
        for i in range(data_array.shape[0]):
            for j in data_array[i].flatten().tolist():
                new_list.append(j)
        new_list = numpy.asarray(new_list)
        data_array = new_list.reshape(new_shape)

    data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], data_array.shape[2], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (data_array, label_array)








data_arr, label_arr = create_array_and_labels(na_mfcc_arr, ind_mfcc_arr, china_mfcc_arr)

sampled_test_data, sampled_test_labels, sampled_train_data, sampled_train_labels  = uniform_random_sampling(data_arr, label_arr, 500)


print(sampled_train_data.shape)
print(sampled_train_labels.shape)
print(sampled_test_data.shape)
print(sampled_test_labels.shape)








print("BEGINNING MFCC TRAINING.....")




input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
num_classes = sampled_test_labels.shape[1]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 100

model.fit(sampled_train_data, sampled_train_labels,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(sampled_test_data, sampled_test_labels))

score = model.evaluate(sampled_test_data, sampled_test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


print(model.predict(sampled_test_data))
