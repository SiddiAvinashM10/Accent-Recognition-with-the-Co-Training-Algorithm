from keras.models import Sequential
from keras.layers import Dense
from scipy import misc
import convert_WAVtoMFCC
import numpy
from pydub import AudioSegment
from os import path
import os
import imageio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import copy
import time

from FeatureExtraction import create_mfcc_array
from FeatureExtraction import listdir_ignore_hidden



def load_mfccs(mfcc_folder):
    files = listdir_ignore_hidden(mfcc_folder)
    mfcc_arr = []
    for file in files:
        file_path = os.path.join(mfcc_folder, file)
        mfcc = imageio.imread(file_path)
        mfcc_arr.append(mfcc)
    return mfcc_arr

def load_wavs(wav_folder, split_num):
    files = listdir_ignore_hidden(wav_folder)
    wav_arr = []
    for file in files:
        file_path = os.path.join(wav_folder, file)
        audio_seg = AudioSegment.from_file(file_path)
        array = audio_seg.get_array_of_samples()
        array = array[::split_num]
        wav_arr.append(array)
    return wav_arr


#Takes raw MFCC, or Wav arrays, appends them into one array and assigns Labels
#First argument gets label 0, second gets label 1, etc.
def create_array_and_labels(*argv):
    num_classes = len(argv)
    non_empty_arrays = True
    data_array = []
    label_array = []
    while(non_empty_arrays):
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        if(len(argv[arr_choice]) != 0):
            data_array.append(argv[arr_choice].pop())
            label_array.append(arr_choice)
        for arg in argv:
            if len(arg) != 0:
                non_empty_arrays = True

    data_array = numpy.asarray(data_array)
    data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], data_array.shape[2], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (data_array, label_array)

#Takes array of data (MFCC's or Chromas), and an array of their Labels
#Samples out sample_size number of elements
def uniform_random_sampling(data, labels, sample_size):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    data_size = len(complement_data)
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
    count2 = 0
    for i in range(0, sample_size):
        arr_choice = numpy.random.randint(0,data_size-i)
        count2 = count2 + 1
        if(count2 % 1000 == 0):
            print("Sampled " + str(count2) + " out of " + str(sample_size))
        sampled_data[i] = complement_data[arr_choice]
        complement_data = numpy.delete(complement_data, arr_choice, 0)
        sampled_labels[i] = complement_labels[arr_choice]
        complement_labels = numpy.delete(complement_labels, arr_choice, 0)
    return (sampled_data, sampled_labels, complement_data, complement_labels)


#Samples out num_folds folds from data and labels
#returns them in an array d
def sample_for_k_folds(data, labels, num_folds):
    assert(len(data) == len(labels))
    complement_data = copy.deepcopy(data)
    complement_labels = copy.deepcopy(labels)
    sample_size = len(data)//num_folds
    folds_sampled_data=[]
    folds_sampled_data=numpy.asarray(folds_sampled_data)
    folds_sampled_labels=[]
    folds_sampled_labels=numpy.asarray(folds_sampled_labels)
    d={}
    count = 1
    for fold in range(num_folds):
        sampled_data, sampled_labels, complement_data, complement_labels = uniform_random_sampling(complement_data, complement_labels, sample_size)
        print("Sampled " + str(count) + " out of " + str(num_folds) + " folds")
        d["folds_sampled_data_{}".format(fold)]=sampled_data
        d["folds_sampled_labels_{}".format(fold)]=sampled_labels
        count = count + 1
    return (d)

#Allows a model to make a prediction on an entire MP3 file instead of a single chunks
#by performing a vote on classifications of its chunks
def predict_class(model,num_classes):
    print("--------------------------The Prediction--------------------------------------")
    prediction_mfcc_arr = create_mfcc_array(os.path.join(os.path.dirname(__file__),'Audio_Data', 'prediction', 'predictionWAV'))
    predict_data,label = create_array_and_labels(prediction_mfcc_arr)
    predictions = model.predict(predict_data)
    rounded = [(x[0]) for x in predictions]
    print("prediction [0] = ",rounded)
    rounded1 = [(x[1]) for x in predictions]
    print("prediction [1] = ",rounded1)
    val=[0 if x[0]>x[1] else 1 for x in predictions]
    print(val)
    count_of_classes=[]
    print("--------------------------The Prediction--------------------------------------")



#Trains first model for k-folds
def train_model1(sampled_train_data, sampled_train_labels, sampled_test_data, sampled_test_labels):
    print("BEGINNING TRAINING.....")

    input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(input_shape)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    num_classes = 3

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
    print(score)
    test_accuracy=score[1]
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return test_accuracy
    #Accuracy Array =  [0.6523404254304602, 0.6453191489868976, 0.6487234041031371, 0.6561702127152301, 0.644042553140762]
    #predict_class(model,num_classes)

#Trains second model for k-folds
def train_model2(sampled_train_data, sampled_train_labels, sampled_test_data, sampled_test_labels):
    print("BEGINNING TRAINING.....")

    input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(input_shape)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    num_classes = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
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
    print(score)
    test_accuracy=score[1]
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return test_accuracy
    #Accuracy Array =  [0.2680851064337061, 0.2757446809271549, 0.41319148931097477, 0.326595744687192, 0.31319148933633845]
    #predict_class(model,num_classes)



#Trains third model for k-folds
def train_model3(sampled_train_data, sampled_train_labels, sampled_test_data, sampled_test_labels):
    print("BEGINNING TRAINING.....")

    input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(input_shape)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    num_classes = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
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
    print(score)
    test_accuracy=score[1]
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return test_accuracy

'''
#Perform k-folds on MFCC

print("Testing Create Array")

pwd = os.path.dirname(os.path.realpath(__file__))


na_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
ind_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
china_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))

data_arr, label_arr = create_array_and_labels(na_mfcc_arr, ind_mfcc_arr, china_mfcc_arr)

d = sample_for_k_folds(data_arr, label_arr, 5)

#folds_sampled_data=numpy.asarray(folds_sampled_data)
#folds_sampled_labels=numpy.asarray(folds_sampled_labels)
print("???????????????????????????????????????????????????????????????????????")
print("data shape: ",data_arr.shape)
print("label shape: ",label_arr.shape)
print("???????????????????????????????????????????????????????????????????????")
accuracy_array1 = []
accuracy_array2 = []
accuracy_array3 = []
for i in range(5):
    sampled_test_data=d["folds_sampled_data_{}".format(i)]
    sampled_test_labels=d["folds_sampled_labels_{}".format(i)]
    k=(i+1)%5
    sampled_train_data = d["folds_sampled_data_{}".format(k)]
    print("Initial shape = ",sampled_train_data.shape)
    #sampled_train_data = sampled_train_data.reshape(3165, 22, 22, 1)
    sampled_train_labels = d["folds_sampled_labels_{}".format(k)]
    print("Initial shape = ",sampled_train_labels.shape)
    print(sampled_test_data.shape)
    print(sampled_test_labels.shape)
    for j in range(5):
        if j==i or j==k:
            continue
        else:
            sampled_train_data = numpy.append(sampled_train_data,d["folds_sampled_data_{}".format(j)],axis=0)
            sampled_train_labels = numpy.append(sampled_train_labels,d["folds_sampled_labels_{}".format(j)],axis=0)
    print(sampled_train_data.shape)
    print(sampled_train_labels.shape)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    score1 = train_model1(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    score2 = train_model2(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    score3 = train_model3(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(score)
    accuracy_array1.append(score1)
    accuracy_array2.append(score2)
    accuracy_array3.append(score3)

print("Accuracy Array 1 = ", accuracy_array1)
print("Accuracy Array 2 = ", accuracy_array2)
print("Accuracy Array 3 = ", accuracy_array3)
'''


#Perform k-folds on Chroma
'''
print("Testing Create Array")

pwd = os.path.dirname(os.path.realpath(__file__))


na_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
ind_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
china_mfcc_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))

data_arr, label_arr = create_array_and_labels(na_mfcc_arr, ind_mfcc_arr, china_mfcc_arr)

d = sample_for_k_folds(data_arr, label_arr, 5)

#folds_sampled_data=numpy.asarray(folds_sampled_data)
#folds_sampled_labels=numpy.asarray(folds_sampled_labels)
print("???????????????????????????????????????????????????????????????????????")
print("data shape: ",data_arr.shape)
print("label shape: ",label_arr.shape)
print("???????????????????????????????????????????????????????????????????????")
accuracy_array1 = []
accuracy_array2 = []
accuracy_array3 = []
for i in range(5):
    sampled_test_data=d["folds_sampled_data_{}".format(i)]
    sampled_test_labels=d["folds_sampled_labels_{}".format(i)]
    k=(i+1)%5
    sampled_train_data = d["folds_sampled_data_{}".format(k)]
    print("Initial shape = ",sampled_train_data.shape)
    #sampled_train_data = sampled_train_data.reshape(3165, 22, 22, 1)
    sampled_train_labels = d["folds_sampled_labels_{}".format(k)]
    print("Initial shape = ",sampled_train_labels.shape)
    print(sampled_test_data.shape)
    print(sampled_test_labels.shape)
    for j in range(5):
        if j==i or j==k:
            continue
        else:
            sampled_train_data = numpy.append(sampled_train_data,d["folds_sampled_data_{}".format(j)],axis=0)
            sampled_train_labels = numpy.append(sampled_train_labels,d["folds_sampled_labels_{}".format(j)],axis=0)
    print(sampled_train_data.shape)
    print(sampled_train_labels.shape)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    score1 = train_model1(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    score2 = train_model2(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    score3 = train_model3(sampled_train_data,sampled_train_labels,sampled_test_data,sampled_test_labels)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(score)
    accuracy_array1.append(score1)
    accuracy_array2.append(score2)
    accuracy_array3.append(score3)

print("Accuracy Array 1 = ", accuracy_array1)
print("Accuracy Array 2 = ", accuracy_array2)
print("Accuracy Array 3 = ", accuracy_array3)
'''
