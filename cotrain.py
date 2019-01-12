from keras.models import Sequential
from keras.layers import Dense
from scipy import misc
import numpy
from pydub import AudioSegment
from os import path
import os
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, RNN, SimpleRNNCell, Embedding
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Lambda
from keras import backend as K
import copy
import time
import array

import convert_WAVtoMFCC
from FeatureExtraction import listdir_ignore_hidden
from FeatureExtraction import create_mfcc_array, create_chroma_array, create_feature_dictionary






#Takes dictionaries containing mfcc and chroma data for each class
#Returns a dictionary containing all the MFCC, Chromas, and Class labels
def create_array_and_labels_for_cotraining(*argv):
    num_classes = len(argv)
    non_empty_arrays = True
    mfcc_data_array = []
    chroma_data_array = []
    label_array = []
    count = 0
    while(non_empty_arrays):
        non_empty_arrays = False
        arr_choice = numpy.random.randint(0,num_classes)
        if (len(argv[arr_choice]['MFCC']) != 0) and (len(argv[arr_choice]['MFCC']) == len(argv[arr_choice]['Chroma'])):
            mfcc_data_array.append(argv[arr_choice]['MFCC'].pop())
            chroma_data_array.append(argv[arr_choice]['Chroma'].pop())
            label_array.append(arr_choice)
        for arg in argv:
            if (len(arg['MFCC']) != 0) or (len(arg['Chroma']) != 0):
                non_empty_arrays = True
        count = count + 1
    mfcc_data_array = numpy.asarray(mfcc_data_array)
    chroma_data_array = numpy.asarray(chroma_data_array)


    #ADDED ADDITIONAL FORMATTING (ONLY EXECUTES ON CHROMA DATA)
    if(len(chroma_data_array.shape) != len(chroma_data_array[0].shape)+1):
        new_shape = (chroma_data_array.shape[0],) + chroma_data_array[0].shape
        new_list = []
        for i in range(chroma_data_array.shape[0]):
            for j in chroma_data_array[i].flatten().tolist():
                new_list.append(j)
        new_list = numpy.asarray(new_list)
        chroma_data_array = new_list.reshape(new_shape)

    mfcc_data_array = mfcc_data_array.reshape(mfcc_data_array.shape[0], mfcc_data_array.shape[1], mfcc_data_array.shape[2], 1)
    chroma_data_array = chroma_data_array.reshape(chroma_data_array.shape[0], chroma_data_array.shape[1], chroma_data_array.shape[2], 1)
    label_array = numpy.asarray(label_array)
    label_array = keras.utils.to_categorical(label_array, num_classes)
    return (mfcc_data_array, chroma_data_array, label_array)


#Samples sample_size number of samples from the given arrays
#Ensures that mfccs and chromas corresponding to the same audio sample stay together
def uniform_random_sampling_for_cotraining(mfcc_data, chroma_data, labels, sample_size):
    assert(len(mfcc_data) == len(labels) and len(mfcc_data) == len(chroma_data))
    complement_mfcc_data = copy.deepcopy(mfcc_data)
    complement_chroma_data = copy.deepcopy(chroma_data)
    complement_labels = copy.deepcopy(labels)
    data_size = len(complement_mfcc_data)

    sampled_mfcc_data_shape = [sample_size]
    count = 0
    for s in complement_mfcc_data.shape:
        if(count!=0):
            sampled_mfcc_data_shape.append(s)
        count = count+1
    sampled_mfcc_data_shape = tuple(sampled_mfcc_data_shape)



    #Determine Shape of sampled Chroma data
    sampled_chroma_data_shape = [sample_size]
    count = 0
    for s in complement_chroma_data.shape:
        if(count!=0):
            sampled_chroma_data_shape.append(s)
        count = count+1
    sampled_chroma_data_shape = tuple(sampled_chroma_data_shape)


    #Determine Shape of sampled Chroma data
    #Should end up being (sample_size, n_classes)
    sampled_labels_shape = [sample_size]
    count = 0
    for s in complement_labels.shape:
        if(count!=0):
            sampled_labels_shape.append(s)
        count = count + 1
    sampled_labels_shape = tuple(sampled_labels_shape)


    sampled_mfcc_data = numpy.zeros(shape=(sampled_mfcc_data_shape))
    sampled_chroma_data = numpy.zeros(shape=(sampled_chroma_data_shape))
    sampled_labels = numpy.zeros(shape=sampled_labels_shape)
    for i in range(0, sample_size):
        if(i%1000 == 0):
            print("Sampled " + str(i) + " out of " + str(sample_size))
        arr_choice = numpy.random.randint(0,data_size-i)
        sampled_mfcc_data[i] = complement_mfcc_data[arr_choice]
        sampled_chroma_data[i] = complement_chroma_data[arr_choice]

        complement_mfcc_data = numpy.delete(complement_mfcc_data, arr_choice, 0)
        complement_chroma_data = numpy.delete(complement_chroma_data, arr_choice, 0)

        sampled_labels[i] = complement_labels[arr_choice]
        complement_labels = numpy.delete(complement_labels, arr_choice, 0)
    return (sampled_mfcc_data, sampled_chroma_data, sampled_labels, complement_mfcc_data, complement_chroma_data, complement_labels)




#Scans through output of predictions from a classifer to find the p most likely positive samples and n most likely negative samples
def find_p_and_n_indices(p, n, predictions):
    p_index_arr = [] #Indexes of p most likely positive samples
    n_index_arr = [] #Indexes of n most likely negative samples
    p_prob_arr = [] #Probabilities associated with current found indices
    n_prob_arr = []
    for i in range(p):
        p_prob_arr.append(float(0))
        p_index_arr.append(-1)
    for i in range(n):
        n_prob_arr.append(float(0))
        n_index_arr.append(-1)
    count = 0
    for i in range(len(predictions)):
        p_pred = predictions[i][0]
        n_pred = predictions[i][1]
        p_min_prob_index = p_prob_arr.index(min(p_prob_arr))
        n_min_prob_index = n_prob_arr.index(min(n_prob_arr))
        if(p_pred > p_prob_arr[p_min_prob_index]):
            p_prob_arr[p_min_prob_index] = p_pred
            p_index_arr[p_min_prob_index] = i
        if(n_pred > n_prob_arr[n_min_prob_index]):
            n_prob_arr[n_min_prob_index] = n_pred
            n_index_arr[n_min_prob_index] = i
        count = count + 1
    return (p_index_arr, n_index_arr)


#Works just like find_p_and_n_indices, but with extended functionality for non-binary classifiers
def find_probable_indices(class_addition_arr, predictions):

    #Normalize Predictions(If they aren't already)
    count = 0
    for raw in predictions:
        norm = [float(i)/sum(raw) for i in raw]
        predictions[count] = norm
        count = count + 1


    arr_of_arr_indices = []
    arr_of_arr_probs = []
    num_classes = len(class_addition_arr)
    for i in range(num_classes):
        arr_of_arr_indices.append([]) #Each subarray will hold the most probable indexes of the class
        arr_of_arr_probs.append([]) #These will hold the corresponding probabilities at each index
    for i in range(num_classes):
        for j in range(class_addition_arr[i]):
            arr_of_arr_indices[i].append(-1)
            arr_of_arr_probs[i].append(float(0))
    temp_preds = [] #Holds predictions for the current element
    min_prob_index_arr = [] #Holds the indexes of each subarray which contains minimum probability
    for i in range(num_classes):
        temp_preds.append([])
        min_prob_index_arr.append([])
    for i in range(len(predictions)): #For each prediction
        for j in range(num_classes):
            temp_preds[j] = predictions[i][j] #Holds the current predicted probabilities of each class
        for j in range(num_classes):
            min_prob_index_arr[j] = arr_of_arr_probs[j].index(min(arr_of_arr_probs[j]))
        for j in range(num_classes):
            if(temp_preds[j] > arr_of_arr_probs[j][min_prob_index_arr[j]]): #If the current prediction is more likely than the least likely stored prediction
                arr_of_arr_probs[j][min_prob_index_arr[j]] = temp_preds[j] #Update the stored prediction to be the current prediction
                arr_of_arr_indices[j][min_prob_index_arr[j]] = i
    return arr_of_arr_indices








#Index arr:
#[[a,b,c], [d,e,f], [h,i]] -> Index a,b,c become labeled as 0; d,e,f become labeled as 1; h,i become labeled as 2
def add_indices_to_dict(index_arr, from_dict, to_dict):
    from_dict = copy.deepcopy(from_dict)
    to_dict = copy.deepcopy(to_dict)
    temp_e_mfcc_arr = []
    temp_e_chroma_arr = []
    temp_e_label_arr = []
    for i in range(len(index_arr)):
        for index in index_arr[i]:
            e_mfcc = from_dict['MFCC'][index]
            e_mfcc_new_shape = [1]
            for j in e_mfcc.shape:
                e_mfcc_new_shape.append(j)

            e_mfcc_new_shape = tuple(e_mfcc_new_shape)
            e_mfcc.reshape(e_mfcc_new_shape)
            temp_e_mfcc_arr.append(e_mfcc)

            e_chroma = from_dict['Chroma'][index]
            e_chroma_new_shape = [1]
            for k in e_chroma.shape:
                e_chroma_new_shape.append(k)

            e_chroma_new_shape = tuple(e_chroma_new_shape)
            e_chroma.reshape(e_chroma_new_shape)
            temp_e_chroma_arr.append(e_chroma)

            e_label = keras.utils.to_categorical(i, len(index_arr))
            temp_e_label_arr.append(e_label)

    assert(len(temp_e_mfcc_arr) == len(temp_e_chroma_arr) == len(temp_e_label_arr))
    for i in range(len(temp_e_mfcc_arr)):
        e_mfcc = temp_e_mfcc_arr[i]
        e_chroma = temp_e_chroma_arr[i]
        e_label = temp_e_label_arr[i]
        to_dict['MFCC'] = numpy.append(to_dict['MFCC'], numpy.expand_dims(e_mfcc, axis = 0), axis = 0)
        to_dict['Chroma'] = numpy.append(to_dict['Chroma'], numpy.expand_dims(e_chroma, axis = 0), axis = 0)
        to_dict['Labels'] = numpy.append(to_dict['Labels'], numpy.expand_dims(e_label, axis = 0), axis = 0)


    #Just in case there are duplicates in the index_arr (There shouldn't be)
    #Removes duplicates in index_arr and returns it in reverse sorted order
    #to be deleted from from_dict
    found_nums = []
    for i in index_arr:
        for j in i:
            if(j not in found_nums):
                found_nums.append(j)
    found_nums.sort(reverse=True)
    index_arr = found_nums



    for index in index_arr:
        from_dict['MFCC'] = numpy.delete(from_dict['MFCC'], index, 0)
        from_dict['Chroma'] = numpy.delete(from_dict['Chroma'], index, 0)
        from_dict['Labels'] = numpy.delete(from_dict['Labels'], index, 0)

    return (from_dict, to_dict)


#Functionality for repleneshing the small unlabeled data pool from the large unlabeled data pool after each
#round of cotraining cotr
def replenish_dict(from_dict, to_dict, num_samples):
    if(from_dict['Labels'].shape[0] < num_samples): #In case from_dict does not have enough samples
        num_samples = from_dict['Labels'].shape
    indices = []
    while len(indices) < num_samples:
        rand_num = numpy.random.randint(0, from_dict['Labels'].shape[0])
        if(rand_num not in indices):
            indices.append(rand_num)
    temp_e_mfcc_arr = []
    temp_e_chroma_arr = []
    temp_e_label_arr = []
    for index in indices:
            e_mfcc = from_dict['MFCC'][index]
            temp_e_mfcc_arr.append(e_mfcc)
            e_chroma = from_dict['Chroma'][index]
            temp_e_chroma_arr.append(e_chroma)
            e_label = from_dict['Labels'][index]
            temp_e_label_arr.append(e_label)


    assert(len(temp_e_mfcc_arr) == len(temp_e_chroma_arr) == len(temp_e_label_arr))
    for i in range(len(temp_e_mfcc_arr)):
        e_mfcc = temp_e_mfcc_arr[i]
        e_chroma = temp_e_chroma_arr[i]
        e_label = temp_e_label_arr[i]
        to_dict['MFCC'] = numpy.append(to_dict['MFCC'], numpy.expand_dims(e_mfcc, axis = 0), axis = 0)
        to_dict['Chroma'] = numpy.append(to_dict['Chroma'], numpy.expand_dims(e_chroma, axis = 0), axis = 0)
        to_dict['Labels'] = numpy.append(to_dict['Labels'], numpy.expand_dims(e_label, axis = 0), axis = 0)

    indices.sort(reverse=True)

    for index in indices:
        from_dict['MFCC'] = numpy.delete(from_dict['MFCC'], index, 0)
        from_dict['Chroma'] = numpy.delete(from_dict['Chroma'], index, 0)
        from_dict['Labels'] = numpy.delete(from_dict['Labels'], index, 0)

    return (from_dict, to_dict)


#index_arr of form [[a,b c],[d, e, f]] where a,b,c are predicted to belong to class 0 and d, e, f are predicted to belong to class 1
#data is pool of data from which indices were found
def check_for_misclassifications(index_arr, data):
    cur_misclass_arr = []
    for index in range(len(index_arr)):
        cur_misclass_arr.append(0)
    for i in range(len(index_arr)):
        misclass_count = 0
        for j in index_arr[i]:
            label = -1
            prob = -1
            for x in range(len(data['Labels'][j])):
                if(data['Labels'][j][x] > prob):
                    prob = data['Labels'][j][x]
                    label = x
            #If misclassified
            if i != label:
                misclass_count = misclass_count + 1
                cur_misclass_arr[label] = cur_misclass_arr[label] + 1
    return cur_misclass_arr


#Checks for any disagreements between the two classifier's labeling selections and returns them
def check_for_disagreements(h1_indexes, h2_indexes, data):
    disagreement_arr = []
    for h1_i in range(len(h1_indexes)):
        for h2_i in range(len(h2_indexes)):
            for h1_j in h1_indexes[h1_i]:
                for h2_j in h2_indexes[h2_i]:
                    if(h1_j == h2_j and h1_i != h2_i):
                        label = -1
                        prob = -1
                        for it in range(len(data['Labels'][h1_j])):
                            if(data['Labels'][h1_j][it] > prob):
                                label = it
                                prob = data['Labels'][h1_j][it]
                        elmnt = {'TrueClass': label, 'H1Prediction': h1_i ,'H2Prediction': h2_i}
                        disagreement_arr.append(elmnt)
    return disagreement_arr






#h1 -> classifier for feature 1
#h2 -> classifier for feature 2
#U -> Unlabeled Sample Data Set (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: In U, Label array is only included for record-keeping, and is not used in training
#U_Prime -> Unlabeled Sample Data Pool (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: U_Prime is constructed initialized by sampling u elements from U and then replenished by sampling 2p+2n samples from U per iteration
#L -> Labeled Training data (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>})
#Note: L is initially small, then on each iteration both h1 and h2 choose p most likely positive and n most likely negative samples to add from
#U_prime into L
#k is the number of iterations co-training will run for
#test_data -> (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>}) used for testing accuracy
#validation_data -> (Dictionary like {'MFCC': <MFCC training array>, 'Chroma': <Chroma training array>, 'Labels': <Label array>}) used for training validation
def cotraining(h1, h2, U, L, class_addition_arr, k, u, test_data, Validation_Dict):
    #Save original settings of classifiers
    h1_copy = copy.deepcopy(h1)
    h2_copy = copy.deepcopy(h2)
    misclassification_dict = {'MFCC': [], 'Chroma': []}
    disagreement_arr = []
    cmfcc_data, cchroma_data, c_labels, smfcc_data, schroma_data, s_labels, = uniform_random_sampling_for_cotraining(U['MFCC'], U['Chroma'], U['Labels'], u)
    U = {'MFCC': smfcc_data, 'Chroma': schroma_data, 'Labels': s_labels}
    U_Prime = {'MFCC': cmfcc_data, 'Chroma': cchroma_data, 'Labels': c_labels}
    accuracy_arr = []
    h1_accuracy_arr = []
    h2_accuracy_arr = []
    #For each round of cotraining
    for iteration in range(k):
        #Initialize both classifiers
        h1 = copy.deepcopy(h1_copy)
        h2 = copy.deepcopy(h2_copy)
        index_arr = []

        #Train MFCC classifier
        print("TRAINING H1")
        h1.fit(L['MFCC'], L['Labels'],
                batch_size=100,
                epochs=10,
                verbose=1,
                validation_data=(Validation_Dict['MFCC'],Validation_Dict['Labels']))

        #Get MFCC classifier's predictions on unlabeled data set
        h1_predictions = h1.predict(U_Prime['MFCC'])
        #Log MFCC classifier's accuracy on test set
        h1_score = h1.evaluate(test_data['MFCC'], test_data['Labels'], verbose=0)
        h1_accuracy_arr.append(h1_score[1])
        #Find the samples that the MFCC classifier wishes to label
        h1_added_index_arr = find_probable_indices(class_addition_arr, h1_predictions)
        #Check if any of those labelings are incorrect and log them
        h1_misclassifications = check_for_misclassifications(h1_added_index_arr, U_Prime)
        misclassification_dict['MFCC'].append(h1_misclassifications)

        #Train Chroma classifier
        print("TRAINING H2")
        h2.fit(L['Chroma'], L['Labels'],
                batch_size=100,
                epochs=10,
                verbose=1,
                validation_data=(Validation_Dict['Chroma'], Validation_Dict['Labels']))

        #Get Chroma classifier's predictions on unlabeled data set
        h2_predictions = h2.predict(U_Prime['Chroma'])
        #Log Chroma classifier's accuracy on test set
        h2_score = h2.evaluate(test_data['Chroma'], test_data['Labels'], verbose=0)
        h2_accuracy_arr.append(h2_score[1])
        #Find the samples that the Chroma classifier wishes to label
        h2_added_index_arr = find_probable_indices(class_addition_arr, h2_predictions)
        #Check if any of those labels are incorrect and log them
        h2_misclassifications = check_for_misclassifications(h2_added_index_arr, U_Prime)
        misclassification_dict['Chroma'].append(h2_misclassifications)


        #Check to see if MFCC and Chroma classifier disagree on any labelings and log it
        disagreement_arr.append(check_for_disagreements(h1_added_index_arr, h2_added_index_arr, U_Prime))
        #Construct an array of all the indices to be labeled this round of co-training
        index_arr = []
        for i in range(len(h1_added_index_arr)):
            index_arr.append(h1_added_index_arr[i] + h2_added_index_arr[i])

        #index_arr looks like [[p1,p2,p3,...], [n1,n2,n3,...]]
        #Calculate how many samples to replenish the unlabeled data pool with
        replenish_num = 0
        for num in class_addition_arr:
            replenish_num = replenish_num + num
        replenish_num = replenish_num * 2

        #Label samples from index_arr accordingly and add them to the labeled set
        U_Prime, L = add_indices_to_dict(index_arr, U_Prime, L)
        #Replenish the small unlabeled pool from the large unlabeled pool
        U, U_Prime = replenish_dict(U, U_Prime, replenish_num)

    #Log the accuracy of each classifier for this round of cotraining
    accuracy_arr.append(h1_accuracy_arr)
    accuracy_arr.append(h2_accuracy_arr)

    #Return the accuracies, misclassification data, and disagreement data
    return accuracy_arr, misclassification_dict, disagreement_arr
