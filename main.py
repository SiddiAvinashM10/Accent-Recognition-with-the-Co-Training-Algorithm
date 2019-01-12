import os
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import copy

from FeatureExtraction import convert_folder_to_windows, create_mfcc_array, create_chroma_array
from x_val import  sample_for_k_folds, create_array_and_labels, uniform_random_sampling, train_model1, train_model2, train_model3
from cotrain import create_feature_dictionary, cotraining, create_array_and_labels_for_cotraining, uniform_random_sampling_for_cotraining
from graphs import create_MFCC_figure, create_chroma_figure, bar_graph, line_graph, create_misclassification_graph, create_disagreement_table




#Path to the directory this file is in
pwd = os.path.dirname(os.path.realpath(__file__))

#Path to folder where figures will be saved
figure_path = os.path.join(pwd,'Figures')



print("Converting MP3 Files to Wav Windows...")
#Convert MP3 Files into .5 Second Wav chunks
convert_folder_to_windows(os.path.join(pwd, 'Audio_Data', 'Mp3_Data', 'North_America'),os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'), 500)
convert_folder_to_windows(os.path.join(pwd, 'Audio_Data', 'Mp3_Data','India'),os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'), 500)
convert_folder_to_windows(os.path.join(pwd, 'Audio_Data', 'Mp3_Data', 'China'),os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'), 500)
print("Done Converting MP3 Files to Wav Windows")


print("Generating Example Figures for MFCC and Chroma...")
#Save MFCC and Chroma Figures
create_MFCC_figure(os.path.join(pwd, 'Audio_Data','Wav_Data','India', 'bengali 1_3.wav'), os.path.join(figure_path,'mfcc_chroma_examples'))
create_chroma_figure(os.path.join(pwd, 'Audio_Data','Wav_Data','India'), os.path.join(figure_path,'mfcc_chroma_examples'))
print("Done Generating Example Figures for MFCC and Chroma")


##############################Begin K-FOLDS#######################################################

print("Converting Wav's to MFCC arrays...")
#Convert Wav Files to MFCC's
na_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
ind_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
china_mfcc_arr = create_mfcc_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))
print("Done Converting Wav's to MFCC arrays")

print("Converting Wav's to Chroma arrays...")
#Convert Wav Files to Chroma's
na_chroma_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'North_America'))
ind_chroma_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'India'))
china_chroma_arr = create_chroma_array(os.path.join(pwd, 'Audio_Data', 'Wav_Data', 'China'))
print("Done Converting Wav's to Chroma arrays")


print("Combining arrays into data pool and adding labels...")
#Combine the arrays and create labels
mfcc_data_arr, mfcc_label_arr = create_array_and_labels(na_mfcc_arr, ind_mfcc_arr, china_mfcc_arr)
chroma_data_arr, chroma_label_arr = create_array_and_labels(na_chroma_arr, ind_chroma_arr, china_chroma_arr)
print("Done combining arrays into data pool and adding labels")


print("Sampling for k-folds...")
#Sample out 5 Folds for Cross Validation
mfcc_d = sample_for_k_folds(mfcc_data_arr, mfcc_label_arr, 5)
chroma_d = sample_for_k_folds(chroma_data_arr, chroma_label_arr, 5)
print("Done sampling for k-folds")


print("Performing k-folds cross validation on MFCC data...")
#Perform k-folds x-val on MFCC data
#Model 1 is CNN with Dropout
#Model 2 is CNN without Droput
#Model 3 is CNN with Droput and increased Kernel Size
#---------------------------------------------------------------------------------------#
print("???????????????????????????????????????????????????????????????????????")
print("data shape: ",mfcc_data_arr.shape)
print("label shape: ",mfcc_label_arr.shape)
print("???????????????????????????????????????????????????????????????????????")
mfcc_accuracy_array1 = []
mfcc_accuracy_array2 = []
mfcc_accuracy_array3 = []
for i in range(5):
    sampled_test_data=mfcc_d["folds_sampled_data_{}".format(i)]
    sampled_test_labels=mfcc_d["folds_sampled_labels_{}".format(i)]
    k=(i+1)%5
    sampled_train_data = mfcc_d["folds_sampled_data_{}".format(k)]
    print("Initial shape = ",sampled_train_data.shape)
    #sampled_train_data = sampled_train_data.reshape(3165, 22, 22, 1)
    sampled_train_labels = mfcc_d["folds_sampled_labels_{}".format(k)]
    print("Initial shape = ",sampled_train_labels.shape)
    print(sampled_test_data.shape)
    print(sampled_test_labels.shape)
    for j in range(5):
        if j==i or j==k:
            continue
        else:
            sampled_train_data = numpy.append(sampled_train_data,mfcc_d["folds_sampled_data_{}".format(j)],axis=0)
            sampled_train_labels = numpy.append(sampled_train_labels,mfcc_d["folds_sampled_labels_{}".format(j)],axis=0)
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
    mfcc_accuracy_array1.append(score1)
    mfcc_accuracy_array2.append(score2)
    mfcc_accuracy_array3.append(score3)
#---------------------------------------------------------------------------------------#
print("Done performing k-folds cross validation on MFCC data")


print("Generating bar graph of MFCC k-folds results...")
#Save Graph of MFCC K-Folds Accuracies
bar_graph(mfcc_accuracy_array1, mfcc_accuracy_array2, mfcc_accuracy_array3, "MFCC K-Fold", os.path.join(figure_path, 'k_folds_results'))
print("Done generating bar graph of MFCC k-folds results")


print("Performing k-folds cross validation on Chroma data...")
#Perform k-folds x-val on Chroma data
#Model 1 is CNN with Dropout
#Model 2 is CNN without Droput
#Model 3 is CNN with Droput and increased Kernel Size
#---------------------------------------------------------------------------------------#
print("???????????????????????????????????????????????????????????????????????")
print("data shape: ",chroma_data_arr.shape)
print("label shape: ",chroma_label_arr.shape)
print("???????????????????????????????????????????????????????????????????????")
chroma_accuracy_array1 = []
chroma_accuracy_array2 = []
chroma_accuracy_array3 = []
for i in range(5):
    sampled_test_data=chroma_d["folds_sampled_data_{}".format(i)]
    sampled_test_labels=chroma_d["folds_sampled_labels_{}".format(i)]
    k=(i+1)%5
    sampled_train_data = chroma_d["folds_sampled_data_{}".format(k)]
    print("Initial shape = ",sampled_train_data.shape)
    #sampled_train_data = sampled_train_data.reshape(3165, 22, 22, 1)
    sampled_train_labels = chroma_d["folds_sampled_labels_{}".format(k)]
    print("Initial shape = ",sampled_train_labels.shape)
    print(sampled_test_data.shape)
    print(sampled_test_labels.shape)
    for j in range(5):
        if j==i or j==k:
            continue
        else:
            sampled_train_data = numpy.append(sampled_train_data,chroma_d["folds_sampled_data_{}".format(j)],axis=0)
            sampled_train_labels = numpy.append(sampled_train_labels,chroma_d["folds_sampled_labels_{}".format(j)],axis=0)
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
    chroma_accuracy_array1.append(score1)
    chroma_accuracy_array2.append(score2)
    chroma_accuracy_array3.append(score3)
#---------------------------------------------------------------------------------------#
print("Done performing k-folds cross validation on Chroma data")


print("Generating bar graph of Chroma k-folds results...")
#Save Graph of Chroma K-Folds Accuracies
bar_graph(chroma_accuracy_array1, chroma_accuracy_array2, chroma_accuracy_array3, "Chroma K-Fold", os.path.join(figure_path, 'k_folds_results'))
print("Done generating bar graph of Chroma k-folds results")

##############################END K-FOLDS#######################################################


##############################BEGIN CO-TRAINING#######################################################

#Paths to Folders Containing Wav Chunks for Each Class
na_path = os.path.join(pwd, "Audio_Data", "Wav_Data", "North_America")
ind_path = os.path.join(pwd, "Audio_Data", "Wav_Data", "India")
china_path = os.path.join(pwd, "Audio_Data", "Wav_Data", "China")


print("Creating dictionaries of MFCC and Chroma features for 3 class co-training...")
#Create Dictionaries Containing MFCC and Chroma Features for each Class
na_feature_dict = create_feature_dictionary(na_path)
ind_feature_dict = create_feature_dictionary(ind_path)
china_feature_dict = create_feature_dictionary(china_path)
print("Done creating dictionaries of MFCC and Chroma features for 3 class co-training")


print("Combining dictionaries and adding labels for 3 class co-training...")
mfcc_data_array, chroma_data_array, label_array = create_array_and_labels_for_cotraining(na_feature_dict, ind_feature_dict, china_feature_dict)
print("Done combining dictionaries and adding labels for 3 class co-training...")


#Set Up Parameters 3 class for Cotraining
L_size = 1000   #Labeled Data Will Originally Start with 1000 samples
Test_size = 500 #Test Data will contain 500 samples
Validation_size = 500 #Validation Data will contain 500 samples
u = 1000 #Unlabeled data pool will contain 1000 elements
k = 100 #Co-Training will run for 100 iterations
class_addition_arr = [10,10,10] #Each classifier will be able to label 10 samples per iteration


print("Sampling data for 3 class co-training...")
#Sample Data for Co-Training
L_mfcc_data, L_chroma_data, L_labels, U_mfcc_data, U_chroma_data, U_labels  = uniform_random_sampling_for_cotraining(mfcc_data_array, chroma_data_array, label_array, L_size)
Test_mfcc_data, Test_chroma_data, Test_labels, U_mfcc_data, U_chroma_data, U_labels  = uniform_random_sampling_for_cotraining(U_mfcc_data, U_chroma_data, U_labels, Test_size)
Validation_mfcc_data, Validation_chroma_data, Validation_labels, U_mfcc_data, U_chroma_data, U_labels = uniform_random_sampling_for_cotraining(U_mfcc_data, U_chroma_data, U_labels, Validation_size)
print("Done sampling data for 3 class co-training")

print("Combining sampled data into final dictionaries for 3 class co-training...")
#Build Dictionaries of Features to be used in Co-Training
U_Dict = {'MFCC': U_mfcc_data, 'Chroma': U_chroma_data, 'Labels': U_labels}
L_Dict = {'MFCC': L_mfcc_data, 'Chroma': L_chroma_data, 'Labels': L_labels} #L_Size number of samples
Test_Dict = {'MFCC': Test_mfcc_data, 'Chroma': Test_chroma_data, 'Labels': Test_labels} #Test_size number of samples
Validation_Dict = {'MFCC': Validation_mfcc_data, 'Chroma': Validation_chroma_data, 'Labels': Validation_labels} #Validation_size number of samples
print("Done combining sampled data into final dictionaries for 3 class co-training")


print("Preparing MFCC (H1) classifier for 3 class co-training...")
#-----------------------PREPARE MFCC CLASSIFIER FOR COTRAINING-----------------------------------#
sampled_train_data = U_Dict['MFCC']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['MFCC']

input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(input_shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
num_classes = Test_Dict['Labels'].shape[1]

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

h1 = copy.deepcopy(model)########################
#------------------------------------------------------------------------------------------------#
print("Done preparing MFCC (H1) classifier for 3 class co-training")



print("Preparing Chroma (H2) classifier for 3 class co-training...")
#-----------------------PREPARE CHROMA CLASSIFIER FOR COTRAINING---------------------------------#
sampled_train_data = U_Dict['Chroma']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['Chroma']

input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(input_shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
num_classes = Test_Dict['Labels'].shape[1]

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

h2 = copy.deepcopy(model)#####################
#------------------------------------------------------------------------------------------------#
print("Done preparing Chroma (H2) classifier for 3 class co-training")









#--------------------------------3 Class Co-Training---------------------------#
print("Performing 3 Class Co-Training...")
#Perform Co-Training for 3 classes
cotraining_accuracy_arr, misclassification_dict, disagreement_arr = cotraining(h1, h2, U_Dict, L_Dict, class_addition_arr, k, u, Test_Dict, Validation_Dict)
print("Done performing 3 class Co-Training")

#Previous 3 class results for training classifiers on all data samples
class3_mfcc_total_accuracy = 0.6520000004768371
class3_chroma_total_accuracy = 0.5060000002384186


print("Creating graph of 3 class co-training accuracies...")
#Save Graph of 3 Class Co-Training Accuracy Results
line_graph(cotraining_accuracy_arr,os.path.join(figure_path,'line_graphs', 'three_class'), class3_mfcc_total_accuracy, class3_chroma_total_accuracy)
print("Done creating graph of 3 class co-training accuracies")

print("Creating graph of 3 class co-training misclassifications...")
#Save Graph of Number of Misclassifications for 3 Classes
create_misclassification_graph("Three Class Misclassifications by Classifier", "three_class",os.path.join(figure_path,"misclassification_labels"), misclassification_dict)
print("Done creating graph of 3 class co-training misclassifications")

print("Creating disagreement table for 3 class co-training...")
#Save Table of Labelling Disagreements for 3 classes
create_disagreement_table("Three Class Disagreement Matrix", "three_class_disagree",os.path.join(figure_path,"disagreement_matrices"), disagreement_arr, 3)
print("Done creating disagreement table for 3 class co-training")
#------------------------------------------------------------------------------#



print("Creating dictionaries of MFCC and Chroma features for 2 class co-training...")
#Create Dictionaries Containing MFCC and Chroma Features for each Class
na_feature_dict = create_feature_dictionary(na_path)
ind_feature_dict = create_feature_dictionary(ind_path)
print("Done creating dictionaries of MFCC and Chroma features for 2 class co-training")





print("Combining dictionaries and adding labels for 2 class co-training...")
#Prepare Data for 2-Class Cotraining
mfcc_data_array, chroma_data_array, label_array = create_array_and_labels_for_cotraining(na_feature_dict, ind_feature_dict)
print("Done combining dictionaries and adding labels for 2 class co-training")


#Set Up Parameters 2 class for Cotraining
L_size = 1000   #Labeled Data Will Originally Start with 1000 samples
Test_size = 500 #Test Data will contain 500 samples
Validation_size = 500 #Validation Data will contain 500 samples
u = 1000 #Unlabeled data pool will contain 1000 elements
k = 100 #Co-Training will run for 100 iterations
class_addition_arr = [10,10] #Each classifier will be able to label 10 samples per iteration


print("Sampling data for 2 class co-training...")
#Sample Data for Co-Training
L_mfcc_data, L_chroma_data, L_labels, U_mfcc_data, U_chroma_data, U_labels  = uniform_random_sampling_for_cotraining(mfcc_data_array, chroma_data_array, label_array, L_size)
Test_mfcc_data, Test_chroma_data, Test_labels, U_mfcc_data, U_chroma_data, U_labels  = uniform_random_sampling_for_cotraining(U_mfcc_data, U_chroma_data, U_labels, Test_size)
Validation_mfcc_data, Validation_chroma_data, Validation_labels, U_mfcc_data, U_chroma_data, U_labels = uniform_random_sampling_for_cotraining(U_mfcc_data, U_chroma_data, U_labels, Validation_size)
print("Done sampling data for 2 class co-training")


print("Combining sampled data into final dictionaries for 2 class co-training...")
#Build Dictionaries of Features to be used in Co-Training
U_Dict = {'MFCC': U_mfcc_data, 'Chroma': U_chroma_data, 'Labels': U_labels}
L_Dict = {'MFCC': L_mfcc_data, 'Chroma': L_chroma_data, 'Labels': L_labels} #L_Size number of samples
Test_Dict = {'MFCC': Test_mfcc_data, 'Chroma': Test_chroma_data, 'Labels': Test_labels} #Test_size number of samples
Validation_Dict = {'MFCC': Validation_mfcc_data, 'Chroma': Validation_chroma_data, 'Labels': Validation_labels} #Validation_size number of samples
print("Done combining sampled data into final dictionaries for 2 class co-training")






print("Preparing MFCC (H1) classifier for 2 class co-training...")
#-----------------------PREPARE MFCC CLASSIFIER FOR COTRAINING-----------------------------------#
sampled_train_data = U_Dict['MFCC']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['MFCC']

input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(input_shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
num_classes = Test_Dict['Labels'].shape[1]

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

h1 = copy.deepcopy(model)########################
#------------------------------------------------------------------------------------------------#
print("Done preparing MFCC (H1) classifier for 2 class co-training")



print("Preparing Chroma (H2) classifier for 2 class co-training...")
#-----------------------PREPARE CHROMA CLASSIFIER FOR COTRAINING---------------------------------#
sampled_train_data = U_Dict['Chroma']
sampled_train_labels = U_Dict['Labels']
test_data = Test_Dict['Chroma']

input_shape = (sampled_train_data.shape[1],sampled_train_data.shape[2],sampled_train_data.shape[3])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(input_shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
num_classes = Test_Dict['Labels'].shape[1]

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

h2 = copy.deepcopy(model)#####################
#------------------------------------------------------------------------------------------------#
print("Done preparing Chroma (H2) classifier for 2 class co-training")





#--------------------------------2 Class Co-Training---------------------------#
print("Performing 2 Class Co-Training...")
#Perfom 2-Class CoTraininng
cotraining_accuracy_arr, misclassification_dict, disagreement_arr = cotraining(h1, h2, U_Dict, L_Dict, class_addition_arr, k, u, Test_Dict, Validation_Dict)
print("Done performing 2 Class Co-Training...")

#Previous 2 class results for training classifiers on all data samples
class2_mfcc_total_accuracy = 0.8539999995231629
class2_chroma_total_accuracy = 0.616

print("Creating graph of 2 class co-training accuracies...")
#Save Graph of 2 Class Co-Training Accuracy Results
line_graph(cotraining_accuracy_arr,os.path.join(figure_path,'line_graphs', 'two_class'), class2_mfcc_total_accuracy, class2_chroma_total_accuracy)
print("Done creating graph of 2 class co-training accuracies")

print("Creating graph of 2 class co-training misclassifications...")
#Save Graph of Number of Misclassifications for 2 Classes
create_misclassification_graph("Two Class Misclassifications by Classifier", "two_class",os.path.join(figure_path,"misclassification_labels"), misclassification_dict)
print("Done creating graph of 2 class co-training misclassifications")


print("Creating disagreement table for 2 class co-training...")
#Save Table of Labelling Disagreements for 3 classes
create_disagreement_table("Two Class Disagreement Matrix", "two_class_disagree",os.path.join(figure_path,"disagreement_matrices"), disagreement_arr, 2)
print("Done creating disagreement table for 2 class co-training")
#------------------------------------------------------------------------------#

##############################END CO-TRAINING#######################################################
