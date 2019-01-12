import matplotlib.pyplot as plt
import numpy as np
import os
from pydub import AudioSegment
import librosa
from librosa.feature.spectral import chroma_stft
import librosa.display


#Builds graph of k-folds accuracy from arrays of k-folds accuracies
def bar_graph(accuracy_array_m1, accuracy_array_m2, accuracy_array_m3, feature_name, destination_folder):

    index = np.arange(5)
    bar_width = 0.25
    opacity = 0.8

    plt.bar(index, accuracy_array_m1 , alpha=opacity, color='b',width=0.25, label='Model 1')
    plt.bar(index + bar_width, accuracy_array_m2 , alpha=opacity, color='r',width=0.25, label='Model 2')
    plt.bar(index + bar_width + bar_width, accuracy_array_m3 , alpha=opacity, color='g',width=0.25, label='Model 3')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('{} Accuracy Graph'.format(feature_name))
    plt.xticks(index + bar_width, ('fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5'))
    plt.legend()

    plt.tight_layout()
    graph_name = "{}_Accuracy_Graph.png".format(feature_name)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    #plt.show()
    plt.close()

#Builds graph of cotrain accuracies from array of cotraining accuracies and optimal accuracies of classifiers
def line_graph(accuracy_array, destination_folder, total_accuracy_mfcc, total_accuracy_chroma):
    t = np.arange(0, 100, 1)
    s1 = accuracy_array[0]
    s2 = accuracy_array[1]
    s3 = []
    s4 = []
    for i in s1:
        s3.append(total_accuracy_mfcc)
        s4.append(total_accuracy_chroma)
    mfcc_line, = plt.plot(t, s1, label = 'MFCC ACCURACY')
    chroma_line, = plt.plot(t, s2, label = 'CHROMA ACCURACY')
    total_mfcc_line, = plt.plot(t, s3, label = 'ACTUAL MFCC ACCURACY', linestyle = ':')
    total_chroma_line, = plt.plot(t, s4, label = 'ACTUAL CHROMA ACCURACY', linestyle = ':')
    first_legend = plt.legend(handles=[mfcc_line, chroma_line, total_mfcc_line, total_chroma_line], loc=1)
    ax = plt.gca().add_artist(first_legend)
    plt.xlabel('Co-training Iterations')
    plt.ylabel('Accuracy')

    graph_name = "Co-training_graph.png"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    plt.close()

def listdir_ignore_hidden(path):
    files = []
    for file in os.listdir(path):
        if not file.startswith('.') and not os.path.isdir(os.path.join(path,file)):
            files.append(file)
    return files

#Creates sample chroma figure
def create_chroma_figure(wav_folder, destination_folder):
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
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_arr[0], x_axis='time')
    plt.title('Chroma')
    plt.tight_layout()
    graph_name = "Chroma_graph.png"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    plt.close()

#Creates sample mfcc figure
def create_MFCC_figure(filename,destination_folder):
    y, sr=librosa.load(filename)
    mfccs=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=100)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCC')
    plt.tight_layout()
    plt.axis('off')
    fn = os.path.basename(filename) #Spencer
    fn = "MFCC_graph.png"
    graph_name=os.path.join(destination_folder,fn)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    plt.close()


#Creates bar graph of misclassifications during cotraining
def create_misclassification_graph(graph_name, file_name, folder_path, data):
    index = np.arange(2)
    bar_width = 0.25
    opacity = 0.8

    file_name = os.path.join(folder_path, file_name)
    num_classes = len(data['MFCC'][0])
    mfcc_misclass_totals = []
    chroma_misclass_totals = []
    for nc in range(num_classes):
        mfcc_misclass_totals.append(0)
        chroma_misclass_totals.append(0)
    for i in range(len(data['MFCC'])):
        for j in range(len(data['MFCC'][i])):
            mfcc_misclass_totals[j] = mfcc_misclass_totals[j] + data['MFCC'][i][j]
            chroma_misclass_totals[j] = chroma_misclass_totals[j] + data['Chroma'][i][j]

    class_misclass_arr = []
    for nc in range(num_classes):
        class_misclass_arr.append([0,0])
    for nc in range(num_classes):
        class_misclass_arr[nc][0] = mfcc_misclass_totals[nc]
        class_misclass_arr[nc][1] = chroma_misclass_totals[nc]


    bar_alignment = index
    color_arr = ['r','g','b']
    class_label_arr = ["North America", "India", "China"]
    for inds in range(len(class_misclass_arr)):
        bar_alignment = index + (inds*(bar_width))
        class_lab = "Class " + str(inds + 1)
        plt.bar(bar_alignment, class_misclass_arr[inds] , alpha=opacity, color=color_arr[inds],width=0.25, label=class_label_arr[inds])

    plt.xlabel('Classifier')
    plt.ylabel('Misclassifications')
    plt.title(graph_name)
    plt.xticks(index + bar_width, ('MFCC Classifier', 'Chroma Classifier'))
    plt.legend()

    plt.tight_layout()
    graph_name = file_name
    destination_folder = folder_path
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    plt.close()

#Creates disagreement matrix from disagreement cotraining data
def create_disagreement_table(graph_name, file_name, folder_path, data, num_classes):
    collabel = ["MFCC", "Chroma", "Neither"]
    rowlabel_full = ['    N_America', '      India', '     China']
    rowlabel = rowlabel_full[0:num_classes]
    fig, ax = plt.subplots()
    if(num_classes == 2):
        props = dict(boxstyle='square', facecolor='white', alpha=0.0)
        ax.text(0.17, 0.67, "Label Disagreement Matrix", transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        props = dict(boxstyle='square', facecolor='white', alpha=0.0)
        ax.text(0.25, 0.61, "CORRECT CLASSIFIER", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    if(num_classes == 3):
        props = dict(boxstyle='square', facecolor='white', alpha=0.0)
        ax.text(0.17, 0.68, "Label Disagreement Matrix", transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        props = dict(boxstyle='square', facecolor='white', alpha=0.0)
        ax.text(0.25, 0.625, "CORRECT CLASSIFIER", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.ylabel("N/A")
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    tabel_data = np.zeros((num_classes, 3))
    for i in range(len(data)):
        for j in range(len(data[i])):
            true_class = data[i][j]['TrueClass']
            h1_prediction = data[i][j]['H1Prediction']
            h2_prediction = data[i][j]['H2Prediction']
            correct_classifier = -1
            if(h1_prediction == true_class):
                correct_classifier = 0
            elif(h2_prediction == true_class):
                correct_classifier = 1
            else:
                correct_classifier = 2
            tabel_data[true_class][correct_classifier] = int(tabel_data[true_class][correct_classifier] + 1)


    ax.table(cellText=tabel_data,colLabels=collabel, rowLabels = rowlabel,loc='center')
    fig.tight_layout()

    graph_name = file_name
    destination_folder = folder_path
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name))
    plt.savefig(os.path.join(destination_folder,graph_name))
    plt.close()


'''
#Example results
#--------------#

#2 Class cotraining accuracies [[<H1 iteration 1 accuracy>, <H1 iteration 2 accuracy> ...], [<H2 iteration 1 accuracy>, <H2 iteration 2 accuracy> ...]]
class2_100iterations = [[0.658, 0.6440000009536743, 0.538, 0.6500000004768371, 0.6279999995231629, 0.6559999990463257, 0.6360000009536744, 0.5159999995231629, 0.5159999995231629, 0.6839999995231628, 0.69, 0.666, 0.6740000004768372, 0.64, 0.6099999995231629, 0.684, 0.6299999990463256, 0.6860000009536743, 0.5159999995231629, 0.5159999995231629, 0.5159999995231629, 0.6680000009536743, 0.5159999995231629, 0.5159999995231629, 0.6600000009536743, 0.6620000004768372, 0.6859999990463257, 0.6360000004768371, 0.6760000009536743, 0.658, 0.656, 0.6320000004768371, 0.6479999995231629, 0.666, 0.6679999995231628, 0.6860000004768372, 0.6840000004768372, 0.6559999990463257, 0.6780000009536743, 0.6500000009536743, 0.688, 0.6680000009536743, 0.6859999995231628, 0.6700000009536743, 0.666, 0.686, 0.6699999990463257, 0.6499999995231629, 0.6599999995231628, 0.6640000004768372, 0.6679999990463257, 0.6619999995231628, 0.6979999990463257, 0.686, 0.6380000009536743, 0.6779999995231628, 0.6780000004768372, 0.6579999995231628, 0.6699999990463257, 0.6520000004768371, 0.6780000009536743, 0.6799999995231628, 0.6539999990463257, 0.6400000009536743, 0.6839999990463257, 0.6719999990463257, 0.6619999990463257, 0.7040000004768372, 0.6559999995231628, 0.6739999995231628, 0.6539999990463257, 0.6640000009536743, 0.6379999990463256, 0.6739999995231628, 0.5159999995231629, 0.6699999995231628, 0.6500000004768371, 0.6460000009536743, 0.6580000004768372, 0.67, 0.6860000009536743, 0.6580000009536743, 0.6479999990463257, 0.6619999995231628, 0.6779999995231628, 0.6580000009536743, 0.6440000009536743, 0.666, 0.6220000004768371, 0.6040000009536743, 0.6540000009536743, 0.6479999990463257, 0.6660000009536743, 0.6619999995231628, 0.6439999990463257, 0.6439999990463257, 0.6520000004768371, 0.6400000009536743, 0.6500000004768371, 0.6639999995231628], [0.488, 0.5040000009536744, 0.48800000047683717, 0.5140000004768371, 0.5060000004768371, 0.5240000004768371, 0.5020000004768371, 0.5220000004768371, 0.5080000004768371, 0.5060000004768371, 0.5239999995231629, 0.5120000004768371, 0.5180000009536743, 0.5180000004768371, 0.5399999990463257, 0.5200000004768371, 0.5180000004768371, 0.5159999990463257, 0.516, 0.4999999990463257, 0.5199999990463257, 0.5140000004768371, 0.5140000004768371, 0.5279999995231628, 0.5339999995231628, 0.5420000009536743, 0.5219999990463257, 0.53, 0.5179999995231629, 0.5359999990463257, 0.5219999990463257, 0.5259999995231628, 0.5260000004768371, 0.526, 0.5180000004768371, 0.538, 0.534, 0.5060000009536744, 0.5120000004768371, 0.5219999990463257, 0.5300000004768372, 0.5220000004768371, 0.5080000004768371, 0.5300000002384185, 0.522, 0.5080000004768371, 0.5239999995231629, 0.5220000004768371, 0.5159999995231629, 0.5160000009536743, 0.5220000004768371, 0.514, 0.542, 0.5139999990463257, 0.53, 0.5280000009536743, 0.5299999997615814, 0.5259999995231628, 0.5320000004768372, 0.5360000009536743, 0.534, 0.5319999995231628, 0.5300000009536743, 0.5259999990463257, 0.5560000004768372, 0.5300000009536743, 0.5659999995231628, 0.5340000009536743, 0.5380000004768372, 0.5440000009536743, 0.5279999995231628, 0.5280000004768372, 0.536, 0.54, 0.55, 0.5379999995231628, 0.5220000004768371, 0.5280000009536743, 0.5360000004768372, 0.5320000004768372, 0.5400000009536743, 0.5519999995231628, 0.5540000004768372, 0.5479999995231628, 0.5340000004768372, 0.5480000009536743, 0.5380000009536743, 0.5280000009536743, 0.5480000004768372, 0.5520000004768372, 0.5380000004768372, 0.5460000004768372, 0.5380000009536743, 0.5340000004768372, 0.5379999990463257, 0.5360000004768372, 0.5500000004768372, 0.528, 0.5720000009536743, 0.5479999990463257]]

#2 class accuracies for training on all data (MFCC and Chroma classifiers)
class2_mfcc_total_accuracy = 0.8539999995231629
class2_chroma_total_accuracy = 0.616


#3 Class cotraining accuracies [[<H1 iteration 1 accuracy>, <H1 iteration 2 accuracy> ...], [<H2 iteration 1 accuracy>, <H2 iteration 2 accuracy> ...]]
class3_100iterations = [[0.48399999952316286, 0.5039999995231629, 0.4859999997615814, 0.5020000002384186, 0.49199999952316287, 0.5380000004768372, 0.528, 0.45600000047683714, 0.4699999997615814, 0.52, 0.5299999995231628, 0.5219999995231629, 0.5200000002384185, 0.512, 0.5060000004768371, 0.5039999995231629, 0.51, 0.49199999952316287, 0.5240000002384185, 0.48200000047683716, 0.4839999997615814, 0.4579999997615814, 0.43999999976158144, 0.512, 0.43999999976158144, 0.516, 0.4860000002384186, 0.522, 0.49000000023841855, 0.5279999995231628, 0.5479999995231628, 0.482, 0.5, 0.5260000004768371, 0.4759999997615814, 0.5340000004768372, 0.498, 0.5000000002384186, 0.48999999976158143, 0.52, 0.5139999997615814, 0.5099999997615814, 0.5320000002384185, 0.4680000002384186, 0.49999999976158144, 0.48800000023841855, 0.4840000002384186, 0.5039999995231629, 0.46799999952316285, 0.494, 0.49399999952316287, 0.49199999952316287, 0.49600000023841856, 0.5060000004768371, 0.4759999997615814, 0.4800000002384186, 0.49600000023841856, 0.47000000047683715, 0.48399999952316286, 0.5259999997615814, 0.47200000047683716, 0.4800000002384186, 0.4840000002384186, 0.492, 0.5120000002384186, 0.5000000002384186, 0.48999999976158143, 0.5040000002384186, 0.5240000004768371, 0.4700000002384186, 0.49999999976158144, 0.5219999997615814, 0.4819999997615814, 0.47399999952316285, 0.4779999997615814, 0.4820000002384186, 0.48399999952316286, 0.5040000002384186, 0.4840000002384186, 0.49800000023841856, 0.506, 0.48800000047683717, 0.4760000002384186, 0.4679999997615814, 0.4620000002384186, 0.474, 0.45999999952316284, 0.4979999995231628, 0.48000000047683716, 0.4839999997615814, 0.49000000023841855, 0.5059999997615814, 0.4820000001192093, 0.468, 0.4720000002384186, 0.4739999997615814, 0.4799999997615814, 0.4780000002384186, 0.4980000001192093, 0.486], [0.412, 0.40400000047683715, 0.4319999995231628, 0.40799999952316285, 0.406, 0.3800000004768372, 0.4120000002384186, 0.41, 0.37000000023841856, 0.37400000023841856, 0.36199999952316286, 0.37999999976158144, 0.354, 0.388, 0.3660000004768372, 0.3940000002384186, 0.37599999976158144, 0.4, 0.3700000004768372, 0.38200000047683713, 0.382, 0.3500000002384186, 0.3560000002384186, 0.3600000002384186, 0.368, 0.36600000023841855, 0.38400000023841857, 0.38199999976158144, 0.3999999997615814, 0.354, 0.3720000004768372, 0.35400000047683716, 0.3479999997615814, 0.36400000047683717, 0.3520000002384186, 0.356, 0.38799999952316283, 0.4060000002384186, 0.38199999976158144, 0.37599999976158144, 0.3579999997615814, 0.37399999976158144, 0.36, 0.37399999976158144, 0.35999999952316286, 0.37199999976158143, 0.3399999997615814, 0.37600000023841856, 0.3660000004768372, 0.3579999997615814, 0.3660000004768372, 0.3700000004768372, 0.3660000004768372, 0.36599999976158143, 0.364, 0.37999999976158144, 0.37999999976158144, 0.36799999976158143, 0.36999999976158143, 0.37799999976158144, 0.376, 0.36799999976158143, 0.362, 0.36400000047683717, 0.3780000004768372, 0.36200000047683717, 0.3639999997615814, 0.3740000004768372, 0.3680000004768372, 0.36200000047683717, 0.3720000001192093, 0.3720000004768372, 0.36799999976158143, 0.3760000004768372, 0.38400000023841857, 0.37399999976158144, 0.36799999976158143, 0.368, 0.3740000004768372, 0.4000000001192093, 0.36400000047683717, 0.38800000047683714, 0.36599999976158143, 0.38600000047683714, 0.35000000047683716, 0.3739999995231628, 0.38200000047683713, 0.392, 0.3740000004768372, 0.37, 0.372, 0.39000000047683714, 0.3720000004768372, 0.378, 0.35800000047683717, 0.386, 0.368, 0.38400000047683713, 0.3660000004768372, 0.3660000004768372]]

#3 class accuracies for training on all data (MFCC and Chroma classifiers)
class3_mfcc_total_accuracy = 0.6520000004768371
class3_chroma_total_accuracy = 0.5060000002384186

#MFCC K-folds accuracies for classifier 1, 2, and 3
accuracy_array_m1 =  [0.6523404254304602, 0.6453191489868976, 0.6487234041031371, 0.6561702127152301, 0.644042553140762]
accuracy_array_m2 =  [0.2680851064337061, 0.2757446809271549, 0.41319148931097477, 0.326595744687192, 0.31319148933633845]
accuracy_array_m3 =  [0.6363829786726769, 0.6282978724926076, 0.6325531916415438, 0.6142553191996635, 0.6144680850049282]

#Chroma K-folds accuracies for classifier 1, 2, and 3
accuracy_array_c1 = [0.47063829792306777, 0.4636170213019594, 0.47191489361702127, 0.46085106382978724, 0.46361702112441366]
accuracy_array_c2 = [0.44212765959983175, 0.44212765962519546, 0.42148936165140033, 0.4510638298379614, 0.4382978722896982]
accuracy_array_c3 = [0.48382978713258784, 0.4589361701874023, 0.4676595744934488, 0.46510638305481444, 0.4714893616514003]

#Dictionary containing data about 2 class misclassifications
#Dictionary elements correspond to which <Classifier is responsible for misclassification>: [[<True Class Label>, <Assigned Class Label>]]
two_class_misclassification_dict = {'MFCC': [[0, 3], [0, 1], [0, 1], [0, 0], [2, 2], [1, 3], [2, 1], [4, 3], [3, 4], [0, 1], [1, 1], [1, 1], [2, 3], [1, 3], [0, 3], [3, 1], [0, 0], [3, 0], [1, 1], [7, 3], [3, 2], [0, 2], [5, 5], [3, 4], [1, 2], [3, 3], [0, 3], [5, 1], [0, 1], [2, 4], [1, 4], [1, 2], [3, 1], [1, 2], [2, 5], [1, 2], [0, 1], [3, 2], [2, 3], [1, 3], [0, 3], [4, 3], [4, 1], [2, 1], [4, 0], [5, 2], [0, 3], [2, 4], [3, 6], [2, 2], [1, 4], [1, 2], [3, 6], [7, 6], [2, 1], [3, 1], [0, 2], [3, 2], [2, 3], [2, 2], [1, 3], [1, 3], [3, 5], [1, 2], [2, 5], [2, 3], [0, 4], [2, 1], [3, 4], [2, 3], [4, 3], [3, 3], [4, 2], [2, 1], [1, 3], [1, 1], [2, 3], [3, 3], [2, 2], [2, 2], [2, 2], [3, 3], [2, 2], [2, 4], [3, 2], [3, 3], [5, 4], [1, 3], [6, 2], [1, 2], [0, 2], [2, 2], [4, 3], [3, 5], [8, 5], [4, 1], [6, 2], [3, 3], [1, 0], [4, 4]], 'Chroma': [[3, 5], [3, 2], [5, 5], [3, 7], [1, 5], [2, 6], [4, 4], [5, 6], [1, 6], [3, 3], [2, 5], [2, 5], [1, 5], [3, 5], [3, 9], [1, 2], [2, 3], [0, 4], [0, 6], [3, 3], [3, 3], [5, 5], [3, 6], [4, 3], [3, 6], [3, 4], [5, 6], [6, 5], [3, 5], [6, 2], [2, 7], [5, 5], [2, 4], [3, 3], [1, 3], [2, 3], [5, 3], [2, 5], [3, 7], [1, 4], [2, 5], [1, 5], [0, 4], [4, 4], [4, 5], [3, 5], [4, 3], [5, 5], [2, 4], [3, 5], [4, 5], [4, 5], [2, 4], [6, 5], [3, 6], [5, 6], [3, 6], [5, 4], [2, 2], [3, 4], [4, 6], [4, 6], [2, 4], [2, 4], [4, 7], [4, 5], [1, 4], [4, 4], [3, 5], [4, 5], [2, 6], [5, 8], [3, 6], [6, 4], [4, 5], [2, 6], [4, 6], [1, 2], [4, 6], [3, 6], [2, 4], [6, 4], [4, 4], [5, 6], [2, 2], [3, 4], [5, 5], [2, 4], [2, 7], [4, 2], [6, 6], [1, 4], [2, 5], [3, 4], [4, 3], [3, 4], [3, 4], [3, 6], [3, 6], [5, 4]]}

#Dictionary containing data about 3 class misclassifications
#Dictionary elements correspond to which <Classifier is responsible for misclassification>: [[<True Class Label>, <Assigned Class Label>]]
three_class_misclassification_dict = {'MFCC': [[0, 2, 3], [1, 4, 7], [2, 2, 9], [3, 4, 2], [1, 2, 5], [4, 2, 7], [6, 3, 4], [4, 5, 8], [1, 4, 7], [4, 2, 6], [3, 4, 4], [5, 4, 5], [3, 3, 3], [2, 5, 7], [7, 1, 8], [3, 6, 7], [2, 2, 5], [3, 3, 7], [0, 2, 7], [3, 1, 3], [7, 3, 9], [2, 6, 7], [3, 4, 10], [2, 4, 6], [1, 4, 9], [6, 7, 7], [3, 2, 7], [6, 2, 4], [6, 2, 5], [2, 0, 11], [2, 5, 4], [4, 2, 7], [6, 4, 7], [6, 4, 11], [3, 4, 3], [5, 2, 6], [5, 2, 4], [3, 5, 7], [3, 2, 8], [8, 2, 7], [2, 5, 4], [7, 3, 7], [5, 5, 3], [4, 4, 5], [1, 4, 10], [8, 2, 7], [4, 3, 5], [2, 7, 8], [5, 3, 7], [3, 3, 4], [4, 3, 8], [2, 2, 10], [3, 8, 6], [3, 4, 7], [3, 4, 11], [5, 5, 8], [2, 7, 10], [1, 4, 10], [6, 3, 9], [2, 5, 11], [3, 5, 4], [4, 2, 10], [1, 3, 10], [3, 7, 6], [7, 3, 5], [6, 3, 7], [1, 6, 5], [1, 4, 9], [5, 2, 5], [5, 3, 8], [4, 2, 8], [3, 8, 5], [3, 7, 3], [6, 3, 6], [6, 3, 7], [2, 4, 8], [3, 8, 5], [7, 5, 8], [5, 5, 5], [3, 2, 11], [3, 4, 8], [6, 2, 6], [2, 7, 8], [3, 5, 5], [0, 3, 10], [3, 4, 7], [5, 4, 6], [3, 6, 6], [4, 7, 6], [4, 3, 8], [4, 5, 7], [5, 2, 4], [5, 5, 6], [1, 4, 4], [4, 2, 10], [9, 5, 3], [3, 2, 10], [6, 7, 4], [1, 7, 5], [3, 6, 7]], 'Chroma': [[3, 7, 8], [6, 7, 6], [5, 10, 3], [6, 5, 10], [4, 11, 9], [1, 2, 8], [5, 4, 9], [5, 5, 7], [4, 6, 7], [4, 13, 5], [3, 6, 7], [7, 8, 7], [6, 7, 10], [5, 10, 5], [2, 3, 7], [8, 6, 7], [6, 4, 8], [2, 5, 6], [4, 8, 6], [4, 8, 9], [3, 4, 12], [3, 10, 7], [2, 6, 11], [6, 3, 8], [5, 6, 11], [0, 8, 7], [5, 3, 11], [5, 5, 7], [6, 5, 11], [4, 4, 7], [9, 5, 7], [3, 8, 10], [9, 7, 5], [5, 4, 11], [6, 5, 6], [8, 0, 12], [3, 6, 9], [2, 2, 9], [3, 9, 6], [3, 10, 5], [8, 5, 6], [11, 5, 9], [4, 7, 8], [6, 7, 7], [4, 3, 6], [5, 5, 8], [3, 6, 11], [2, 8, 7], [10, 8, 3], [1, 5, 8], [2, 9, 11], [8, 4, 4], [5, 7, 4], [5, 8, 3], [0, 8, 8], [4, 9, 5], [2, 9, 9], [3, 6, 5], [2, 5, 8], [2, 8, 6], [4, 7, 5], [7, 7, 7], [6, 4, 5], [1, 5, 8], [5, 7, 6], [4, 5, 7], [5, 6, 10], [5, 4, 11], [2, 8, 10], [5, 5, 8], [6, 5, 7], [4, 9, 6], [2, 10, 7], [7, 8, 8], [3, 8, 9], [3, 8, 6], [2, 8, 6], [4, 7, 6], [5, 11, 6], [4, 7, 7], [5, 6, 10], [8, 3, 7], [4, 6, 6], [2, 6, 4], [3, 3, 13], [5, 9, 4], [6, 5, 7], [4, 6, 9], [5, 5, 5], [6, 4, 11], [1, 7, 11], [5, 4, 8], [4, 7, 9], [1, 11, 8], [7, 2, 9], [1, 8, 4], [3, 6, 7], [6, 7, 7], [3, 2, 11], [4, 9, 9]]}


#Array containing information about 2 class classifier disagreements
#One subarray per iteration, each subdict contains information about one disagreement instance
two_class_disagreement_arr = [[], [], [], [], [], [], [], [], [], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [], [{'TrueClass': 0, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [], [], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [], [], [], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}]]


#Array containing information about 2 class classifier disagreements
#One subarray per iteration, each subdict contains information about one disagreement instance
three_class_disagreement_arr = [[], [{'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 2}], [], [], [], [], [], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 2}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}], [], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 2}], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [], [{'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [], [], [], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 1}], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 1}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 0, 'H2Prediction': 2}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 0, 'H1Prediction': 1, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 2, 'H1Prediction': 1, 'H2Prediction': 0}], [], [], [], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 2}, {'TrueClass': 0, 'H1Prediction': 1, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 1, 'H2Prediction': 0}], [{'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 1}, {'TrueClass': 1, 'H1Prediction': 2, 'H2Prediction': 1}], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 1}], [], [], [], [], [{'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 0}], [], [], [{'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 1}], [], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 0}, {'TrueClass': 2, 'H1Prediction': 2, 'H2Prediction': 1}], [{'TrueClass': 2, 'H1Prediction': 0, 'H2Prediction': 1}, {'TrueClass': 0, 'H1Prediction': 0, 'H2Prediction': 2}], [{'TrueClass': 0, 'H1Prediction': 2, 'H2Prediction': 1}]]




#Example Results Figure Generation
#--------------------------------#
figure_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Figures')
print(figure_path)

##################################################################################################################################################

create_MFCC_figure(os.path.join(os.path.dirname(__file__), 'Audio_Data','Wav_Data','India', 'bengali 1_3.wav'), os.path.join(figure_path,'mfcc_chroma_examples'))
create_chroma_figure(os.path.join(os.path.dirname(__file__), 'Audio_Data','Wav_Data','India'), os.path.join(figure_path,'mfcc_chroma_examples'))


line_graph(class3_100iterations,os.path.join(figure_path,'line_graphs', 'three_class'), class3_mfcc_total_accuracy, class3_chroma_total_accuracy)
line_graph(class2_100iterations,os.path.join(figure_path,'line_graphs', 'two_class'), class2_mfcc_total_accuracy, class2_chroma_total_accuracy)
bar_graph(accuracy_array_m1, accuracy_array_m2, accuracy_array_m3, "MFCC K-Fold", os.path.join(figure_path, 'k_folds_results'))
bar_graph(accuracy_array_c1, accuracy_array_c2, accuracy_array_c3, "Chroma K-Fold", os.path.join(figure_path, 'k_folds_results'))


create_misclassification_graph("Two Class Misclassifications by Classifier", "two_class",os.path.join(figure_path,"misclassification_labels"), two_class_misclassification_dict)
create_misclassification_graph("Three Class Misclassifications by Classifier", "three_class",os.path.join(figure_path,"misclassification_labels"), three_class_misclassification_dict)



create_disagreement_table("Two Class Disagreement Matrix", "two_class_disagree",os.path.join(figure_path,"disagreement_matrices"),two_class_disagreement_arr, 2)
create_disagreement_table("Three Class Disagreement Matrix", "three_class_disagree",os.path.join(figure_path,"disagreement_matrices"),three_class_disagreement_arr, 3)
'''
