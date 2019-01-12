import librosa
import librosa.display
import os.path
import matplotlib.pyplot as plt

def convert_WAVtoMFCC(filename,destination_folder):
    #y,sr=librosa.load(filename+".wav")
    y, sr=librosa.load(filename)
    print("*******************************************************************")
    mfccs=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=100)
    print(mfccs)
    print(type(mfccs))
    print(mfccs.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    #plt.colorbar()
    #plt.title('MFCC')
    plt.tight_layout()
    plt.axis('off')
    #graph_name="{}{}".format(filename,".png")
    fn = os.path.basename(filename) #Spencer
    fn = fn.split('.')[0]    #Spencer
    fn = fn + ".png"
    graph_name=os.path.join(destination_folder,fn)
    #print("GRAPH NAME: " + graph_name)
    #print("fn: " + fn)
    #print("DEST: " + destination_folder)
    #print(os.path.join(destination_folder,graph_name))
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    plt.savefig(os.path.join(destination_folder,graph_name), bbox_inches='tight')
    #plt.show()

def create_mfcc(filename):
    y, sr=librosa.load(filename)
    mfcc=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=22) #Was 100
    #print(mfcc)



    return mfcc
#convert_WAVtoMFCC("test1","X:\\CS 688\\Project\\Accent Data\\India\\Hindi\\MFCC plot")
#filename,destination folder



'''
###################################################################################
    count = 0
    for raw in mfcc:
        norm = [float(i)/sum(raw) for i in raw]
        mfcc[count] = norm
        count = count + 1

###################################################################################
'''
