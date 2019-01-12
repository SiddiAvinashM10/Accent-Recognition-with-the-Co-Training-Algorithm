# cs688
Pattern Recognition Final Project

Spencer Riggins (sriggins2@gmu.edu)
Siddi Avinash Chenmilla (schenmil@gmu.edu)

################################################################################
How to run our project:

Since there is no direct download link for our data set, we zipped up the folder
containing our data and placed it in a Google Drive for you to download. The
link to the Google Drive is:

https://drive.google.com/drive/folders/1Zm2zT5nBsE1vfjeZWz03CyTfTJkEIlOB

To execute our code you must first unzip the Audio_Data in the same directory that you have
the Python files in. This gives the Python scripts access to the data.

Once this is done you will be able to run our code. Our original charts and
figures will originally be stored in the 'Figures' subdirectory.

Running the 'main.py' script will run through the entire logic of our project,
from data processing, feature extraction, k-folds cross validation, and then the
co-training algorithm. This will replace all of the charts and figures in the
'Figures' subdirectory with the charts and figures for that particular run of the
program.

If you wish to go back to the original figures, or wish to see the representation of
the data used to generate the charts, then uncommenting the large block of commented
code at the bottom of 'graphs.py' and then running 'graphs.py' will use the saved
data to regenerate the orignal charts and graphs.

An overview of the use of each file:

'nn.py' was used to train the MFCC and Chroma classifiers on all the data to get
the theoretical optimum performance of each classifiers
'graphs.py' contains methods for gnenerating the graphs and figures of our data
'x_val.py' contains methods for cross validation of our MFCC and Chroma classifiers
'FeatureExtraction.py' contains methods for data preprocessing and feature extraction
'Cotrain.py' contains methods for performing co-training
'main.py' contains the overall logic of the project

Note: This code is written in Python 3, and the 'main.py' file takes a number of
hours to execute fully (between 6-10).
################################################################################


Division of labor:

Spencer:

FeatureExtraction.py
Cotrain.py
main.py
x_val.py (Data Preparation and Sampling Methods)
graphs.py (2 of 6 graphing methods)
nn.py (1/2)
Presentation (1/2)
Paper (1/2)


Sid:

x_val.py (train_model methods, classifier parameter tuning)
graphs.py (4 of 6 graphing methods)
nn.py (1/2)
convert_WAVtoMFCC.py
Presentation (1/2)
Paper (1/2)
