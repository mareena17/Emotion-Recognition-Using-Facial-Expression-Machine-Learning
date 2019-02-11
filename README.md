# Emotion-Recognition-Using-Facial-Expression--Machine-Learning

Project setup:
Python version: 3.6
Install following libraries:
1) cv2
command: pip install opencv
2) numpy
command: pip install numpy
3) dlib
command: pip install dlib
4) sklearn
command: pip install sklearn
5) seaborn
command: pip install seaborn
6) pandas
command: pip install pandasRecognition
7) matplotlib
command: pip install matplotlib

Project folder structure:
source
	|_er_dataset
	|_dataset
	|_models
	|_confusion_matrix
	|_data_labeler.py
	|_train_model.py
	|_predict.py
	|_ReadMe.txt
	|_shape_predictor_68_face_landmarks.dat

er_dataset=This folder is CK+ dataset
dataset=This folder contains sorted set of emotions namely [anger, contempt, disgust, fear, happiness, neutral, sadness, surprise], obtained after executing data_labeler.py
models=This folder contains list of trained models
confusion_matrix=This folder contains plot of confusion matrices obtained for various classifiers
data_labeler.py=This program is used to sort the CK+ dataset into emotion specific folders
train_model.py=This program is used to train different models on the dataset and save the models along with the confusion matrices
predict.py=This program predicts the class of a static image or predicts the emotion class for a video stream

Commands to execute via Anaconda Promt:
1. Sort the CK+ dataset:
cd source
python data_labeler.py

2. Train the models:
cd source
python train_model.py <arg>
<arg> = path of dataset
Example execution:
source>python train_model.py dataset

3. Predict classes of test data:
1) Static image
cd source
python predict.py <arg1> <arg2> <arg3>
<arg1> = path of the trained model saved during training phase, values = models/linear_svm.model or models/decision_tree.model or models/ada_boost.model
<arg2> = T
<arg3> = path of image whose class has to be predicted
Example execution: 
source>python predict.py models/linear_svm.model T dataset/anger/S010_004_00000019.png

2) Real time
cd source
python predict.py <arg1> <arg2> <arg3>
<arg1> = path of the trained model saved during training phase, values = models/linear_svm.model or models/decision_tree.model or models/ada_boost.model
<arg2> = F
<arg3> = NA
Example execution: 
source>python predict.py models/linear_svm.model F NA
