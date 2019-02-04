import cv2
import glob
import os
import random
import math
from sklearn.metrics import confusion_matrix
import numpy as np
import dlib
import itertools
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sys

path = sys.argv[1]
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
#Using Adaptive Historam Equalization technique as a preprocessing measure to improve background contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#shape predictor for obtaining set of point locations of facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Detector for detecting human face from an image
detector = dlib.get_frontal_face_detector()
#Getting the one vs one libsvm implementation
linear_svm = SVC(kernel='linear', probability=True, tol=1e-3)
#AdaBoost classifier
ada_boost = AdaBoostClassifier()
#Decision tree classifier
decision_tree = DecisionTreeClassifier()
#Creating dictionary for all values
landmark_data = {}

def get_image_files(emotion):
    image_files = glob.glob(path+"\\%s\\*" %emotion)
    random.shuffle(image_files)
    training = image_files[:int(len(image_files)*0.8)]
    test = image_files[-int(len(image_files)*0.2):]
    return training, test

def get_face_landmarks(image):
    detected_face = detector(image, 1)
    #Obtaining landmarks for an image
    for k,d in enumerate(detected_face):
        #Draw Facial Landmarks with the predictor class
        shape = predictor(image, d)
        x_positions = []
        y_positions = []
        #Store X and Y coordinates in two lists
        for i in range(1,68):
            x_positions.append(float(shape.part(i).x))
            y_positions.append(float(shape.part(i).y))
            
        x_mean = np.mean(x_positions)
        y_mean = np.mean(y_positions)
        x_centre = [(x - x_mean) for x in x_positions]
        y_centre = [(y - y_mean) for y in y_positions]

        landmarks = []
        for x, y, w, z in zip(x_centre, y_centre, x_positions, y_positions):
            landmarks.append(w)
            landmarks.append(z)
            np_mean = np.asarray((y_mean, x_mean))
            np_coordinate = np.asarray((z,w))
            np_distance = np.linalg.norm(np_coordinate - np_mean)
            landmarks.append(np_distance)
            landmarks.append((math.atan2(y, x)*360)/(2*math.pi))

        landmark_data['landmarks'] = landmarks
    if len(detected_face) < 1: 
        landmark_data['landmarks'] = "erroneous_image"

def create_training_test_sets():
    training_data = []
    training_classes = []
    test_data = []
    test_classes = []
    for emotion in emotions:
        print("Creating training and test dataset for emotion: %s" %emotion)
        training, test = get_image_files(emotion)
        #Append data to training and test list, and generate labels 0-7
        for image_object in training:
            #Reading the image
            image = cv2.imread(image_object)
            #Converting to grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_gray_image = clahe.apply(gray)
            get_face_landmarks(clahe_gray_image)
            if landmark_data['landmarks'] == "erroneous_image":
                print("Error detecting face in the image")
            else:
                #append image array to training data list
                training_data.append(landmark_data['landmarks'])
                training_classes.append(emotions.index(emotion))
    
        for image_object in test:
            image = cv2.imread(image_object)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_gray_image = clahe.apply(gray)
            get_face_landmarks(clahe_gray_image)
            if landmark_data['landmarks'] == "erroroneous_image":
                print("Error detecting face in the image")
            else:
                test_data.append(landmark_data['landmarks'])
                test_classes.append(emotions.index(emotion))

    return training_data, training_classes, test_data, test_classes   

accuracy_linear_svm = []
print("Training Linear SVM")
for i in range(0, 10):
    #Creating sets by randomly sampling 80:20
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    #Obtaining the numpy array for the training and test dataset for the classifier
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training linear SVM for set %s" %(i+1)) #train SVM
    linear_svm.fit(training_data_array, training_classes)
    prediction_accuracy = linear_svm.score(test_data_array, test_classes)
    print("Accuracy for set %s:" %(i+1), prediction_accuracy*100)
    #Store accuracy in a list
    accuracy_linear_svm.append(prediction_accuracy)
    prediction_labels = linear_svm.predict(test_data)
    #print the confusion matrix
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    img_name = 'confusion_matrix/svm_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)
    #print(result)
#Obtain mean accuracy of the 10 iterations
print("Mean value of accuracies using linear SVM: %s" %(np.mean(accuracy_linear_svm) * 100))
svm_model_name = os.path.join(os.getcwd() + "/models/",  "linear_svm.model")
joblib.dump(linear_svm, svm_model_name, compress = 3)

accuracy_decision_tree = []
print("Training Decision Tree")
for i in range(0, 10):
    #Creating sets by randomly sampling 80:20
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    #Obtaining the numpy array for the training and test dataset for the classifier
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training Decision Tree for set %s" %(i + 1))
     #Train Decision Tree
    decision_tree.fit(training_data_array, training_classes)
    prediction_accuracy = decision_tree.score(test_data_array, test_classes)
    print("Accuracy for set %s:" %(i + 1), prediction_accuracy*100)
    #Store accuracy in a list
    accuracy_decision_tree.append(prediction_accuracy)
    prediction_labels = decision_tree.predict(test_data)
    #print the confusion matrix
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    img_name = 'confusion_matrix/dt_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)
#Obtain mean accuracy of the 10 iterations
print("Mean value of accuracies using Decision Tree: %s" %(np.mean(accuracy_decision_tree) * 100))
dt_model_name = os.path.join(os.getcwd() + "/models/",  "decision_tree.model")
joblib.dump(decision_tree, dt_model_name, compress = 3)

accuracy_ada_boost = []
print("Training AdaBoost Classifier")
for i in range(0, 10):
    #Creating sets by randomly sampling 80:20
    print("Creating set %s" %(i + 1)) 
    training_data, training_classes, test_data, test_classes = create_training_test_sets()
    #Obtaining the numpy array for the training and test dataset for the classifier
    training_data_array = np.array(training_data) 
    test_data_array = np.array(test_data)
    print("Training AdaBoost for set %s" %(i + 1))
     #Train AdaBoost Classifier
    ada_boost.fit(training_data_array, training_classes)
    prediction_accuracy = ada_boost.score(test_data_array, test_classes)
    print("Accuracy for set %s:" %(i + 1), prediction_accuracy*100)
    #Store accuracy in a list
    accuracy_ada_boost.append(prediction_accuracy)
    prediction_labels = ada_boost.predict(test_data)
    #print the confusion matrix
    result = confusion_matrix(test_classes, prediction_labels)
    df_cm = pd.DataFrame(result, index = [i for i in emotions], columns = [i for i in emotions])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    img_name = 'confusion_matrix/adb_confusion_matrix%s.png' %(i + 1)
    plt.savefig(img_name)
#Obtain mean accuracy of the 10 iterations
print("Mean value of accuracies using AdaBoost: %s" %(np.mean(accuracy_ada_boost) * 100))
adb_model_name = os.path.join(os.getcwd() + "/models/",  "ada_boost.model")
joblib.dump(ada_boost, adb_model_name, compress = 3)