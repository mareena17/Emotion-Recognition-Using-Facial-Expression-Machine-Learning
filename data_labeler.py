import os
#library for matching pathnames
import glob
#copying image files into respective folders
from shutil import copyfile
facial_emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
subjects = glob.glob(os.getcwd() + "/er_dataset/source_emotions//*")

for subject in subjects:
    s_no = "%s" %subject[-4:]
    for sessions in glob.glob("%s//*" %subject):
        for files in glob.glob("%s//*" % sessions):
            current_session = sessions[-3:]
            file = open(files, 'r')

            emotion = int(float(file.readline()))

            sourcefile_emotion = glob.glob(os.getcwd() + "/er_dataset/source_images//%s//%s//*" % (s_no, current_session))[
                -1]
            sourcefile_neutral = glob.glob(os.getcwd() + "/er_dataset/source_images//%s//%s//*" % (s_no, current_session))[
                0]
            dest_neutral = os.getcwd() + "/dataset//neutral//%s" % sourcefile_neutral[-21:]
            dest_emotions = os.getcwd() + "/dataset//%s//%s" % (facial_emotions[emotion], sourcefile_emotion[-21:])

            copyfile(sourcefile_neutral, dest_neutral)
            copyfile(sourcefile_emotion, dest_emotions)
