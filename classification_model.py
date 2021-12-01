import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import time

from absl import flags, app
from absl.flags import FLAGS


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from skimage.io import imread
import cv2
import matplotlib.pyplot as plt
import threading
from threading import Thread
import warnings
import traceback
import contextlib
import sys
from PIL import Image, ImageChops, ImageTk
from sklearn.preprocessing import Binarizer
import tkinter as tk
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import sys
from tensorflow.keras import models
import shutil



save_path='C:/Users/365mc/Desktop/jsh_code/classification/save_image/'
def callback():

    
    class Target:
        watchDir = "C:/Users/365mc/Desktop/jsh_code/classification/image"
    #watchDir에 감시하려는 디렉토리를 명시

        def __init__(self):
            self.observer = Observer()   #observer객체를 만듦

        def run(self):
            event_handler = Handler()
            self.observer.schedule(event_handler, self.watchDir, recursive=True)
            self.observer.start()
            try:
                while True:
                    time.sleep(1)
            except:
                self.observer.stop()
                self.observer.join()
                
    class Handler(FileSystemEventHandler):
        def on_created(self, event):  # 파일, 디렉터리가 생성되면 실행
            filepath = event.src_path
            filepath = os.path.dirname(filepath)
            
            filename = event.src_path
            filename = os.path.basename(filename)
            
            a = os.path.join(filepath, filename)
            
            # 이미지데이터 불러오기
            image = Image.open(a)
            image = image.resize((224, 224))
            image = np.array(image)
            image = np.reshape(image, (1, 224, 224, 3))


            model = models.load_model('my_model') # 분류모델 load
            prediction = model.predict(image)
            pred_class = np.argmax(prediction, axis=1)
             
            a= pred_class[0]
            #classification
            shutil.move(filepath+filename ,  save_path +'/{}/'.format(a) + filename)
            # if custom_predict == '1':
            #     shutil.move(filepath+filename ,save_path +'/1/' +'_1'+filename)
            
            # elif custom_predict == '2':
            #     shutil.move(filepath+filename ,save_path +'/2/' +'_2'+filename)
            
            # elif custom_predict == '3':
            #     shutil.move(filepath+filename ,save_path +'/3/' +'_3'+filename)
            
            # elif custom_predict == '4':
            #     shutil.move(filepath+filename ,save_path +'/4/' +'_4'+filename)
            
            # elif custom_predict == '5':
            #     shutil.move(filepath+filename ,save_path +'/5/' +'_5'+filename)

            # elif custom_predict == '6':
            #     shutil.move(filepath+filename ,save_path +'/6/' +'_6'+filename)
            
            # else :
            #     shutil.move(filepath+filename ,save_path +'/7/' +'_7'+filename)
            





                        
                        
if __name__ == "__main__" :#본 파일에서 실행될 때만 실행되도록 함
    print('Sites folder watchdog is running...')
    w = Target()
    w.run()

