

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import re
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from glob import glob

from absl.flags import FLAGS
from absl import app, flags


######################
flags.DEFINE_string('ori_dir', default='D:\Label\data', help = 'path of pet folder')
flags.DEFINE_string('breed', default = 'dog', help = 'dog or cat')

def main(_argv):
  IMG_SIZE = 224

  path = glob(FLAGS.ori_dir+ '/*/*.png')
  images = []
  labels = []

  class_list = set()
  for image_file in path:
      folder_name = image_file.split('\\')[-2]
      class_list.add(folder_name)
      labels.append(folder_name)

      image = Image.open(image_file)
      image = image.resize((IMG_SIZE, IMG_SIZE))
      image = np.array(image)
      images.append(image)

  class_list = list(class_list)
  class_list.sort()

  with open('class_list.txt', 'w') as f:
    for i in class_list:
      f.write(i)
      f.write('\n')


  class2idx = {cls:idx for idx, cls in enumerate(class_list)}
  for i in range(len(labels)):
      labels[i] = class2idx[labels[i]]

#   x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=123)
#   x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.25, random_state=123)

  X = np.array([i for i in images])
  Y = np.array([i for i in labels])

  np.savez_compressed('dataset.npz', X = X, Y = Y)

#   with open('data/' + FLAGS.breed + '/num.txt', 'w') as f:
#     f.write(str(len(X)))
#     f.write('\n')
#     f.write(str(len(x_val)))
#     f.write('\n')
#     f.write(str(len(class_list)))


if __name__ == '__main__':
  app.run(main)



####################






