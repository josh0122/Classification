import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras import models

from absl import flags, app
from absl.flags import FLAGS

from glob import glob

import cv2
import numpy as np
from PIL import Image
import re



model = models.load_model('my_model')
image = Image.open('1.2.410.200013.1.300.1.20160428130444026.86.png')
image = image.resize((224, 224))
image = np.array(image)



image = np.reshape(image, (1, 224, 224, 3))

prediction = model.predict(image)
pred_class = np.argmax(prediction, axis=1)

# pred_breed = class_list[int(pred_class)]
print(pred_class)




