from data import *
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
model = tf.keras.models.load_model(r'C:\Users\Kong Be\Desktop\unet_segmentation\my_model_6.hdf5')#load model
testGene = testGenerator(r"C:\Users\Kong Be\Desktop\unet_segmentation\Predict")#prepare data

results = model.predict_generator(testGene,10,verbose=1)#predict
saveResult(r"C:\Users\Kong Be\Desktop\unet_segmentation\Predict",results)#save results to path
saveText(r"C:\Users\Kong Be\Desktop\unet_segmentation\Predict"
         ,results) # save text for each image