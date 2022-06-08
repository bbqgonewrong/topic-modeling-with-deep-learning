

######### topic model updated with 1 neuron each for estimating the weight of topics for k=4 ###########
from tensorflow.keras import regularizers
from json import JSONEncoder
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,AveragePooling2D,Dropout,Flatten,Dense,Input,Concatenate
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.metrics import multilabel_confusion_matrix,classification_report
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LambdaCallback
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.utils import shuffle
import json
import os, re
import seaborn as sn

model_name = 'topic_weight_k4_full_ds_updated_no_cw_3'
json_filename = model_name+'.json'
#json_batch = 'test.json'
#Strictly responsible for retraining the data .......
#retrain = True
#Data directory stuff ###############################################
cache_dir = '/data/s4133366/data_cache'
train_dir = '/data/s4133366/data/train'
save_dir = '/data/s4133366/saved_models'
data_dir = '/data/s4133366/data'
model_dir = os.path.join(save_dir,model_name)
plot_dir = os.path.join(model_dir,'plots')

####################################################################


def topic_col_to_df(frame):
  pd.DataFrame(frame)
  frame = frame.map(lambda x:str(x).replace('[[',''))
  frame = frame.map(lambda x:str(x).replace(']]',''))
  frame = frame.map(lambda x:str(x).replace('[',''))
  frame = frame.map(lambda x:str(x).replace(']',''))
  frame = frame.map(lambda x:str(x).replace(']]]',''))
  frame = frame.map(lambda x:str(x).replace('[[[',''))

  frame = frame.str.split(',',expand=True)

  frame.columns = ['Topic 1','Topic 2','Topic 3','Topic 4']
  return frame

def make_topic_df(filename):
  data = [json.loads(line) for line in open(filename, 'r')]
  df_data = pd.DataFrame(data)
  df_data['epoch'] = df_data['epoch']+1 
  topic_column = df_data['topic_layer_weights']
  topic_df = topic_col_to_df(topic_column)
  testing = pd.merge(df_data,topic_df,how='inner',on=df_data.index)
  final_data = testing[['epoch','categorical_accuracy','loss','Topic 1','Topic 2','Topic 3','Topic 4']]
  return final_data

#Plitting the weights
def plot_topics(df):
  plots_dir = os.path.join(plot_dir,'weights_plot') 
  if(not os.path.isdir(plots_dir)):
      print(plots_dir,' does not exist yet.. creating')
      os.mkdir(plots_dir)
      print(plots_dir,'Created')
  topic_list = ['Topic 1','Topic 2','Topic 3','Topic 4']
  for topic in topic_list:
    weight = []
    #Topic Weights
    y = df[topic].tolist()
    x = df['epoch'].tolist()
    for el in y:
      
      weight.append(float(el))
    ax = sn.barplot(x=x, y=weight, data=df)
    plt.xlabel("Epoch")
    plt.ylabel("Topic Weight Value")
    plt.title("Weight Distribution for "+topic)
    plt.savefig(os.path.join(plots_dir,'weights_'+topic+'.png')) #Place in plot dir 
    plt.close()
    print(topic+' Weights plotted...')
    #Accuracy
    
    plt.plot(df['categorical_accuracy'])
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy with respect to '+topic)
    plt.savefig(os.path.join(plots_dir,'acc_'+topic+'.png'))#Place in plot dir 
    plt.close()
    print(topic+' Accuracy plotted...')
    #Loss
    plt.plot(df['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss with respect to '+topic)
    plt.savefig(os.path.join(plots_dir,'loss_'+topic+'.png'))#Place in plot dir  
    plt.close()
    print(topic+' Loss plotted...')


final_data = make_topic_df(json_filename)
final_data.head(1)
print('Plotting the weights')
plot_topics(final_data)
print('Topic Weights have been plotted ....')






