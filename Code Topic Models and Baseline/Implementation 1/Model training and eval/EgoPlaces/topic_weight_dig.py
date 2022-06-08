#######IMPLEMENTATION 3
#Trying with focal loss
######### topic model updated with 1 neuron each for estimating the weight of topics for k=4 ###########
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

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
#Tensorflow addons
import tensorflow_addons as tfa
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

#HYPER PARAMS #########
#batch_logsfile = 'test.json'
t_batch_size = 151
v_batch_size = 128
num_epochs =40
model_name = 'topic_weight_k4_imp_1_dig_1'
json_filename = model_name+'.json'
test_filename = 'test_'+model_name+'.json'
learning_rate = 0.001
#Strictly responsible for retraining the data .......
#retrain = True
#Data directory stuff ###############################################
cache_dir = '/data/s4133366/data_cache'
train_dir = '/data/s4133366/data/train'
val_dir = '/data/s4133366/data/val'
save_dir = '/data/s4133366/saved_models'
data_dir = '/data/s4133366/data'
####################################################################
train_csv_file = 'train_mod.csv'
val_csv_file = 'val.csv'
####################################################################
cache_t = os.path.join(cache_dir,model_name+'_t')
cache_v = os.path.join(cache_dir,model_name+'_v')
####################################################################
regu_t = os.path.split(cache_t)[-1]+'*'
regu_v = os.path.split(cache_v)[-1]+'*'
print(regu_v,regu_t)
####################################################################
####################################################################
###### ONLY FOR RETRAINING MODELS
retrain = False
prev_model = 'topic_full_ds'
prev_model_dir = os.path.join(save_dir,prev_model)
prev_weights_dir = os.path.join(prev_model_dir,prev_model+'_weights')

####################################################################

model_dir = os.path.join(save_dir,model_name)
prev_model_dir = os.path.join(save_dir,prev_model)

#Directory for plots....
plot_dir = os.path.join(model_dir,'plots')
#Directory for storing the numpy arrays for test and train ...
history_dir = os.path.join(model_dir,'history')

#Responsible for the name of the npy files for both test and train
train_metrics_path = 'history_'+model_name+'.npy'
res_path = 'results_'+model_name+'.npy'
#Responsible for Checkpointing
chk_dir = os.path.join(model_dir,model_name+'_chk')
#Saving weights for retraining ....
weights_dir = os.path.join(model_dir,model_name+'_weights')
####################################################################

#Function for clearing cache
def purge(dir, pattern):
  for f in os.listdir(dir):
    if re.search(pattern, f):
      os.remove(os.path.join(dir, f))
#Function for making and storing the plots

def plot_metrics(history_path,results_path):
  read_train = np.load(history_path,allow_pickle=True).item()
  read_test = np.load(results_path,allow_pickle=True).item()
  history_path = os.path.split(history_path)[-1]
  model_string = 'Topic Model for k=4 for Places365 Dataset'
  history_path = history_path.replace('.npy','')
  if not os.path.isdir(plot_dir):
    print('Plot directory '+plot_dir+' does not exist at the moment. Creating ....')
    os.mkdir(plot_dir)
  print(plot_dir+' created ...')
  if not os.path.isdir(os.path.join(plot_dir,history_path)) :
    os.mkdir(os.path.join(plot_dir,history_path))
    folder_path = os.path.join(plot_dir,history_path)
    print(folder_path+' created...')
  else:
    folder_path = os.path.join(plot_dir,history_path)
    print(folder_path,' exists ...')  
    print('Adding the metric plots to ',folder_path)

  #summarize history for accuracy

  plt.plot(read_train['categorical_accuracy'])
  plt.axhline(read_test['categorical_accuracy'], color='g', linestyle='--')
  plt.title('model categorical accuracy for '+str(model_string))
  plt.ylabel('Categorical Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'accuracy_'+str(history_path)+'.png'))
  plt.clf()

  # summarize history for loss

  plt.plot(read_train['loss'])
  plt.axhline(read_test['loss'], color='g', linestyle='--')
  #plt.plot(read_train['val_loss'])
  plt.title('model loss for '+str(model_string))
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  #plt.show()
  plt.savefig(os.path.join(folder_path,'loss_'+str(history_path)+'.png'))
  plt.clf()

  #summarize history for Precision 
  plt.plot(read_train['precision'])
  #plt.plot(read_train['val_precision'])
  plt.axhline(read_test['precision'], color='g', linestyle='--')
  plt.title('model precision for '+str(model_string))
  plt.ylabel('Precision')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'precision_'+str(history_path)+'.png'))
  plt.clf()

  #summarize history for Recall 

  plt.plot(read_train['recall'])
  #plt.plot(read_train['val_recall'])
  plt.axhline(read_test['recall'], color='g', linestyle='--')
  plt.title('Model Recall for '+str(model_string))
  plt.ylabel('Recall')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'recall_'+str(history_path)+'.png'))
  plt.clf()

  plt.plot(read_train['lr'])
  plt.title('Model Learning rate for '+str(model_string))
  plt.ylabel('Learning Rate')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'lr_'+str(history_path)+'.png'))
  plt.clf()

print('########## WEIGHTED TOPIC MODEL TRAINING FOR FULL DS  ... ######################')
print('Loading CSVs')


train_csv = pd.read_csv(os.path.join(data_dir,train_csv_file))
train_csv = train_csv.sample(frac=1,random_state = 42)


val_csv = pd.read_csv(os.path.join(data_dir,val_csv_file))
#Use this to shuffle val data
#val_csv = val_csv.sample(frac=1,random_state = 42)




############# WEIGHT CLASS 

class Test_Topics(keras.layers.Layer):
    def __init__(self, units):
        super(Test_Topics, self).__init__()
        self.units = units
    def build(self,input_shape):
        
        self.w1 = self.add_weight(name='multiply_weight_1', shape=(self.units,), trainable=True,initializer='random_normal',regularizer=tf.keras.regularizers.l1_l2())
        self.w2 = self.add_weight(name='multiply_weight_2', shape=(self.units,), trainable=True,initializer='random_normal',regularizer=tf.keras.regularizers.l1_l2())
        self.w3 = self.add_weight(name='multiply_weight_3', shape=(self.units,), trainable=True,initializer='random_normal',regularizer=tf.keras.regularizers.l1_l2())
        self.w4 = self.add_weight(name='multiply_weight_4', shape=(self.units,), trainable=True,initializer='random_normal',regularizer=tf.keras.regularizers.l1_l2())
        
    def call(self, input1,input2,input3,input4):
          
          
        return tf.multiply(input1, self.w1),tf.multiply(input2, self.w2),tf.multiply(input3, self.w3),tf.multiply(input4, self.w4)
###################JSON ENCODER CLASS




class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


######################################
#Call backs class for resetting weights:

class Weight_Reset(keras.callbacks.Callback):

    def on_train_begin(self,logs):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))
        
    
    def on_train_batch_end(self,batch,logs = {}):
        
        keys = list(logs.keys())
        
        json_batch.write(
                json.dumps({'batch': batch, 
                            'categorical_accuracy': logs['categorical_accuracy'],
                            'loss': logs['loss'],
                            'topic_layer_weights': list(self.model.get_layer('test__topics').get_weights())
                            #,'model_output' : list(self.model.layers[-1].output)
                },cls = NumpyArrayEncoder) + '\n')
        
    
    def on_epoch_begin(self,epoch,logs={}):
        #print('Entered the weights callback ...')
        return

    def on_epoch_end(self, epoch, logs={}):
        layer_index = -7  ## index of the layer you want to change
        # random weights to reset the layer
        #new_weights = [np.random.rand(1),np.random.rand(1),np.random.rand(1),np.random.rand(1)]

        #self.model.layers[layer_index].set_weights(new_weights)





######################################
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


######################################
class_names = np.array(sorted([dir1 for dir1 in os.listdir(train_dir)]))


print('LOADING THE DATA ON TO TENSORS......')



print('Cleaning prior cache ...')
purge(cache_dir,regu_v)   
print('value cache cleared ....')
purge(cache_dir,regu_t)
print('train cache cleared ....')
training_data = train_csv
valid_data = val_csv
print(len(training_data))
print(len(valid_data))


#train_data = training_data['Image']
#train_labels = training_data['Label']
#class_weight = class_weight.compute_class_weight('balanced',np.unique(training_data['Label']),training_data['Label'])
#Adding sample weights
#sample_weights = class_weight/np.max(class_weight)

#print('Class weights: ....')
#weights = {i:el for el,i in zip(class_weight,range(0,24))}
#print(weights)

#print(len(train_data))
#val_labels = valid_data['Label']
#val_data = valid_data['Image']
#print(len(val_data))
#steps = int(len(train_data)/t_batch_size)
#steps_val = int(len(val_data)/v_batch_size)
print('Loading gens ...')
datagen = ImageDataGenerator(rescale=1./255)
trainGen = datagen.flow_from_dataframe(training_data,directory = train_dir,x_col="Image",y_col="Label",target_size=(224,224),color_mode='rgb',batch_size=t_batch_size,shuffle=True,class_mode='categorical',seed=42)
valGen = datagen.flow_from_dataframe(valid_data,directory = val_dir,x_col="Image",y_col="Label",target_size=(224,224),color_mode='rgb',batch_size=v_batch_size,shuffle=True,class_mode='categorical',seed=42)

steps = trainGen.n//trainGen.batch_size
steps_val = valGen.n//valGen.batch_size

print('Defining the model ..')
#base_model = keras.models.load_model(os.path.join(save_dir,'baseline_p365_without_top'))

base_model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
#x = base_model.output
#x = AveragePooling2D(pool_size=(7,7))(x)
#x = Flatten(name ="flatten")(x)
#x = Dense(1024,activation="relu")(x)
#x = Dropout(0.5)(x)
#preds = Dense(1000,name = 'topics',activation="softmax")(x)

#Activating the model

#model1 = Model(inputs = base_model.input,outputs = preds)


for layer in base_model.layers:
  layer.trainable = False
#for layer in model.layers:
#  layer.trainable = True

wt_add = Test_Topics(1)
#new_tensor = tf.reshape(wt_add, shape=[tf.shape(wt_add)[0]*tf.shape(wt_add)[1],4])
#sum_layer = wt_add(k_1,k_2,k_3,k_4)
sum_layer = wt_add(base_model.output,base_model.output,base_model.output,base_model.output)

conc = keras.layers.Concatenate(name = 'conc')(sum_layer)
#flat = keras.layers.GlobalAveragePooling1D()(conc)
#test = keras.layers.GlobalAveragePooling1D()(flat)
flat_2 = keras.layers.Flatten(name = 'flat_2')(conc)
topic_flat = Dense(2000,name = 'first_dense',activation = 'relu')(flat_2)
topic = Dropout(0.45,name = 'drop_1')(topic_flat)
topic = Dense(256,name = 'int_topic',activation='relu')(topic)
topic = Dropout(0.2,name = 'drop_2')(topic)
final = Dense(24,name = 'places',activation='softmax')(topic)


cent_model = Model(inputs = base_model.input,outputs = final)

cent_model.summary()
opti = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999, epsilon=0.01,amsgrad=True)



#######CALL BACKS###################

#if not isfile(json_log):
json_batch = open(test_filename, mode='wt', buffering=1)
json_log = open(json_filename, mode='wt', buffering=1)
callbacks = [
  ModelCheckpoint(
    
    filepath=os.path.join(chk_dir,model_name),
    save_best_only=True,  # Only save a model if `val_loss` has improved.
    monitor="categorical_accuracy",
    verbose=1,
    save_freq = 'epoch',mode = 'max'
  ),
  #EarlyStopping(
  #  monitor="loss",
  #  min_delta=0,
  #  patience=13,
  #  verbose=1,
  #  mode="auto",
  #  baseline=None,
  #  restore_best_weights=True,
  #),
  ReduceLROnPlateau(monitor='loss', factor=0.25,
                          patience=2, min_lr=1e-10,verbose=1),
  
  LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 
                            'categorical_accuracy': logs['categorical_accuracy'],
                            'loss': logs['loss'],
                            'topic_layer_weights': list(cent_model.get_layer('test__topics').get_weights())},cls = NumpyArrayEncoder) + '\n'),
            on_train_end=lambda logs: json_log.close()
  ),
  Weight_Reset()

         
]


###################################

if retrain == True:
    print('Retrain has been set to true ...')
    print('Retraining the existing model. Loading model from'+prev_weights_dir+'...')
    cent_model = keras.models.load_model(os.path.join(prev_model_dir,prev_model))
    print('Previous weights have been loaded from :'+os.path.join(prev_model_dir,prev_model)+'...')
else:
    print('Training the model from scratch as retrain is set to False ...')
print('Model loaded starting training...')
print('Accessing the model directory. creating if needed....')
if not os.path.isdir(model_dir):
  print(os.path.join(save_dir,model_dir)+' not found. Creating ....')
  os.mkdir(model_dir)
if not os.path.isdir(history_dir):
  print('Creating '+history_dir +'...')
  os.mkdir(history_dir)
if not os.path.isdir(plot_dir):
  os.mkdir(plot_dir)
  print('Creating '+plot_dir+'...')
if not os.path.isdir(chk_dir):
  os.mkdir(chk_dir)
  print('Creating '+chk_dir+'...')
if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)
    print('Creating '+weights_dir+'...')

print('Loading model from best performing checkpoint')

#cent_model = keras.models.load_model('/data/s4133366/saved_models/topic_weight_k4_imp_1_1/topic_weight_k4_imp_1_1_chk/topic_weight_k4_imp_1_1',compile = False)
#cent_model.load_weights('/data/s4133366/saved_models/topic_weight_k4_imp_1_3/topic_weight_k4_imp_1_3_weights/topic_weight_k4_imp_1_3_weights.h5')
cent_model.compile(optimizer=opti,metrics=[tf.keras.metrics.CategoricalAccuracy(),keras.metrics.Precision(),keras.metrics.Recall()],loss ='categorical_crossentropy')
print('Folders checked. Training model ...')
history = cent_model.fit(trainGen,epochs=num_epochs,callbacks=callbacks,verbose=1,workers=8,use_multiprocessing = False,steps_per_epoch = steps,shuffle = False)
cent_model.save(os.path.join(model_dir,model_name))
report = history.history
np.save(os.path.join(history_dir,train_metrics_path), report)
history_filename = os.path.join(history_dir,train_metrics_path)
print('Results saved to disk ...')

print('Evaluating the model ...') 
cent_model = keras.models.load_model(os.path.join(model_dir,model_name))

results = cent_model.evaluate(valGen,workers = 4,steps = steps_val,use_multiprocessing = True)
results = dict(zip(cent_model.metrics_names,results))


results_filename = os.path.join(history_dir,res_path)
print('Storing results in :'+results_filename)
np.save(os.path.join(history_dir,res_path), results)


print('Trying to plot ...')
try:
  print('Storing the plots in '+plot_dir+'....')    
  plot_metrics(history_filename,results_filename)
except : 
  print('Something went wrong while plotting.Try running the plot_metrics function manually. Skipping ...')

print('Metrics have been plotted ...')

print('Saving model ...')

try:
    print('Storing model in:'+os.path.join(model_dir,model_name+'_eval'))
    cent_model.save(os.path.join(model_dir,model_name+'_eval'))
    print('Stored model in:'+os.path.join(model_dir,model_name+'_eval'))
except:
    print('Error : Model could not be saved again.Skipping ...')
    
print('Storing model weights in:'+os.path.join(weights_dir,model_name+'_weights'))
try:
    cent_model.save_weights(os.path.join(weights_dir,model_name+'_weights.h5'))
    print('Stored model weights in:'+os.path.join(weights_dir,model_name+'_weights.h5'))
except:
    print('Model weights could not be stored.Skipping....')
    
print('Plotting the model weights')
try:
    final_data = make_topic_df(json_filename)
    final_data.head(1)
    print('Plotting the weights')
    plot_topics(final_data)
    print('Topic Weights have been plotted ....')
except:
    print('Something went wrong. Plot them manually')

    
print('Models and Weights have been stored successfully....Clearing session')





tf.keras.backend.clear_session()


print('Clearning cache ...')
#purge(cache_dir,regu_v)   
print('value cache cleared ....')
#purge(cache_dir,regu_t)
print('train cache cleared ....')

print('Cache cleared')




print('Job ended ...')