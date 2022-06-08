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
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.utils import shuffle
import json
import os, re


#batch_logsfile = 'test.json'
t_batch_size = 128
v_batch_size = 128
num_epochs =10
model_name = 'baseline_model_kfold_new_3_'

learning_rate = 0.01
#Strictly responsible for retraining the data .......
#retrain = True
#Data directory stuff ###############################################
cache_dir = '/data/s4133366/data_cache'
train_dir = '/data/s4133366/data/train'
save_dir = '/data/s4133366/saved_models'
data_dir = '/data/s4133366/data'
####################################################################
train_csv_file = 'train_updated.csv'
val_csv_file = 'val_updated.csv'
####################################################################
cache_t = os.path.join(cache_dir,model_name+'_t')
cache_v = os.path.join(cache_dir,model_name+'_v')
####################################################################
regu_t = os.path.split(cache_t)[-1]+'*'
regu_v = os.path.split(cache_v)[-1]+'*'
print(regu_v,regu_t)
model_dir = os.path.join(save_dir,model_name)
#Directory for plots....
plot_dir = os.path.join(model_dir,'plots')
#Directory for storing the numpy arrays for test and train ...
history_dir = os.path.join(model_dir,'history')





#Function for clearing cache
def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))
#Function for making and storing the plots

def plot_metrics(history_path):
  read_train = np.load(history_path,allow_pickle=True).item()
  #read_test = np.load(results_path,allow_pickle=True).item()
  history_path = os.path.split(history_path)[-1]
  get_fold = history_path[-5]
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
  plt.plot(read_train['val_categorical_accuracy'])
  #plt.axhline(read_test['categorical_accuracy'], color='g', linestyle='--')
  #plt.plot(read_test['categorical_accuracy'])
  plt.title('model accuracy for fold_'+str(get_fold))
  plt.ylabel('Categorical Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'accuracy_'+str(history_path)+'.png'))
  plt.clf()
  
  # summarize history for loss
  
  plt.plot(read_train['loss'])
  plt.plot(read_train['val_loss'])
  #plt.axhline(read_test['loss'], color='g', linestyle='--')
  plt.title('model loss for fold_'+str(get_fold))
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  #plt.show()
  plt.savefig(os.path.join(folder_path,'loss_'+str(history_path)+'.png'))
  plt.clf()
  
  #summarize history for Precision 
  plt.plot(read_train['precision'])
  #plt.axhline(read_test['precision'], color='g', linestyle='--')
  plt.plot(read_train['val_precision'])
  plt.title('model precision for fold_'+str(get_fold))
  plt.ylabel('Precision')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'precision_'+str(history_path)+'.png'))
  plt.clf()
  
  #summarize history for Recall 
  
  plt.plot(read_train['recall'])
  plt.plot(read_train['val_recall'])
  #plt.axhline(read_test['recall'], color='g', linestyle='--')
  plt.title('Model Recall for fold_'+str(get_fold))
  plt.ylabel('Recall')
  plt.xlabel('Epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(os.path.join(folder_path,'recall_'+str(history_path)+'.png'))
  plt.clf()



print('########## BASELINE MODEL TRAINING USING STRATIFIED K FOLD WITHOUT SHUFFLE... ######################')
print('Loading CSVs')
train_meta = pd.read_csv(os.path.join(data_dir,train_csv_file))
val_meta = pd.read_csv(os.path.join(data_dir,val_csv_file))
#test_meta = pd.read_csv('/data/s4133366/data/test.csv')


train_met = pd.DataFrame(train_meta)
val_met = pd.DataFrame(val_meta)

print('Combining the train and val data for stratified k folding ... ')
tv = [train_met,val_met]
tnv = pd.concat(tv)

tnv = tnv.sample(frac=1,random_state = 42)

train = tnv

Y = train['Label']
X = train['Image']

def get_label(file_path):
  # convert the path to a list of path components separated by sep
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.cast(one_hot, tf.int32)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)  
  # resize the image to the desired size
  return tf.image.resize(img, [224, 224])

def process_TL(file_path):
  label = get_label(file_path) 
  img = tf.io.read_file(file_path) 
  img = decode_img(img)
  img = preprocess_input(img)
  #img = tf.cast(img/255. ,tf.float32)
  
  return img, label

class_names = np.array(sorted([dir1 for dir1 in os.listdir('/data/s4133366/data/train')]))
num_folds = 4
skfold = StratifiedKFold(n_splits=num_folds, shuffle=True,random_state=42)
fold_no = 1

print('LOADING THE DATA ON TO TENSORS......')

  

for train_index, val_index in skfold.split(X,Y):
    training_data = train.iloc[train_index]
    valid_data = train.iloc[val_index]
    print(len(training_data))
    print(len(valid_data))
    train_data = training_data['Image']
    train_labels = training_data['Label']
    val_labels = valid_data['Label']
    val_data = valid_data['Image']
    steps = int(len(train_data)/t_batch_size)
    steps_val = int(len(val_data)/v_batch_size)
    
    train_tensor = tf.data.Dataset.from_tensors(train_data)
    val_tensor = tf.data.Dataset.from_tensors(val_data)
    
    train_tensor = train_tensor.unbatch()
    val_tensor = val_tensor.unbatch()
    
    train_tensor = train_tensor.shuffle(len(train_data))
    val_tensor = val_tensor.shuffle(len(val_data))
    
    
    
    print('Shuffling the dataframes internally for more randomness...')
    
    tra_tens = train_tensor.map(process_TL)
    val_tens = val_tensor.map(process_TL)
    
    train_cache = tra_tens.cache(cache_t)
    val_cache = val_tens.cache(cache_v)
    print('Created cache ...')
    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    val_batch = val_cache.batch(v_batch_size)
    train_batch= train_cache.repeat(num_epochs).batch(t_batch_size)
    train_ds = train_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(train_ds)
    
    
    # class_weights = class_weight.compute_class_weight(
    #          'balanced',
    #           np.unique(ktrainGen.classes), 
    #           ktrainGen.classes)
    
    # train_class_weights = dict(enumerate(class_weights))
    
    #Create model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten(name ="flatten")(x)
    x = Dense(128,activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(24,activation="softmax")(x)
    
    # #Activating the model

    model = Model(inputs = base_model.input,outputs = preds)
    
    model.summary()
    
    #if fold_no == 1:
        #print('plotting model in '+os.path.join(plot_dir,'baseline_kfold.png')+' ...')
        #tf.keras.utils.plot_model(model, to_file=os.path.join(plot_dir,'baseline_kfold.png'), show_shapes=True,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    for layer in base_model.layers:
        layer.trainable = True

    chk_dir = os.path.join(model_dir,model_name+str(fold_no)+'_chk')  
    opti = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95, epsilon=1e-07, name="Adadelta")
    model.compile(optimizer=opti,metrics=[tf.keras.metrics.CategoricalAccuracy(),keras.metrics.Precision(),keras.metrics.Recall()],loss='categorical_crossentropy')

    callbacks = [
        ModelCheckpoint(
            
            filepath=os.path.join(chk_dir,model_name+str(fold_no)),
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="categorical_accuracy",
            verbose=1,
            save_freq = 'epoch',mode = 'max'
        ),
        EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=7,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        ),
        ReduceLROnPlateau(monitor='loss', factor=0.1,
                                  patience=3, min_lr=1e-5,verbose=1)
        ]
    
    
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
      
    #Responsible for the name of the npy files for both test and train
    train_metrics_path = 'history_'+str(model_name)+str(fold_no)+'.npy'
    res_path = 'results_'+str(model_name)+str(fold_no)+'.npy'
    #Responsible for Checkpointing
    #Saving weights for retraining ....
    weights_dir = os.path.join(model_dir,model_name+str(fold_no)+'_weights')
    
    
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
        
        
      
    model.load_weights('/data/s4133366/saved_models/baseline_model_kfold_new_1_/baseline_model_kfold_new_1_'+str(fold_no)+'_weights/baseline_model_kfold_new_1_'+str(fold_no)+'_weights.h5')    
    history = model.fit(train_ds,epochs=num_epochs,callbacks=callbacks,verbose=1,workers=8,use_multiprocessing = False,shuffle=True,steps_per_epoch = steps,validation_data = val_ds,validation_steps = steps_val)
    model.save(os.path.join(model_dir,model_name+str(fold_no)))
    report = history.history
      
    np.save(os.path.join(history_dir,train_metrics_path), report)
    history_filename = os.path.join(history_dir,train_metrics_path)
    print('Results saved to disk ...')
    
    #print('Evaluating the model ...') 
    #model = keras.models.load_model(os.path.join(model_dir,model_name+str(fold_no)))
    
    #results = model.evaluate(val_ds,workers = 4,steps = steps_val,use_multiprocessing = True)
    #results = dict(zip(model.metrics_names,results))
    
    
    #results_filename = os.path.join(history_dir,res_path)
    #print('Storing results in :'+results_filename)
    #np.save(os.path.join(history_dir,res_path), results)
    
    
    print('Trying to plot ...')
    try:
        print('Storing the plots in '+plot_dir+'....')    
        plot_metrics(history_filename)
    except : 
        print('Something went wrong while plotting.Try running the plot_metrics function manually. Skipping ...')
    
    print('Metrics have been plotted ...')
    
    print('Saving model ...')
    
    try:
        print('Storing model in:'+os.path.join(model_dir,model_name+str(fold_no)+'_eval'))
        model.save(os.path.join(model_dir,model_name+str(fold_no)+'_eval'))
        print('Stored model in:'+os.path.join(model_dir,model_name+str(fold_no)+'_eval'))
    except:
        print('Error : Model could not be saved again.Skipping ...')
    
    
    
    print('Storing model weights in:'+os.path.join(weights_dir,model_name+str(fold_no)+'_weights'))
    try:
        model.save_weights(os.path.join(weights_dir,model_name+str(fold_no)+'_weights.h5'))
        print('Stored model weights in:'+os.path.join(weights_dir,model_name+str(fold_no)+'_weights.h5'))
    except:
        print('Model weights could not be stored.Skipping....')
    
 
    
    print('Models and Weights have been stored successfully....Clearing session')





    tf.keras.backend.clear_session()


    print('Clearning cache ...')
    purge(cache_dir,regu_v)   
    print('value cache cleared ....')
    purge(cache_dir,regu_t)
    print('train cache cleared ....')

    print('Cache cleared')

    fold_no = fold_no+1
    print('Going to next fold '+str(fold_no)+'................')

print('Job ended ...')
