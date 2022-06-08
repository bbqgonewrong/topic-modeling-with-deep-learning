######################       MODEL PREDICTION USING TF DATA               ###################################################
######                          ONLY CHANGE THE MODEL NAME ....           ###################################################
#############################################################################################################################
import os,re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow import keras
import tensorflow as tf
from sklearn.utils import shuffle
########################
model_name = 'topic_weight_k9_imp_2_final'
model_string = 'Topic Model Imp 2 Prediction'
db_name = 'EgoPlaces'
predictions_np_file = str(model_name)+'_predictions_tens.npy'
data_dir = '/data/s4133366/data/train'
cr_name = 'Classification_Report_'+model_name+'_tens.csv'
cm_name = 'Confusion matrix_'+model_name+'_tens.png'
num_classes = 23
test_csv_file = 'test_nn_final.csv'
csv_dir = '/data/s4133366/data'
save_dir = '/data/s4133366/saved_models'
model_dir = os.path.join(save_dir,model_name)
pred_dir = os.path.join(model_dir,'predictions_full')
cr_path = os.path.join(pred_dir,cr_name)
cm_path = os.path.join(pred_dir,cm_name)
cache_dir = '/data/s4133366/data_cache'
cache_te = os.path.join(cache_dir,model_name+'_te_full')
regu_te = os.path.split(cache_te)[-1]+'*'
batch_size = 163
#############################
#Try using the other workflow i.e from csv ...
print('Loading CSVs')

#For purging the cache ....
def purge(dir, pattern):
  for f in os.listdir(dir):
    if re.search(pattern, f):
      os.remove(os.path.join(dir, f))

print('Cleaning prior cache ...')
purge(cache_dir,regu_te)


test_csv = pd.read_csv(os.path.join(csv_dir,test_csv_file))


testing_data = test_csv

print(len(testing_data))



print('Data read and loading onto tensors.....')

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


def process_TL_Test(file_path):
  label = get_label(file_path) 
  img = tf.io.read_file(file_path) 
  img = decode_img(img)
  img = preprocess_input(img)
  img = tf.cast(img/255. ,tf.float32)
  return img
def process_TL_Labels(file_path):
  label = get_label(file_path) 
  img = tf.io.read_file(file_path) 
  img = decode_img(img)
  img = preprocess_input(img)
  img = tf.cast(img/255. ,tf.float32)
  return label
class_names = np.array(sorted([dir1 for dir1 in os.listdir(data_dir)]))

#
label_dict = {'Balcony': 0,
 'Bathroom': 1,
 'Beach': 2,
 'Bedroom': 3,
 'Buildings': 4,
 'Education,science': 5,
 'Forest, field, jungle': 6,
 'Garden': 7,
 'Hospital': 8,
 'Kitchen': 9,
 'Living room': 10,
 'Mountains, hills, desert, sky': 11,
 'Museum': 12,
 'Office': 13,
 'Other rooms': 14,
 'Others': 15,
 'Pathways': 16,
 'Recreation': 17,
 'Restaurant,Bar': 18,
 'Shop': 19,
 'Sport fields': 20,
 'Transportation': 21,
 'Water': 22}
labels = label_dict.keys()

#
print('Converting labels to integers')

test_labels = testing_data['Label']
test_labels = test_labels.map(label_dict)
test_labels = test_labels.to_list()
labels = label_dict.keys()

print('Labels used in cm plotting: ',labels)

print('LOADING THE DATA ON TO TENSORS......')

test_data = testing_data['Image']

print('Cleaning prior cache ...')
purge(cache_dir,regu_te)   
print('test cache cleared ....')
testing_data = test_csv
print(len(testing_data))



print(len(test_data))




test_tensor = tf.data.Dataset.from_tensors(test_data)


test_tensor = test_tensor.unbatch()



test_tens = test_tensor.map(process_TL_Test)


print('test_tens:',tf.data.experimental.cardinality(test_tens))

test_cache = test_tens.cache(cache_te)

#print(tf.data.experimental.cardinality(test_cache))

print('Created cache ...')

test_batch = test_cache.batch(batch_size)

test_ds = test_batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




############################
model = keras.models.load_model(os.path.join(model_dir,model_name+'_eval'))
print('Model Loaded: '+os.path.join(model_dir,model_name+'_eval')+' ...')
print("Predicting the model")

predictions = model.predict(test_ds,verbose=1,workers = 4,use_multiprocessing = False)
################################################


#### CHANGE THIS WITH RESPECT TO THE TYPE OF DATA AUG USED #############
y_true = test_labels
predictions_test = np.argmax(predictions,axis=1)
y_pred = predictions_test

#########################################################################################################
########################################################################################################


if not os.path.isdir(pred_dir):
    print('Predictions directory not present. Creating ...')
    os.mkdir(pred_dir)

print('Predictions directory not present. Creating ...')



############### PREDICTION PLOTTING
print('Saving the predictions as a csv for reference ...')
#pd.DataFrame(predictions).to_csv(os.path.join(pred_dir,predictions_csv_file))
np.save(os.path.join(pred_dir,predictions_np_file),predictions)
predictions_test = np.argmax(predictions,axis=1)

class_report = classification_report( y_true, y_pred,target_names=labels,output_dict=True)
df_cr = pd.DataFrame(class_report).transpose()
df_cr.to_csv(cr_path)
print('Classification report saved to: '+cr_path)
print(classification_report(y_true, y_pred,target_names=labels))

print('Plotting the confusion matrix ....')
cm = confusion_matrix(y_true,y_pred,labels =  list(range(0,num_classes)),normalize='all' )

df_cm = pd.DataFrame(cm,index = class_names  ,columns = class_names  )
print('Saving confusion matrix for future reference ...')
df_cm.to_csv(os.path.join(pred_dir,'cm_imp2.csv'))

plt.figure(figsize=(20,20))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 15},cmap="BuPu") # font size
plt.title('Confusion matrix for '+model_string+' using '+db_name+'Dataset')
print('Saving plot in '+cm_path+' ...')
plt.savefig(cm_path,bbox_inches="tight")
print('Model predicted ....')

print('Clearning cache ...')
purge(cache_dir,regu_te) 

print('Job ended ...')