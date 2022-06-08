######################       MODEL PREDICTION USING IMAGE DATA GENERATORS ###################################################
######                                  ONLY CHANGE THE MODEL NAME ....   ###################################################
#############################################################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow import keras
import tensorflow as tf
import seaborn as sn
from sklearn.utils import shuffle
########################
model_name = 'topic_weight_k4_imp_4_4'
predictions_npy_file = str(model_name)+'_predictions.npy'
data_dir = '/data/s4133366/data/train'
cr_name = 'Classification_Report_'+model_name+'.csv'
cm_name = 'Confusion matrix_'+model_name+'.png'
num_classes = 24
test_csv_file = '/data/s4133366/data/test.csv'
save_dir = '/data/s4133366/saved_models'
model_dir = os.path.join(save_dir,model_name)
pred_dir = os.path.join(model_dir,'predictions')
cr_path = os.path.join(pred_dir,cr_name)
cm_path = os.path.join(pred_dir,cm_name)
test_csv = pd.read_csv(test_csv_file)
#############################
#In case data generators work
datagen = ImageDataGenerator(rescale=1./255)
testGen = datagen.flow_from_dataframe(test_csv,data_dir,x_col="Image",y_col="Label",target_size=(224,224),color_mode='rgb',batch_size=116,class_mode=None,shuffle=False,seed=42)
#num_test_samples = len(testGen.filenames)
#num_test_classes = len(testGen.class_indices)
#STEP_SIZE_TEST = testGen.n//testGen.batch_size
########################## LOADING THE LATEST MODEL
model = keras.models.load_model(os.path.join(model_dir,model_name))
print('Model Loaded: '+os.path.join(model_dir,model_name)+' ...')
print("Predicting the model")
testGen.reset()
predictions = model.predict(testGen,verbose=1,workers = 4,use_multiprocessing = False)
################################################


if not os.path.isdir(pred_dir):
    print('Predictions directory not present. Creating ...')
    os.mkdir(pred_dir)

print('Predictions directory not present. Creating ...')

######## TESTING




##########################




print('Saving the predictions as a csv for reference ...')
np.save('predictions.npy',predictions)
#pd.DataFrame(predictions).to_csv(os.path.join(pred_dir,predictions_npy_file))
np.save(os.path.join(pred_dir,predictions_npy_file),predictions)
predictions_test = np.argmax(predictions,axis=1)


#### CHANGE THIS WITH RESPECT TO THE TYPE OF DATA AUG USED #############
y_true = testGen.classes
y_pred = predictions_test
labels = testGen.class_indices.keys()
#########################################################################################################
########################################################################################################
class_names = list(sorted([dir1 for dir1 in os.listdir(data_dir)]))
print('Class names are of :',type(class_names))
############### PREDICTION PLOTTING

class_report = classification_report(y_true, y_pred,target_names=labels,output_dict=True)
df_cr = pd.DataFrame(class_report).transpose()
df_cr.to_csv(cr_path)
print('Classification report saved to: '+cr_path)
print(classification_report(y_true, y_pred,target_names=labels))

print('Plotting the confusion matrix ....')
cm = confusion_matrix(y_true,y_pred,labels =  list(range(0,num_classes)) )

df_cm = pd.DataFrame(cm,index = class_labels,columns = class_labels )
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=False, annot_kws={"size": 20},cmap="BuPu") # font size
plt.title('Confusion matrix for '+model_name)
plt.savefig(cm_path,bbox_inches="tight")
print('Model predicted ....')
print('Job ended ...')