
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Bidirectional

from keras.callbacks import ModelCheckpoint

from keras.optimizers import adam_v2

from keras import Model
#PROSECUZIONE TRAINING PARTENDO DA UN MODELLO SALVATO
def extract_sets (ROOT_PATH_DATA):
  X_train = load_set("X_train.pickle",ROOT_PATH_DATA)
  y_train = load_set("y_train.pickle",ROOT_PATH_DATA)
  X_validation = load_set("X_validation.pickle",ROOT_PATH_DATA)
  y_validation = load_set("y_validation.pickle",ROOT_PATH_DATA)

  return X_train,y_train,X_validation,y_validation

def load_set (name_set,path):
  filepath1 = open(path + "/" + name_set, 'rb')
  r_set = pickle.load(filepath1)
  return r_set

def evaluate_model(model,batch_size,X_test,y_test):
  # Evaluate the model on the test data using `evaluate`
  y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

  y_pred_bool = np.argmax(np.array(y_pred[0]), axis=1)
  y_test_bool = np.argmax(np.array(y_test[0]), axis=1)

  print("------------------confusion matrix---------------------")

  cm = confusion_matrix(y_test_bool,y_pred_bool)
  labels = set(y_test_bool)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

  # figsize per modificare la grandezza della matrice di confusione
  fig, ax = plt.subplots(figsize=(35,35))
  disp.plot(cmap=plt.cm.Blues,ax=ax)
  plt.show()

  print("-------------------------------------------------------")

  print(classification_report(y_test_bool, y_pred_bool))

  print("-----------------Evaluate on test data-----------------")

  results = model.evaluate(X_test, y_test, batch_size=batch_size)
  print("test loss", results[0])
  print("test accuracy", (results[4] + results[5] + results[6])/3 )

  print("-------------------------------------------------------")

def plot_accuracy_validation(history):
  print(history.history.keys())
  complex_acc = [history.history['Note_acc'],history.history['Offset_acc'],history.history['Duration_acc'],history.history['Velocity_acc'],history.history['Tempo_acc']]
  summed_acc = list(map(sum, zip(*complex_acc)))
  
  complex_val_acc = [history.history['val_Note_acc'],history.history['val_Offset_acc'],history.history['val_Duration_acc'],history.history['val_Velocity_acc'],history.history['val_Tempo_acc']]
  summed_val_acc = list(map(sum, zip(*complex_val_acc)))
  
  n_feature = 5
  acc = [x / n_feature for x in summed_acc]
  val_acc = [x / n_feature for x in summed_val_acc]

  save_history_element("accuracy_training.npz",acc)
  save_history_element("val_accuracy_training.npz",val_acc)
  save_history_element("accuracy_features.npz",complex_acc)
  save_history_element("val_accuracy_features.npz",complex_val_acc)

  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.show()
  
def plot_loss_validation(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  complex_loss = [history.history['Note_loss'],history.history['Offset_loss'],history.history['Duration_loss'],history.history['Velocity_loss'],history.history['Tempo_loss']]
  complex_val_loss = [history.history['val_Note_loss'],history.history['val_Offset_loss'],history.history['val_Duration_loss'],history.history['val_Velocity_loss'],history.history['val_Tempo_loss']]

  save_history_element("loss_features.npz",complex_loss)
  save_history_element("val_loss_features.npz",complex_val_loss)
  save_history_element("loss_training.npz",loss)
  save_history_element("val_loss_training.npz",val_loss)

  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()
 
def save_history_element(filename,array):
   np.savez_compressed(filename,array)

def continue_training(ROOT_PATH_DATA,PATH_MODEL,PATH_NEW_MODEL,epochs,batch_size,verbose):

  model = load_model(PATH_MODEL)
  model.load_weights("weights-improvement-15-4.1109-bigger.hdf5")
  model.summary()
  X_train,y_train,X_validation,y_validation = extract_sets (ROOT_PATH_DATA)

  filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
  checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=verbose,
    save_best_only=True,
    mode='min'
  )
  callbacks_list = [checkpoint]
  """
  l'addestramenta avrà:
  - la serie di input estratte dalla parserizzazione dei file midi
  - la serie di output corrispondenti agli input estratti nella parserizzazione (OUTPUT NON DI PREDIZIONE)
  """
  #addestramento con dati di validation definiti in dati
  history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_validation,y_validation),
    epochs = epochs, 
    batch_size = batch_size,
    callbacks= callbacks_list
    )
  
  plot_accuracy_validation(history)
  plot_loss_validation(history)
  model.save(PATH_NEW_MODEL)
  """
  X_test = load_set("X_test.pickle", ROOT_PATH_DATA)
  y_test = load_set("y_test.pickle", ROOT_PATH_DATA)
  evaluate_model(model,batch_size,X_test,y_test)
  """
 
continue_training("../../data/processed/dataset_pickle","../../models/model","model_new",15,128,0)