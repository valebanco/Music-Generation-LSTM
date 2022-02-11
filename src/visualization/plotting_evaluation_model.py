"""# Valutazione del modello in base al Test set a disposizione

### REPORT DI CLASSIFICAZIONE SU DATI PRESENTI NEL TEST SET
Questo report pone come obbiettivo quello di presentare il comportamento del modello di fronte a un test set. L'impiego delle varie metriche saranno usate al fine di valutare un modello.

### MATRICE DI CONFUSIONE
la valutazione della matrice di confusione è un modo per valutare la tendenza del modello nel fornire predizioni attendibili anche di fronte a dati non compresi nel test di addestramento o di validation.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import glob,re
from keras.models import Sequential,load_model
import pickle
import plotly.graph_objects as go


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

X_test = load_set("X_test.pickle","../../data/processed/dataset_pickle")
y_test = load_set("y_test.pickle","../../data/processed/dataset_pickle")
model = load_model("../../models/model")

evaluate_model(model,128,X_test,y_test)