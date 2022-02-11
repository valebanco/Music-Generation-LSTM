"""
# Analisi grafica del modello di rete neurale per la generazione di brani musicali

In primo luogo vi saranno un'insieme di sezioni per:

- ***L'Analisi generica sul training del modello***
- ***Analisi grafica basata su metriche e attuate su feature: un confronto***

In secondo luogo vi sara:

- una sezione per la ***Valutazione del modello in base al Test set a disposizione***. Essa fornirà un report di classificazione e una matrice di confusione. Inoltre, essa garantirà una valutazione del modello di fronte ad un test set.
"""

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import glob,re
from keras.models import Sequential,load_model
import pickle
import plotly.graph_objects as go



"""## Analisi generica del modello

Tale analisi fornirà due grafici:

1. grafico di andamento dell'accuracy nei dati di training e validation nell'addestramento del modello;
2.  grafico di andamento del loss nei dati di training e validation nell'addestramento del modello;
"""

history_root_path = "../../reports/history/set_history_dataset_pickle/**/*.npz"

list_acc_tra = []
list_val_acc_tra = []
list_loss_tra = []
list_val_loss_tra = []

#ordinamento file da processare (ordinamento per epoche)
files = glob.glob(history_root_path)
files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

#costruzione vettori di loss,accuracy,validation_loss,validation_accuracy
for infile in files:
    splitted_name_file = infile.split("/")
    cur_file_npz = splitted_name_file[3]
    
    if(cur_file_npz == "accuracy_training.npz"):
      acc_tra = np.load(infile ,mmap_mode='r')
      list_acc_tra.append(acc_tra['arr_0'])

    if(cur_file_npz == "loss_training.npz"):
      loss_tra = np.load(infile ,mmap_mode='r')
      list_loss_tra.append(loss_tra['arr_0'])
    
    if(cur_file_npz == "val_accuracy_training.npz"):
      val_acc_tra = np.load(infile ,mmap_mode='r')
      list_val_acc_tra.append(val_acc_tra['arr_0'])
    
    if(cur_file_npz == "val_loss_training.npz"):
      val_loss_tra = np.load(infile ,mmap_mode='r')
      list_val_loss_tra.append(val_loss_tra['arr_0'])

#flatting dei vettori in 1D
result_acc_tra = [item for sublist in list_acc_tra for item in sublist]
result_val_acc_tra = [item for sublist in list_val_acc_tra for item in sublist]
result_loss_tra = [item for sublist in list_loss_tra for item in sublist]
result_val_loss_tra = [item for sublist in list_val_loss_tra for item in sublist]

# visualizzazione grafico di loss
epochs = range(1, len(result_loss_tra) + 1)
plt.plot(epochs, result_loss_tra, 'bo', label='Training loss')
plt.plot(epochs, result_val_loss_tra, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#visualizzazione grafico di accuracy
epochs = range(1, len(result_acc_tra) + 1)
plt.plot(epochs, result_acc_tra, 'bo', label='Training accuracy')
plt.plot(epochs, result_val_acc_tra, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

"""# Analisi grafica basata su metriche e attuate su feature: un confronto

Questa Analisi pone come obbiettivo quello di:

- **confrontare le metriche di loss e accuracy sia di training che di validation** lungo l'asse temporale (epoche) delle varie feature messe in risalto
- **confrontare singolarmente (per ogni feature) l'andamento dell'addestramento** su dati di training e validation;
- fornire una *rappresentazione tabellare* che consenta di mettere a confronto i **valori ottimali-epoche in corrispondenza del valore ottimale** *(es. loss_ottimale-epoca del loss_ottimale*) per ogni feature esaminata.
"""

def flat_metrics_list_feature(list_metric_tra_features):
  result_list_cur_metric_Note = []
  result_list_cur_metric_Offset = []
  result_list_cur_metric_Duration = []
  result_list_cur_metric_Velocity = []
  result_list_cur_metric_Tempo = []
  
  for i in range(0,len(list_metric_tra_features)):
    result_list_cur_metric_Note.append(list_metric_tra_features[i][0])
    result_list_cur_metric_Offset.append(list_metric_tra_features[i][1])
    result_list_cur_metric_Duration.append(list_metric_tra_features[i][2])
    result_list_cur_metric_Velocity.append(list_metric_tra_features[i][3])
    result_list_cur_metric_Tempo.append(list_metric_tra_features[i][4])
  
  #flatting dei vettori in 1D
  result_metric_Note_flatted = [item for sublist in result_list_cur_metric_Note for item in sublist]
  result_metric_Offset_flatted = [item for sublist in result_list_cur_metric_Offset for item in sublist]
  result_metric_Duration_flatted = [item for sublist in result_list_cur_metric_Duration for item in sublist]
  result_metric_Velocity_flatted = [item for sublist in result_list_cur_metric_Velocity for item in sublist]
  result_metric_Tempo_flatted = [item for sublist in result_list_cur_metric_Tempo for item in sublist]
  
  return result_metric_Note_flatted,result_metric_Offset_flatted,result_metric_Duration_flatted,result_metric_Velocity_flatted,result_metric_Tempo_flatted


list_acc_tra_features = []
list_val_acc_tra_features = []
list_loss_tra_features = []
list_val_loss_tra_features = []

files = glob.glob(history_root_path)
files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for file in files:
  splitted_name_file = file.split("/")
  cur_file_npz = splitted_name_file[3]
  
  if (cur_file_npz == "accuracy_features.npz"):
    acc_tra_features = np.load(file ,mmap_mode='r')
    list_acc_tra_features.append( acc_tra_features['arr_0'])
  if (cur_file_npz == "loss_features.npz"):
    loss_tra_features = np.load(file ,mmap_mode='r')
    list_loss_tra_features.append( loss_tra_features['arr_0'])
  if (cur_file_npz == "val_accuracy_features.npz"):
    val_acc_tra_features = np.load(file ,mmap_mode='r')
    list_val_acc_tra_features.append( val_acc_tra_features['arr_0'])
  if (cur_file_npz == "val_loss_features.npz"):
    val_loss_tra_features = np.load(file ,mmap_mode='r')
    list_val_loss_tra_features.append( val_loss_tra_features['arr_0'])


Note_acc_flatted,Offset_acc_flatted,Duration_acc_flatted,Velocity_acc_flatted,Tempo_acc_flatted = flat_metrics_list_feature(list_acc_tra_features)

epochs = range(1, len(Note_acc_flatted) + 1)
fig_size_conf = (14,7)



# visualizzazione grafico di accuracy per feature A CONFRONTO
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_acc_flatted, 'b', label='Note_accuracy')
plt.plot(epochs, Offset_acc_flatted, 'r', label='Offset_accuracy')
plt.plot(epochs, Duration_acc_flatted, 'g', label='Duration_accuracy')
plt.plot(epochs, Velocity_acc_flatted, 'y', label='Velocity_accuracy')
plt.plot(epochs, Tempo_acc_flatted, 'c', label='Tempo_accuracy')
plt.title('features accuracy')
plt.legend()
plt.show()

Note_val_acc_flatted,Offset_val_acc_flatted,Duration_val_acc_flatted,Velocity_val_acc_flatted,Tempo_val_acc_flatted = flat_metrics_list_feature(list_val_acc_tra_features)

# visualizzazione grafico di accuracy per feature A CONFRONTO
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_val_acc_flatted, 'b', label='Note_val_accuracy')
plt.plot(epochs, Offset_val_acc_flatted, 'r', label='Offset_val_accuracy')
plt.plot(epochs, Duration_val_acc_flatted, 'g', label='Duration_val_accuracy')
plt.plot(epochs, Velocity_val_acc_flatted, 'y', label='Velocity_val_accuracy')
plt.plot(epochs, Tempo_val_acc_flatted, 'c', label='Tempo_val_accuracy')
plt.title('features validation accuracy')
plt.legend()
plt.show()

Note_loss_flatted,Offset_loss_flatted,Duration_loss_flatted,Velocity_loss_flatted,Tempo_loss_flatted = flat_metrics_list_feature(list_loss_tra_features)

# visualizzazione grafico di accuracy per feature A CONFRONTO
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_loss_flatted, 'b', label='Note_loss')
plt.plot(epochs, Offset_loss_flatted, 'r', label='Offset_loss')
plt.plot(epochs, Duration_loss_flatted, 'g', label='Duration_loss')
plt.plot(epochs, Velocity_loss_flatted, 'y', label='Velocity_loss')
plt.plot(epochs, Tempo_loss_flatted, 'c', label='Tempo_loss')
plt.title('features loss')
plt.legend()
plt.show()

Note_val_loss_flatted,Offset_val_loss_flatted,Duration_val_loss_flatted,Velocity_val_loss_flatted,Tempo_val_loss_flatted = flat_metrics_list_feature(list_val_loss_tra_features)


# visualizzazione grafico di accuracy per feature A CONFRONTO
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_val_loss_flatted, 'b', label='Note_val_loss')
plt.plot(epochs, Offset_val_loss_flatted, 'r', label='Offset_val_loss')
plt.plot(epochs, Duration_val_loss_flatted, 'g', label='Duration_val_loss')
plt.plot(epochs, Velocity_val_loss_flatted, 'y', label='Velocity_val_loss')
plt.plot(epochs, Tempo_val_loss_flatted, 'c', label='Tempo_val_loss')
plt.title('features validation loss')
plt.legend()
plt.show()



# visualizzazione grafico di accuracy e validation accuracy per feature: Note
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_acc_flatted, 'o', label='Note_accuracy')
plt.plot(epochs, Note_val_acc_flatted, 'b', label='Note_val_accuracy')
plt.title('Training accuracy Note')
plt.legend()
plt.show()

# visualizzazione grafico di loss e validation loss per feature: Note
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Note_loss_flatted, 'o', label='Note_loss')
plt.plot(epochs, Note_val_loss_flatted, 'b', label='Note_val_loss')
plt.title('Training loss Note')
plt.legend()
plt.show()

# visualizzazione grafico di accuracy e validation accuracy per feature: Offset
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Offset_acc_flatted, 'o', label='Offset_accuracy')
plt.plot(epochs, Offset_val_acc_flatted, 'b', label='Offset_val_accuracy')
plt.title('Training accuracy Offset')
plt.legend()
plt.show()

# visualizzazione grafico di loss e validation loss per feature: Offset
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Offset_loss_flatted, 'o', label='Offset_loss')
plt.plot(epochs, Offset_val_loss_flatted, 'b', label='Offset_val_loss')
plt.title('Training loss Offset')
plt.legend()
plt.show()

# visualizzazione grafico di accuracy e validation accuracy per feature: Duration
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Duration_acc_flatted, 'o', label='Duration_accuracy')
plt.plot(epochs, Duration_val_acc_flatted, 'b', label='Duration_val_accuracy')
plt.title('Training accuracy Duration')
plt.legend()
plt.show()

# visualizzazione grafico di loss e validation loss per feature: Duration
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Duration_loss_flatted, 'o', label='Duration_loss')
plt.plot(epochs, Duration_val_loss_flatted, 'b', label='Duration_val_loss')
plt.title('Training loss Duration')
plt.legend()
plt.show()

# visualizzazione grafico di accuracy e validation accuracy per feature: Velocity
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Velocity_acc_flatted, 'o', label='Velocity_accuracy')
plt.plot(epochs, Velocity_val_acc_flatted, 'b', label='Velocity_val_accuracy')
plt.title('Training accuracy Velocity')
plt.legend()
plt.show()

# visualizzazione grafico di loss e validation loss per feature: Velocity
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Velocity_loss_flatted, 'o', label='Velocity_loss')
plt.plot(epochs, Velocity_val_loss_flatted, 'b', label='Velocity_val_loss')
plt.title('Training loss Velocity')
plt.legend()
plt.show()

# visualizzazione grafico di accuracy e validation accuracy per feature: Tempo
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Tempo_acc_flatted, 'o', label='Tempo_accuracy')
plt.plot(epochs, Tempo_val_acc_flatted, 'b', label='Tempo_val_accuracy')
plt.title('Training accuracy Tempo')
plt.legend()
plt.show()

# visualizzazione grafico di loss e validation loss per feature: Tempo
plt.figure(figsize = fig_size_conf)
plt.plot(epochs, Tempo_loss_flatted, 'o', label='Tempo_loss')
plt.plot(epochs, Tempo_val_loss_flatted, 'b', label='Tempo_val_loss')
plt.title('Training loss Tempo')
plt.legend()
plt.show()

#visualizzazione tabella con schema: Feature,max_accuracy-epoch,max_val_accuracy-epoch,min_loss-epoch,min_val_loss-epoch

biggest_accuracy = [
str(max(Note_acc_flatted)) + "-" + str(Note_acc_flatted.index(max(Note_acc_flatted))),
str(max(Offset_acc_flatted)) + "-" + str(Offset_acc_flatted.index(max(Offset_acc_flatted))),
str(max(Duration_acc_flatted)) + "-" + str(Duration_acc_flatted.index(max(Duration_acc_flatted))),
str(max(Velocity_acc_flatted)) + "-" + str(Velocity_acc_flatted.index(max(Velocity_acc_flatted))),
str(max(Tempo_acc_flatted)) + "-" + str(Tempo_acc_flatted.index(max(Tempo_acc_flatted)))
]

biggest_val_accuracy = [
str(max(Note_val_acc_flatted)) + "-" + str(Note_val_acc_flatted.index(max(Note_val_acc_flatted))),
str(max(Offset_val_acc_flatted)) + "-" + str(Offset_val_acc_flatted.index(max(Offset_val_acc_flatted))),
str(max(Duration_val_acc_flatted)) + "-" + str(Duration_val_acc_flatted.index(max(Duration_val_acc_flatted))),
str(max(Velocity_val_acc_flatted)) + "-" + str(Velocity_val_acc_flatted.index(max(Velocity_val_acc_flatted))),
str(max(Tempo_val_acc_flatted)) + "-" + str(Tempo_val_acc_flatted.index(max(Tempo_val_acc_flatted)))
]

minim_loss = [
str(min(Note_loss_flatted)) + "-" + str(Note_loss_flatted.index(min(Note_loss_flatted))),
str(min(Offset_loss_flatted)) + "-" + str(Offset_loss_flatted.index(min(Offset_loss_flatted))),
str(min(Duration_loss_flatted)) + "-" + str(Duration_loss_flatted.index(min(Duration_loss_flatted))),
str(min(Velocity_loss_flatted)) + "-" + str(Velocity_loss_flatted.index(min(Velocity_loss_flatted))),
str(min(Tempo_loss_flatted)) + "-" + str(Tempo_loss_flatted.index(min(Tempo_loss_flatted)))
]

minim_val_loss = [
str(min(Note_val_loss_flatted)) + "-" + str(Note_val_loss_flatted.index(min(Note_val_loss_flatted))),
str(min(Offset_val_loss_flatted)) + "-" + str(Offset_val_loss_flatted.index(min(Offset_val_loss_flatted))),
str(min(Duration_val_loss_flatted)) + "-" + str(Duration_val_loss_flatted.index(min(Duration_val_loss_flatted))),
str(min(Velocity_val_loss_flatted)) + "-" + str(Velocity_val_loss_flatted.index(min(Velocity_val_loss_flatted))),
str(min(Tempo_val_loss_flatted)) + "-" + str(Tempo_val_loss_flatted.index(min(Tempo_val_loss_flatted)))
]

fig = go.Figure(data=[go.Table(
    header=dict(values=['Feature', 'max_accuracy-epoch','max_val_accuracy-epoch','min_loss-epoch','min_val_loss-epoch'],
                line_color='darkslategray',
    fill_color='royalblue',),

    cells=dict(values=[['Note','Offset','Duration','Velocity','Tempo'], # 1st column
                       biggest_accuracy,
                       biggest_val_accuracy,
                       minim_loss,
                       minim_val_loss], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=1000, height=1000)
fig.show()



