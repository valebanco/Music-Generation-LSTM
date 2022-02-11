import pickle
import numpy
import keras
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

from keras.layers import Bidirectional

from keras.layers import concatenate
from keras.layers import Input
from tensorflow.python.framework.importer import import_graph_def

import pandas as pd

from keras import Model

def load_test_sequences(ROOT_PATH_DATA):
  filepath_X_test = open(ROOT_PATH_DATA + "/X_validation.pickle",'rb')
  test_sequences = pickle.load(filepath_X_test)

  network_input_notes = test_sequences[0]
  network_input_offsets = test_sequences[1]
  network_input_durations = test_sequences[2] 
  network_input_velocities = test_sequences[3]
  network_input_tempos = test_sequences[4] 

  return network_input_notes,network_input_offsets,network_input_durations,network_input_velocities,network_input_tempos

def load_features(ROOT_PATH_DATA):

  filepath_notes = open(ROOT_PATH_DATA + "/notes.pickle",'rb')
  notes = pickle.load(filepath_notes)

  filepath_offsets = open(ROOT_PATH_DATA + "/offsets.pickle",'rb')
  offsets = pickle.load(filepath_offsets)

  filepath_durations = open(ROOT_PATH_DATA + "/durations.pickle",'rb')
  durations = pickle.load(filepath_durations)

  filepath_velocities = open(ROOT_PATH_DATA + "/velocities.pickle",'rb')
  velocities = pickle.load(filepath_velocities)

  filepath_tempos = open(ROOT_PATH_DATA + "/tempos.pickle",'rb')
  tempos = pickle.load(filepath_tempos)

  return notes,offsets,durations,velocities,tempos

def load_model_from_local(PATH_MODEL):
  model = keras.models.load_model(PATH_MODEL)
  return model

def generate_music(
model, 
network_input_notes,network_input_offsets,network_input_durations,network_input_velocities,network_input_tempos,
notenames, offsetnames, durationames,velocitynames, temponames,
n_vocab_notes, n_vocab_offsets, n_vocab_durations,n_vocab_velocities,n_vocab_tempos
):

  """ Generazione di note dalla rete neurale su una sequenza di note """

  # stabilimento di una sequenza casuale dall'input come punto di partenza per la predizione
  start = numpy.random.randint(0, len(network_input_notes)-1)

  # creo un dizionario dei dati nella quale ogni elemento che sia nota che offset che durata sia identificato come una coppia(chiave,valore)
  int_to_note = dict((number, note) for number, note in enumerate(notenames))
  int_to_offset = dict((number, offset) for number, offset in enumerate(offsetnames))
  int_to_duration = dict((number, duration) for number, duration in enumerate(durationames))
  int_to_velocity = dict((number, velocity) for number,velocity  in enumerate(velocitynames))
  int_to_tempo = dict((number, tempo) for number,tempo in enumerate(temponames))

  # inizio la formulazione dei pattern indicando uno degli elementi per ogni lista
  pattern = network_input_notes[start].flatten().tolist()
  pattern2 = network_input_offsets[start].flatten().tolist()
  pattern3 = network_input_durations[start].flatten().tolist()
  pattern4 = network_input_velocities[start].flatten().tolist()
  pattern5 = network_input_tempos[start].flatten().tolist()


  prediction_output = []

  # generazione di note o accordi
  for i in range(300):
    sequence_lenght = len(pattern)

    note_prediction_input = numpy.reshape(pattern, (1, sequence_lenght, 1))
    predictedNote = note_prediction_input[-1][-1][-1]

    offset_prediction_input = numpy.reshape(pattern2, (1, sequence_lenght, 1))

    duration_prediction_input = numpy.reshape(pattern3, (1, sequence_lenght, 1))

    tempo_prediction_input = numpy.reshape(pattern5, (1, sequence_lenght, 1))

    velocity_prediction_input = numpy.reshape(pattern4, (1, sequence_lenght, 1))




    input_prediction = [note_prediction_input, offset_prediction_input, duration_prediction_input,velocity_prediction_input,tempo_prediction_input]

    #print(note_prediction_input)
    #predizione di note, offset e durate
    prediction = model.predict(input_prediction, verbose=0)

    note_index = numpy.argmax(prediction[0])
    note_result = int_to_note[note_index]

    offset = numpy.argmax(prediction[2])
    offset_result = int_to_offset[offset]

    duration = numpy.argmax(prediction[1])
    duration_result = int_to_duration[duration]

    velocity = numpy.argmax(prediction[3])
    velocity_result = int_to_velocity[velocity]

    tempo = numpy.argmax(prediction[4])
    tempo_result = int_to_tempo[tempo]

    """
    print("probability note: " + str(prediction[0][note_index]) 
    + " - probability duration: " + str(prediction[1][duration]) 
    + " - probability Offset: " + str(prediction[2][offset]))
    """
    print("Next note: " + str(int_to_note[note_index]) 
    + " - Duration: " + str(int_to_duration[duration]) 
    + " - Offset: " + str(int_to_offset[offset])
    + " - velocity: " + str(int_to_velocity[velocity]) 
    + " - tempo: " + str(int_to_tempo[tempo]))

    prediction_output.append([note_result, offset_result, duration_result,velocity_result,tempo_result])

    pattern.append(note_index/n_vocab_notes)
    pattern2.append(offset/n_vocab_offsets)
    pattern3.append(duration/n_vocab_durations)
    pattern4.append(offset/n_vocab_offsets)
    pattern5.append(duration/n_vocab_durations)
    
    pattern = pattern[1:len(pattern)]
    pattern2 = pattern2[1:len(pattern2)]
    pattern3 = pattern3[1:len(pattern3)]
    pattern4 = pattern4[1:len(pattern4)]
    pattern5 = pattern5[1:len(pattern5)]

  return prediction_output

def from_feature_to_info_dict(feature_list):
  
  featurnames = sorted(set(item for item in feature_list))
  n_vocab_feature = len(set(feature_list))
  
  return featurnames,n_vocab_feature

def create_csv_musical_content(output_song_info,num_song):
  df = pd.DataFrame(columns=["note_name", "offset", "duration", "velocity", "tempo"])

  for el in output_song_info:
    new_df = pd.DataFrame([[ el[0], el[1], el[2], el[3], el[4] ]], columns=["note_name", "offset", "duration", "velocity", "tempo"])
    df = df.append(new_df, ignore_index=True) 
  
  df.to_csv("test_output"+ str(num_song) +".csv")

def generate(num_bra,ROOT_PATH_DATA,PATH_MODEL):
  print("--------- BRANO" + str(num_bra) + "--------------")

  network_input_notes,network_input_offsets,network_input_durations,network_input_velocities,network_input_tempos = load_test_sequences(ROOT_PATH_DATA)

  notes,offsets,durations,velocities,tempos = load_features(ROOT_PATH_DATA)

  notenames,n_vocab_notes = from_feature_to_info_dict(notes)
  offsetnames,n_vocab_offsets = from_feature_to_info_dict(offsets)
  durationnames,n_vocab_durations = from_feature_to_info_dict(durations)
  velocitynames,n_vocab_velocities = from_feature_to_info_dict(velocities)
  temponames,n_vocab_tempos = from_feature_to_info_dict(tempos)

  print ("load model...")

  model = load_model_from_local(PATH_MODEL)

  print ("finish")

  #attraverso le predizioni del modello è possibile generare le note con le corrispettive durate e offsets

  prediction_output = generate_music(
  model, 
  network_input_notes,network_input_offsets,network_input_durations,network_input_velocities,network_input_tempos, 
  notenames, offsetnames, durationnames,velocitynames, temponames,
  n_vocab_notes, n_vocab_offsets, n_vocab_durations,n_vocab_velocities,n_vocab_tempos
  )

  create_csv_musical_content(prediction_output,num_bra)

#------- MAIN -------

N_BRANI_DA_GENERARE = 10
ROOT_PATH_DATA_PROCESSED ="../../data/processed/dataset_pickle"
PATH_MODEL = "../../models/model"

for i in range(0,N_BRANI_DA_GENERARE):
  generate(i,ROOT_PATH_DATA,PATH_MODEL)