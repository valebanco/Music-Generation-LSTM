import os
import glob
import csv
import pandas as pd
import glob
import pickle
import numpy
from keras.utils import np_utils
import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.model_selection import train_test_split 

"""
________ CONVERSIONE DA CSV A PICKLE CON RELATIVO PRE-PROCESSING________

"""

def build_dictionary (data_list):
    # get all pitch names
    featurenames = sorted(set(item for item in data_list))

    # create a dictionary to map pitches to integers
    feature_to_int = dict((note, number) for number, note in enumerate(featurenames))

    return feature_to_int

def prepare_sequences(current_feature_list, n_vocab):
    sequence_length = 16

	# get all pitch names
    featurenames = sorted(set(item for item in current_feature_list))

    # create a dictionary to map pitches to integers
    feature_to_int = dict((note, number) for number, note in enumerate(featurenames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(current_feature_list) - sequence_length, 1):
        sequence_in = current_feature_list[i:i + sequence_length]
        sequence_out = current_feature_list[i + sequence_length]
        network_input.append([feature_to_int[char] for char in sequence_in])
        network_output.append(feature_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def vect_features(f1,f2,f3,f4,f5):
    new_arr = []
    for i in range(f1.shape[0]):
        new_arr.append([f1[i],f2[i],f3[i],f4[i],f5[i]])
    
    res = numpy.array(new_arr)
    return res

def devect_features(vect):
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    v5 = []

    for i in range(vect.shape[0]):
        v1.append(vect[i][0])
        v2.append(vect[i][1])
        v3.append(vect[i][2])
        v4.append(vect[i][3])
        v5.append(vect[i][4])
    
    res = [numpy.array(v1),numpy.array(v2),numpy.array(v3),numpy.array(v4),numpy.array(v5)]
    return res
def save_set (set,name_set,path):
    filepath1 = open(path + "/" + name_set, 'wb')
    pickle.dump(set, filepath1,protocol=pickle.HIGHEST_PROTOCOL)

def extract_sequences(
    data_list_note_name,data_list_offset,data_list_duration,data_list_velocity,data_list_tempo,
    path_pickle_destination
    ):
    print("prepare sequences...")

    n_vocab_notes = len(set(data_list_note_name))
    in_notes,out_notes = prepare_sequences(data_list_note_name,n_vocab_notes)

    print("prepare sequece notes finish...")

    n_vocab_offsets = len(set(data_list_offset))
    in_offsets,out_offsets = prepare_sequences(data_list_offset,n_vocab_offsets)
    
    print("prepare sequece offset finish...")

    n_vocab_durations = len(set(data_list_duration))
    in_durations,out_durations = prepare_sequences(data_list_duration,n_vocab_durations)
    
    print("prepare sequece durations finish...")

    n_vocab_velocities = len(set(data_list_velocity))
    in_velocities,out_velocities = prepare_sequences(data_list_velocity,n_vocab_velocities)
    
    print("prepare sequece velocities finish...")
    
    n_vocab_tempos = len(set(data_list_tempo))
    in_tempos,out_tempos = prepare_sequences(data_list_tempo,n_vocab_tempos)
    
    print("prepare sequece tempos finish...")

    X = vect_features(in_notes,in_durations,in_offsets,in_velocities,in_tempos) 
    y = vect_features(out_notes,out_durations,out_offsets,out_velocities,out_tempos)
    
    print("split dataset...")

    X_t, X_test,y_t,y_test = train_test_split(X,y,test_size = 0.1,random_state = 42)

    X_train, X_validation,y_train,y_validation = train_test_split(X_t,y_t,test_size = 0.3,random_state = 42)

    res_X_train = devect_features(X_train)
    res_y_train = devect_features(y_train)
    res_X_validation = devect_features(X_validation)
    res_y_validation = devect_features(y_validation)
    res_X_test = devect_features(X_test)
    res_y_test = devect_features(y_test)

    print("Saving sets...")

    save_set(data_list_note_name,'notes.pickle',path_pickle_destination)
    save_set(data_list_offset,'offsets.pickle',path_pickle_destination)
    save_set(data_list_duration,'durations.pickle',path_pickle_destination)
    save_set(data_list_velocity,'velocities.pickle',path_pickle_destination)
    save_set(data_list_tempo,'tempos.pickle',path_pickle_destination)

    save_set(res_X_train,'X_train.pickle',path_pickle_destination)
    save_set(res_y_train,'y_train.pickle',path_pickle_destination)
    save_set(res_X_validation,'X_validation.pickle',path_pickle_destination)
    save_set(res_y_validation,'y_validation.pickle',path_pickle_destination)
    save_set(res_X_test,'X_test.pickle',path_pickle_destination)
    save_set(res_y_test,'y_test.pickle',path_pickle_destination)

    print("Sets saved!!!!")

def create_dataset(path_csv_source,path_pickle_destination):
    data_list_note_name = []
    data_list_offset = []
    data_list_duration = []
    data_list_velocity = []
    data_list_tempo = []
    
    print("read features...")
    
    for file in glob.glob(path_csv_source + "/*.csv"):

        print("Parsing %s" % file)
        
        csv_file = pd.read_csv(file)
        df = pd.DataFrame(csv_file)
        list_data = df.values.tolist()

        for el in list_data:
            data_list_note_name.append(str(el[1]))
            data_list_offset.append(str(el[2]))
            data_list_duration.append(str(el[3]))
            data_list_velocity.append(str(el[4]))
            data_list_tempo.append(str(el[5]))    

    print("feature read complete!")
    
    extract_sequences(data_list_note_name,data_list_offset,data_list_duration,data_list_velocity,data_list_tempo,path_pickle_destination)


create_dataset("../../data/interim/dataset_csv","../../data/processed/dataset_pickle")