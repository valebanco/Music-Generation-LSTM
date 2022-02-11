
# converts all midi files in the current folder

import glob
import os
import music21

diesis_to_bemolle = {
    'C#':'D-',
    'D#':'E-',
    'F#':'G-',
    'G#':'A-',
    'A#':'B-',
    'D-':'D-',
    'E-':'E-',
    'G-':'G-',
    'A-':'A-',
    'B-':'B-',
    'A':'A',
    'B':'B',
    'C':'C',
    'D':'D',
    'E':'E',
    'F':'F',
    'G':'G',
}

# every semitones between -5~0~6
majors = {
    'C': {'A-': 4, 'A': 3, 'B-': 2, 'B': 1, 'C': 0, 'D-': -1, 'D': -2, 'E-': -3, 'E': -4, 'F': -5, 'G-': 6, 'G': 5},
    'D-': {'A-':5, 'A': 4, 'B-': 3, 'B': 2, 'C': 1, 'D-': 0, 'D': -1, 'E-': -2, 'E': -3, 'F': -4, 'G-': -5, 'G': 6},
    'D': {'A-': 6, 'A': 5, 'B-': 4, 'B': 3, 'C': 2, 'D-': 1, 'D': 0, 'E-': -1, 'E': -2, 'F': -3, 'G-': -4, 'G': -5},
    'E-': {'A-': -5, 'A': 6, 'B-': 5, 'B': 4, 'C': 3, 'D-': 2, 'D': 1, 'E-': 0, 'E': -1, 'F': -2, 'G-':-3, 'G': -4},
    'E': {'A-': -4, 'A': -5, 'B-': 6, 'B': 5, 'C': 4, 'D-': 3, 'D': 2, 'E-': 1, 'E': 0, 'F': -1, 'G-': -2, 'G': -3},
    'F': {'A-': -3, 'A': -4, 'B-': -5, 'B': 6, 'C': 5, 'D-': 4, 'D': 3, 'E-': 2, 'E': 1, 'F': 0, 'G-': -1, 'G': -2},
    'G-': {'A-': -2, 'A': -3, 'B-': -4, 'B': -5, 'C': 6, 'D-': 5, 'D': 4, 'E-': 3, 'E': 2, 'F': 1, 'G-': 0, 'G': -1},
    'G': {'A-': -1, 'A': -2, 'B-': -3, 'B': -4, 'C': -5, 'D-': 6, 'D': 5, 'E-': 4, 'E': 3, 'F': 2, 'G-': 1, 'G': 0},
    'A-': {'A-': 0, 'A': -1, 'B-': -2, 'B': -3, 'C': -4, 'D-': -5, 'D': 6, 'E-': 5, 'E': 4, 'F': 3, 'G-': 2, 'G': 1},
    'A': {'A-': 1, 'A': 0, 'B-': -1, 'B': -2, 'C': -3, 'D-': -4, 'D': -5, 'E-': 6, 'E': 5, 'F': 4, 'G-': 3, 'G': 2},
    'B-': {'A-': 2, 'A': 1, 'B-': 0, 'B': -1, 'C': -2, 'D-': -3, 'D': -4, 'E-': -5, 'E': 6, 'F': 5, 'G-': 4, 'G': 3},
    'B': {'A-': 3, 'A': 2, 'B-': 1, 'B': 0, 'C': -1, 'D-': -2, 'D': -3, 'E-': -4, 'E': -5, 'F': 6, 'G-': 5, 'G': 4},
}


# same method in minors
# A minors
minors = {
    'C': {'A-': 4, 'A': 3, 'B-': 2, 'B': 1, 'C': 0, 'D-': -1, 'D': -2, 'E-': -3, 'E': -4, 'F': -5, 'G-': 6, 'G': 5},
    'D-': {'A-':5, 'A': 4, 'B-': 3, 'B': 2, 'C': 1, 'D-': 0, 'D': -1, 'E-': -2, 'E': -3, 'F': -4, 'G-': -5, 'G': 6},
    'D': {'A-': 6, 'A': 5, 'B-': 4, 'B': 3, 'C': 2, 'D-': 1, 'D': 0, 'E-': -1, 'E': -2, 'F': -3, 'G-': -4, 'G': -5},
    'E-': {'A-': -5, 'A': 6, 'B-': 5, 'B': 4, 'C': 3, 'D-': 2, 'D': 1, 'E-': 0, 'E': -1, 'F': -2, 'G-':-3, 'G': -4},
    'E': {'A-': -4, 'A': -5, 'B-': 6, 'B': 5, 'C': 4, 'D-': 3, 'D': 2, 'E-': 1, 'E': 0, 'F': -1, 'G-': -2, 'G': -3},
    'F': {'A-': -3, 'A': -4, 'B-': -5, 'B': 6, 'C': 5, 'D-': 4, 'D': 3, 'E-': 2, 'E': 1, 'F': 0, 'G-': -1, 'G': -2},
    'G-': {'A-': -2, 'A': -3, 'B-': -4, 'B': -5, 'C': 6, 'D-': 5, 'D': 4, 'E-': 3, 'E': 2, 'F': 1, 'G-': 0, 'G': -1},
    'G': {'A-': -1, 'A': -2, 'B-': -3, 'B': -4, 'C': -5, 'D-': 6, 'D': 5, 'E-': 4, 'E': 3, 'F': 2, 'G-': 1, 'G': 0},
    'A-': {'A-': 0, 'A': -1, 'B-': -2, 'B': -3, 'C': -4, 'D-': -5, 'D': 6, 'E-': 5, 'E': 4, 'F': 3, 'G-': 2, 'G': 1},
    'A': {'A-': 1, 'A': 0, 'B-': -1, 'B': -2, 'C': -3, 'D-': -4, 'D': -5, 'E-': 6, 'E': 5, 'F': 4, 'G-': 3, 'G': 2},
    'B-': {'A-': 2, 'A': 1, 'B-': 0, 'B': -1, 'C': -2, 'D-': -3, 'D': -4, 'E-': -5, 'E': 6, 'F': 5, 'G-': 4, 'G': 3},
    'B': {'A-': 3, 'A': 2, 'B-': 1, 'B': 0, 'C': -1, 'D-': -2, 'D': -3, 'E-': -4, 'E': -5, 'F': 6, 'G-': 5, 'G': 4},
}
 
def transpose_all_keys(scale_vector,key_original):
    for el in scale_vector:
        if (el != diesis_to_bemolle[key.tonic.name] ):
            halfSteps = scale_vector[el][diesis_to_bemolle[key.tonic.name]]
            newscore = score.transpose(halfSteps)
            key_target = newscore.analyze('key')
            print('Target key: ', key_target.tonic.name, key_target.mode)
            newFileName = file.split("\\")[1] + '_' +key_target.tonic.name + key_target.mode + '.mid'
            newscore.write('midi',newFileName)

def augment_raw_file(path):

    for file in glob.glob(path):
        score = music21.converter.parse(file)
        key = score.analyze('key')
        print("trasposing %s" % file)
        print('Original key: ', key.tonic.name )

        
        #  halfStep: semitone range
        if key.mode == "major":
            transpose_all_keys(majors,key)
            
        elif key.mode == "minor":
        transpose_all_keys(minors,key)

augment_raw_file("../../data/raw/dataset_midi/*.mid")    