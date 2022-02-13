# Music-Generation-LSTM
Lavoro di tesi sulla generazione musicale con approccio LSTM

# Descrizione del contenuto
#### struttura della source directory

![](albero_sorgente.png)

NOTA -> per una comprensione dei sorgenti di src di:

- generation
- models
- visualization

consultare i file colab contenuti nella directory colab_notebooks
# Istruzioni di installazione package
### Invoca il comandi
```
source MSL/Scripts/activate
pip install -r requirements.txt
```
# Istruzioni creazione dataset
### Una volta collezionato il set di brani midi invocare in src/data
```
python midi_to_csv.py nome_directory_midi_raccolti
python create_dataset.py 
```
il secondo comando dovrà essere eseguito dopo la modifica della funzione nel sorgente di create_dataset("../../data/interim/dataset_csv.csv","../../data/processed/dataset_pickle.pickle")

# Istruzioni conversione file da midi a csv dopo la generazione
### Una volta collezionato il set di brani in formato csv generati dal modello invocare in src/feature
```
python midi_to_csv_table_features.py nome_directory_csv_generati
```
Per ascoltare i brani midi generati ho eseguito una conversione da midi a mp3 con l'impiego di tool online
