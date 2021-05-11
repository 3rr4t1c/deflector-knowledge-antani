# Script per inizializzare una ground truth 
# da etichettare a mano per la valutazione
# NON ESEGUIRE se è già presente una ground truth,
# tutti i dati saranno sovrascritti
import pandas as pd
import os

answ = input('WARNING: This will overwirte pre-existing ground truth in this folder, continue? (yes/no) > ')

if 'y' in answ.lower():
    pass
else:
    exit() 

ground_truth = pd.read_csv('..\output_data\pattern2relation.tsv', sep='\t', keep_default_na=False)
ground_truth = ground_truth.sort_values('relation')
ground_truth['isCorrect'] = 2
ground_truth.to_csv('ground_truth.tsv', sep='\t', index=False)