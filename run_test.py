from LectorPlusPkg.selector import SELector
from ComplExPkg.complexModel import ComplEx, Dataset
from wrappers import REWrapper, LPWrapper
from deflector import Deflector
from tools import saveRecords, custom_stats
import pandas as pd
import torch


# Input: 
# (1) patterns (feature vectors) + pairs (subject, object)
pattern_triples_file = 'input_data/all_patterns_6.2M.tsv'
# (2) knowledge graph (used for training link prediction too)
knowledge_graph_file = 'input_data/knowledge_graph_V2/train.txt'


# Relation Extractor (LectorPlus) Training:
# Distant Supervision: get Unlabeled Patterns (SAVE TSV FILE) n.b. l'etichetta unknown non serve
# LectorPlus will learn (pattern -> relation) associations
lector_model = SELector(rseed=42, unlabeled_sub=0)
lector_model.train(pattern_triples_file, knowledge_graph_file, auto_clean=False)
deflector_input = [x[:-1] for x in lector_model.all_unlabeled_triples] # Remove unknown label 
lector_model.clean()


# Checkpoint 1: save on disk "output_data\deflector_input.tsv"
unlabeled_cnames = ['subject', 'subject_type', 'phrase', 'object_type', 'object']
unlabeled = [(x[1], x[2], x[0], x[4], x[3]) for x in deflector_input]
saveRecords(unlabeled, file_name='output_data/unlabeled.tsv', sep='\t', header=True, cnames=unlabeled_cnames)

# [print(x) for x in deflector_input[:10]] # DEBUG
# ('department of', 'Granada_Department', '[AdministrativeRegion]', 'Nicaragua', '[Country]', 'unknown')
# ('the closure of the', 'South_Africa', '[Country]', 'Reivilo', '[Settlement]', 'unknown')


# Link Prediction (ComplEx) Pre-trained model loading:
complex_saved_model = 'ComplExPkg\stored_models\ComplEx_V2_d500_bs1000.pt'
complex_hyperpar = {'dimension': 500, 'init_scale': 1e-3}
complex_dataset = 'input_data/knowledge_graph_V2'
print('\nCaricamento di ComplEx in corso...', end='')
complex_model = ComplEx(Dataset(name='CUSTOM', home=complex_dataset), complex_hyperpar)
complex_model.to('cuda')
complex_model.load_state_dict(torch.load(complex_saved_model))
complex_model.eval()
print('Fatto.')


# Wrappers
lector_wrapper = REWrapper(lector_model)
complex_wrapper = LPWrapper(complex_model)


# Deflector
deflector = Deflector(lector_wrapper, complex_wrapper)
deflector.deflect_patterns(deflector_input, bl_min_len=1)


# Save pattern to relations associations
pattern2relation_cnames = ['subject_type', 'phrase', 'object_type', 'relation']
pattern2relation = [(x[0][0], x[0][1], x[0][2], x[1]) for x in deflector.pattern2relation]
saveRecords(pattern2relation, 
            file_name='output_data/pattern2relation.tsv', 
            sep='\t', header=True, 
            cnames=pattern2relation_cnames)


# Save blacklisted patterns
banned_cnames = ['subject_type', 'phrase', 'object_type', 'relation']
banned_patterns = [(x[0][0], x[0][1], x[0][2], x[1][0]) for x in deflector.pattern_black_list.items()]
saveRecords(banned_patterns, 
            file_name='output_data/banned_patterns.tsv', 
            sep='\t', header=True, 
            cnames=banned_cnames)


# Print a log
pd.set_option("display.min_rows", 101)

print('\n**UNLABELED**')
unlabeled_df = pd.read_csv('output_data/unlabeled.tsv', sep='\t')
print(unlabeled_df)

print('\n**PATTERN TO RELATION ASSOCIATIONS**')
pattern2relation_df = pd.read_csv('output_data/pattern2relation.tsv', sep='\t')
print(pattern2relation_df)
custom_stats(pattern2relation_df, max_rows=400, col_names=['relation'])

print('\n**BANNED PATTERNS**')
banned_patterns_df = pd.read_csv('output_data/banned_patterns.tsv', sep='\t')
print(banned_patterns_df)
custom_stats(banned_patterns_df, max_rows=400, col_names=['relation'])



# print(f'Pattern rilevati in totale: {len(deflector.pattern2relation)}')
# print(f'Esempio:')
# [print(x) for x in deflector.pattern2relation[:10]]

# print(f'\nPattern inseriti nella blacklist: {len(deflector.pattern_black_list.items())}')
# print(f'Esempio:')
# [print(x) for x in deflector.pattern_black_list.items() if x[1][0] == 'spouse']




