from sklearn.preprocessing import minmax_scale
from tools import read_tsv
import numpy as np


class REWrapper:

    def __init__(self, model):
        self.model = model

    # Relation extraction model wrapper
    def extract_relations(self, pattern_triples):

        if type(pattern_triples) is str:
            re_input = read_tsv(pattern_triples)
        else:
            re_input = pattern_triples

        re_output = self.model.harvest(re_input, keep_unknown=True)

        # Costruisce l'output adatto per deflector
        output = list()
        for i, pattern_triple in enumerate(re_input):
            phr, s, st, o, ot = pattern_triple
            relation = re_output[i][1]
            pattern = (st, phr, ot)
            pair = (s, o)
            output.append((pattern, relation, pair))

        return output



class LPWrapper:

    def __init__(self, model):
        self.model = model

    # The ComplEx model wrapper
    def predict_links(self, list_of_pairs):        
        
        # Structures to handle possible unknown entities
        pair2complex = dict()
        pair2unknown = dict()
        for pair in list_of_pairs:
            try:        
                h_id = self.model.dataset.get_id_for_entity_name(pair[0])
                r_id = 0  # Relation id will be ignored in predictions
                t_id = self.model.dataset.get_id_for_entity_name(pair[1])        
                pair2complex[pair] = (h_id, r_id, t_id)
            except:
                pair2unknown[pair] = set(['unknown'])

        # Build ComplEx input
        complex_input = list(pair2complex.values())
        
        # Compute all scores (output is a 3D tensor with dim (npairs, nrelations, 1))
        all_scores = self.model.all_scores_relations(complex_input)[:, :, 0]

        # Normalize scores to range [0, 1]
        minmax_scale(all_scores, axis=1, copy=False)

        # Array 2d con le posizioni [[riga, colonna]] dei valori che superano la soglia
        best_rels = np.argwhere(all_scores > 0.9) 
            
        # Build wrapper output
        complex_output = [set() for _ in complex_input]
        for row in best_rels:
            rel = self.model.dataset.get_name_for_relation_id(row[1])
            # Exclude inverse relations
            if 'INVERSE' not in rel:
                complex_output[row[0]].add(rel)

        # Re-associate pairs to complex outputs
        pair2complex = dict(zip(pair2complex.keys(), complex_output))

        # Build wrapper output
        output = list()
        for pair in list_of_pairs:
            try:
                output.append(pair2complex[pair])
            except:
                output.append(pair2unknown[pair])

        # Add unknown prediction to empty sets
        for pred in output:
            if not pred:
                pred.add('unknown')
        # Empty sets may occurr when min(x) == max(x) in minMax scaling

        return output



class LPWrapperMaxScore:

    def __init__(self, model):
        self.model = model

    # The set of simmetric relations (set is used for faster lookup)
    sim_rel = set(['spouse']) # TODO: insert simmetric relations

    # The ComplEx model wrapper
    def predict_links(self, list_of_pairs):        
        
        # Structures to handle possible unknown entities
        pair2complex = dict()
        pair2unknown = dict()
        for pair in list_of_pairs:
            try:        
                h_id = self.model.dataset.get_id_for_entity_name(pair[0])
                r_id = 0  # Relation id will be ignored in predictions
                t_id = self.model.dataset.get_id_for_entity_name(pair[1])        
                pair2complex[pair] = (h_id, r_id, t_id)
            except:
                pair2unknown[pair] = set(['unknown'])

        # Build ComplEx input
        complex_input = list(pair2complex.values())
        
        # Compute all scores (output is a 3D tensor with dim (npairs, nrelations, 1))
        all_scores = self.model.all_scores_relations(complex_input)[:, :, 0]

        # Get top k max score indexes
        k = 5 #all_scores.shape[1] // 10
        topk_argmax = np.argsort(all_scores, axis=1)[:, -k:]

        # Build wrapper output
        complex_output = [set() for _ in complex_input]
        for n, topk in enumerate(topk_argmax):
            for relID in topk:
                rel_name = self.model.dataset.get_name_for_relation_id(relID)
                # Exclude inverse relations
                if 'INVERSE_' not in rel_name:
                    complex_output[n].add(rel_name)
                else:
                    rel_name_inv = rel_name.replace('INVERSE_', '')
                    if rel_name_inv in self.sim_rel:
                        complex_output[n].add(rel_name_inv)


        # Re-associate pairs to complex outputs
        pair2complex = dict(zip(pair2complex.keys(), complex_output))

        # Build wrapper output
        output = list()
        for pair in list_of_pairs:
            try:
                output.append(pair2complex[pair])
            except:
                output.append(pair2unknown[pair])

        # Add unknown prediction to empty sets
        for pred in output:
            if not pred:
                pred.add('unknown')
        # Empty sets may occurr when all top k relation are INVERSE and not simmetric

        return output