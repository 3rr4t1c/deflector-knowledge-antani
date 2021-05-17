from tqdm import tqdm
import numpy as np


# Deflector è un meta-sistema per la relation extraction
# che può sfruttare un qualsiasi sistema di link prediction
# per migliorare un qualunque sistema di relaction extraction 
class Deflector:

    # Inizializzazione: 
    # re_wrapper: è una funzione che può essere definita arbitrariamente
    # al suo interno, riceve come input lo stesso input del modello di RE
    # ma deve restituire l'input formattato come lista di triple: [(e1, pattern0, e2)] 
    # e l'output come lista di relazioni: [(pattern0, relation0, (e1, e2))] entrambe della stessa dimensione
    # lp_wrapper_fun: è una funzione che può essere definita arbitrariamente
    # al suo interno ma deve prendere come input una lista di istanze [(e1, e2)]
    # e deve restituire [{set of top relations for e1 e2}]
    # PRE: le dimensioni di input ed output devono essere identiche ed associare alle
    # le relazioni che non possono essere estratte o predette la stringa 'unknown'  
    def __init__(self, re_wrapper, lp_wrapper):
        self.relation_extraction_model = re_wrapper
        self.link_prediction_model = lp_wrapper
        self.pattern_black_list = dict() # {pattern: (old_relation, score)}
        self.pattern_discovery = dict()  # {pattern: (new_relation, score)} # solo per ex relation unknown


    # Applica i filtri sull'output invece che sull'input
    def extract_relations(self, re_input, keep_unknown=False, keep_patterns=False, keep_pairs=False):
        
        # Input ed Output formattati del modello relation extraction
        re_output = self.relation_extraction_model.extract_relations(re_input)

        # Filtra l'output di relation extraction 
        output = list()
        for pattern, relation, pair in re_output:
            sbj, obj = pair
            if relation != 'unknown':
                try:
                    self.pattern_black_list[pattern]
                    output.append((pattern, (sbj, 'unknown', obj)))
                except:
                    output.append((pattern, (sbj, relation, obj)))
            else:
                try:
                    new_rel = self.pattern_discovery[pattern][0]
                    output.append((pattern, (sbj, new_rel, obj)))
                except:
                    output.append((pattern, (sbj, relation, obj)))

        # Se si vogliono escludere fatti con relazione 'unknown'
        if not keep_unknown:
            output = [x for x in output if x[1][1] != 'unknown']

        # Se non si vogliono tenere le coppie anche nell'output
        if not keep_pairs:
            output = [(x[0], x[1][1]) for x in output]

        # Non tiene traccia del pattern che ha predetto il fatto
        if not keep_patterns:
            output = [x[1] for x in output]            

        return output


    # Da dizionario: {pattern: (relation, [(subject, object), ...]), ...}
    # A dizionario: {(pair): {set predizioni}}
    def batch_predict(self, pattern2relpairs, link_predictor, batch_size):

        # Init output structure
        all_pairs = set()
        print('\nFinding all unique pairs to be predicted...')
        for _, pairs in tqdm(pattern2relpairs.values()):
            for pair in pairs:
                all_pairs.add(pair)

        # Batch predict
        all_pairs = list(all_pairs)
        result = list()
        print('\nBatch predicting relations...')
        for i in tqdm(range(0, len(all_pairs), batch_size)):
            result += zip(all_pairs[i:i+batch_size], link_predictor.predict_links(all_pairs[i:i+batch_size]))

        return dict(result)


    # Identifica pattern deboli e li aggiunge alla black list
    def deflect_patterns(self, re_input, bs=5000, bl_min_len=1, bl_thresh=0.7, pd_min_len=10, pd_thresh=0.7, pd_enabled=False):

        # Traccia le relazioni estratte
        ex_trace = self.extract_relations(re_input, keep_unknown=True, keep_patterns=True, keep_pairs=True)

        # Costruisce un dizionario {pattern: (relation, [(subject, object), ...]), ...} (in altre parole: clustering)
        pattern2instances = dict()
        for pattern, opinion in ex_trace:
            try:
                sbj, _, obj = opinion
                pattern2instances[pattern][1].append((sbj, obj))
            except:
                sbj, relation, obj = opinion 
                pattern2instances[pattern] = (relation, [(sbj, obj)])

        # Mantiene una struttura con tutti i pattern noti e le relazioni associate
        self.pattern2relation = [(x[0], x[1][0]) for x in pattern2instances.items() if x[1][0] != 'unknown']

        # Build dictionary {pair: predicted relations as a set}
        pair2pred = self.batch_predict(pattern2instances, self.link_prediction_model, batch_size=bs)
        
        self.get_statistics(pattern2instances, pair2pred) # New: this will collect some statistics about clusters

        print('\nDeflecting patterns...')
        for pattern, track in tqdm(pattern2instances.items()):
            relation, pairs = track            
            predictions = [pair2pred[pair] for pair in pairs if 'unknown' not in pair2pred[pair]]
            preds_len = len(predictions)
            if relation != 'unknown' and preds_len >= bl_min_len:
                matches = [relation in pred for pred in predictions]                
                pattern_score = sum(matches)/len(matches)
                # Pattern blacklist
                if pattern_score < bl_thresh:                    
                    self.pattern_black_list[pattern] = (relation, pattern_score)            
            elif pd_enabled and preds_len >= pd_min_len:
                all_pred_size = [len(pred) for pred in predictions]
                rel2weight = dict()
                for n, preds in enumerate(predictions):
                    for rel in preds:
                        try:
                            rel2weight[rel] += 1/all_pred_size[n]
                        except:
                            rel2weight[rel] = 1/all_pred_size[n]
                
                max_rel1 = max(rel2weight, key=rel2weight.get)
                max_wgh1 = rel2weight[max_rel1]
                del(rel2weight[max_rel1])
                try:                        
                    max_rel2 = max(rel2weight, key=rel2weight.get)
                    max_wgh2 = rel2weight[max_rel2]
                except:
                    max_wgh2 = 0

                max_rel_score = (max_wgh1 - max_wgh2)/len(pairs)
                # Pattern discovery
                if max_rel_score >= pd_thresh:
                    self.pattern_discovery[pattern] = (max_rel1, max_rel_score)


    # Calcola alcune statistiche sui cluster di coppie associate ad un pattern
    def get_statistics(self, pattern2track, pair2pred):
        
        self.statistics = {'total_clusters': 0,
                           'cluster_sizes': [],
                           'unknown_sizes': [],
                           'known_sizes': [],
                           'unknown_perc': []}

        # Itera per ciascun cluster di coppie relativa ad una 
        # associazione pattern-relazione appresa da Lector
        for _, track in pattern2track.items():
            relation, pairs = track
            
            # Salta i pattern associati a nessuna relazione
            if relation == 'unknown':
                continue 

            predictions = [pair2pred[pair] for pair in pairs]
            cluster_size = len(predictions)
            unknown_size = len([p for p in predictions if 'unknown' in p])

            self.statistics['total_clusters'] += 1
            self.statistics['cluster_sizes'].append(cluster_size)
            self.statistics['unknown_sizes'].append(unknown_size)
            self.statistics['known_sizes'].append(cluster_size - unknown_size)
            self.statistics['unknown_perc'].append(unknown_size/cluster_size)
            

        # Aggrega risultati

        # Statistiche sulle dimensioni dei cluster
        for name in ['cluster', 'unknown', 'known']:
            self.statistics[f'mean_{name}_size'] = np.mean(self.statistics[f'{name}_sizes'])
            self.statistics[f'max_{name}_size'] = np.max(self.statistics[f'{name}_sizes'])
            self.statistics[f'min_{name}_size'] = np.min(self.statistics[f'{name}_sizes'])
            self.statistics[f'std_{name}_size'] = np.std(self.statistics[f'{name}_sizes'])
            # Percentuale media e deviazione standard del contenuto di 'unknown' prediction in un cluster
            if name == 'unknown':
                self.statistics[f'mean_{name}_perc'] = np.mean(self.statistics[f'{name}_perc'])
                self.statistics[f'std_{name}_perc'] = np.std(self.statistics[f'{name}_perc'])

        # Cluster contenenti solo unknown (LP non conosce almeno una delle due entità in predizione)
        self.statistics['unknown_clusters'] = 0
        for n, size in enumerate(self.statistics['cluster_sizes']):
            if size == self.statistics['unknown_sizes'][n]:
                self.statistics['unknown_clusters'] += 1

        # Rimozione chiavi associate a liste di valori
        del(self.statistics['cluster_sizes'])
        del(self.statistics['unknown_sizes'])
        del(self.statistics['known_sizes'])
        del(self.statistics['unknown_perc'])
            

        

            
