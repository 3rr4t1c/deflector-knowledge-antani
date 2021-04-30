# Script per inizializzare una ground truth da etichettare poi a mano

ground_truth = pd.read_csv('pattern2relation.tsv', sep='\t', keep_default_na=False)

ground_truth = ground_truth.sort_values('relation')
ground_truth['isCorrect'] = 2

ground_truth.to_csv('output_data/ground_truth.tsv', sep='\t', index=False)