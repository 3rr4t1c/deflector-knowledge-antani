import pandas as pd
import csv


# Legge un TSV come lista di tuple
def read_tsv(file_path):
    out = list()
    with open(file_path, 'r', encoding='utf8') as tsv_file:
        rd = csv.reader(tsv_file, delimiter='\t')
        for line in rd:                
            out.append(tuple(line))
    return out





def saveRecords(list_of_tuple, file_name='data.csv', sep=',', header=False, cnames=None):
    df = pd.DataFrame(list_of_tuple, columns=cnames)
    df.to_csv(file_name, sep=sep, index=False, header=header)
 

# Mostra delle statistiche per un DataFrame passato per parametro.
def custom_stats(df, max_rows=6, col_names=None):
    nrows, ncols = df.shape 
    
    print(f'\n >> DataFrame rows: {nrows}, DataFrame columns: {ncols}\n')
    
    target_columns = list(df.columns)
    if col_names:
        target_columns = col_names
        
    for col_name in target_columns:
        col_nunique = df[col_name].nunique()
        col_dtype = df[col_name].dtype
        col_nan = df[col_name].isnull().sum()
        print('************************')
        print(f'Column: "{col_name}" ', end='') 
        print(f'| Uniques: {col_nunique} in {nrows} ', end='') 
        print(f'| Missing: {col_nan} ({int(col_nan / nrows * 100)}%) ', end='') 
        print(f'| Type: {col_dtype}')
        print('\nClasses: \n')
        vcount = df.groupby(col_name).size().sort_values(ascending=False).reset_index()
        vcount['Perc'] = (vcount.iloc[:,1] / nrows * 100).round(3).astype(str) + '%'
        print(vcount.to_string(header=False, index=False, max_rows=max_rows), '\n')