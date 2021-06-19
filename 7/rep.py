import numpy as np
import pandas as pd

alldata = pd.read_csv(
    'flats.tsv', header=0, sep='\t',
     usecols=['cena', 'Powierzchnia w m2', 'Liczba pokoi', 'Liczba pięter w budynku', 'Piętro', 'Typ zabudowy', 'Materiał budynku', 'Rok budowy', 'opis', 'Forma kuchni'])

alldata = alldata.dropna()
alldata['Piętro'] = alldata['Piętro'].apply(lambda x: 0 if x == 'parter' else x)
# alldata['Piętro'] = alldata['Piętro'].apply(pd.to_numeric, errors='coerce')
print(alldata.head())