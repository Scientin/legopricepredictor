#Imports
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import re
from dateutil.parser import parse

#Table preprocessing
def tablepreprocessing(table, title, year):
  #Convert table to dataframe
  table_df = pd.read_html(str(table))[0]
  #Turn first row to column headers, then drop first row
  table_df.columns = table_df.iloc[0]
  table_df = table_df.drop(0)
  #Add theme column using title for entries
  table_df['Theme']=title
  #Drop image, set number, and set name columns, also removes invalid tables
  if 'Image' not in table_df.columns:
    return
  else:
    table_df = table_df.drop('Image', axis=1)
  table_df = table_df.drop('#', axis=1)
  table_df = table_df.drop('Set', axis=1)
  #Convert entries with NaN for figures to 0
  table_df['Figures'] = table_df['Figures'].fillna(str(0))
  #Convert named figures into int total
  figurecount=1
  for i in table_df['Figures']:
    a = [int(s) for s in re.findall(r'\b\d+\b(?![\s,a-zA-Z])', i)]
    if i.count(",") == 0 and re.findall(r'\b\w+\b',i) == ['0']:
      b = [0]
    else:
      b = [i.count(",")+1]
    c = a+b
    table_df.loc[figurecount, 'Figures'] = c[0]
    figurecount += 1
  #Remove rows with no pieces or non-number
  table_df['Pieces'] = pd.to_numeric(table_df['Pieces'], errors='coerce')
  table_df = table_df[table_df['Pieces'].notna()]
  #Extract USD price as digits
  table_df['Price'] = table_df['Price'].str.extract(r'\$(\d+\.\d+)')
  #Remove rows with no price
  table_df['Price'] = table_df['Price'].replace('', np.nan)
  table_df = table_df[table_df['Price'].notna()]
  table_df['Price'] = table_df['Price'].astype(float)
  #Convert pieces column to int type
  table_df['Pieces'] = table_df['Pieces'].astype(int)
  #Extract year of release, rename released to year
  table_df.drop(columns=['Released'])
  table_df['Year'] = year
  #Redundancies
  table_df = table_df[table_df['Price'].notna()]
  table_df = table_df[table_df['Pieces'].notna()]
  return table_df