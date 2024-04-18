#Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import re
from dateutil.parser import parse
import preprocessing as pp

dataframe = pd.DataFrame(columns = ['Pieces', 'Figures', 'Price', 'Year', 'Theme'])
years = np.arange(2017,2024)

#Page request
for i in years:
  page = requests.get('https://brickipedia.fandom.com/wiki/'+str(i))
  soup = bs(page.content)
  #Scrapes table and finds previous theme title
  for u in np.arange(1,len(soup.select("table"))):
    #Table 0 is skipped due to being a null table on all pages
    table = soup.select("table")[u]
    h3 = table.find_previous("h3")
    title = h3.find('span', class_='mw-headline').text.strip()
    dataframe = pd.concat([dataframe,pp.tablepreprocessing(table,title,i)],ignore_index=True)

dataframe['Pieces'] = dataframe['Pieces'].astype(np.int64)
dataframe['Figures'] = dataframe['Figures'].astype(np.int64)
dataframe['Year'] = dataframe['Year'].astype(np.int64)

train, test = train_test_split(dataframe, test_size=0.2)
#Takes a random 20% of rows and turns them into a test dataframe

#Convert pandas dataframe into tf dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Price')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size, drop_remainder=True)
  return ds

#Feature layer preprocessing
feature_columns = []

for header in ['Year', 'Pieces', 'Figures']:
  feature_columns.append(feature_column.numeric_column(header))

theme = feature_column.categorical_column_with_vocabulary_list(
      'Theme', dataframe.Theme.unique())
theme_embedding = feature_column.embedding_column(theme, dimension=8)
feature_columns.append(theme_embedding)

#Run train and test dataframes through dataset converter
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#Dense neural network w/ normalizer
model = keras.Sequential([keras.layers.DenseFeatures(feature_columns),
    keras.layers.Normalization(axis=-1),
    keras.layers.Dense(16, input_dim=3+len(theme.vocabulary_list), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)]
)

#Compiling, training, and evaluation
model.compile(optimizer='adam',loss='mean_absolute_error')

model.fit(train_ds,
          epochs=4)

model.evaluate(test_ds)
