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

#Run train and test dataframes through dataset converter
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#Preprocessing normalization layer
def get_normalization_layer(name, dataset):
  normalizer = tf.keras.layers.Normalization(axis=None)
  feature_ds = dataset.map(lambda x, y: x[name])
  normalizer.adapt(feature_ds)
  return normalizer

#Preprocessing encoding layer
def get_category_encoding_layer(name, dataset, max_tokens=None):
  index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
  feature_ds = dataset.map(lambda x, y: x[name])
  index.adapt(feature_ds)
  encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
  return lambda feature: encoder(index(feature))


#Getting features
[(train_features, label_batch)] = train_ds.take(1)
#Single input list
all_inputs = []
encoded_features=[]

#Normalize numeric features
for header in ['Pieces', 'Year', 'Figures']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

#Encode categorical Theme column
categorical_col = tf.keras.Input(shape=(1,), name='Theme', dtype='string')
encoding_layer = get_category_encoding_layer(name='Theme',
                                               dataset=train_ds,
                                               max_tokens=5)
encoded_categorical_col = encoding_layer(categorical_col)
all_inputs.append(categorical_col)
encoded_features.append(encoded_categorical_col)

#Concatenate into one vector
all_features = tf.keras.layers.concatenate(encoded_features)
#Create model
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dense(64, activation="relu")(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

#Compiling, training, and evaluation
model.compile(optimizer='adam',loss='mean_absolute_error')

#Train model using callbacks to find best val_loss
epochs = 100
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.keras' ,
verbose=1, save_best_only=True)

train_history = model.fit(train_ds,
validation_data=(test_ds),
epochs=epochs, batch_size=batch_size, callbacks=[ checkpointer], verbose=1)
#Weights with best val_loss are saved, allowing for the predictor to easily load

model.load_weights('weights.keras')
model.save('predictor.keras')