#Import
import tkinter as tk
import tensorflow as tf
import numpy as np
import pandas as pd

#Load pre-trained model
model = tf.keras.models.load_model('predictor.keras')

#Create window
window = tk.Tk()
#Label
label = tk.Label(text="Welcome to the Lego Price Predictor!" '\n' "Please input the parameters of the set you'd like to predict!",
                    fg="white",
                    bg="black",
                    width=70,
                    height=5)
label.pack(fill=tk.BOTH,expand=True)

#Input boxes
pieces_frm = tk.Frame(master=window)
pieces_frm.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
pieces_lbl = tk.Label(master=pieces_frm,text="Pieces")
pieces_ent = tk.Entry(master=pieces_frm)
pieces_lbl.pack()
pieces_ent.pack()

figs_frm = tk.Frame(master=window)
figs_frm.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
figs_lbl = tk.Label(master=figs_frm,text="Minifigures")
figs_ent = tk.Entry(master=figs_frm)
figs_lbl.pack()
figs_ent.pack()

year_frm = tk.Frame(master=window)
year_frm.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
year_lbl = tk.Label(master=year_frm,text="Year")
year_ent = tk.Entry(master=year_frm)
year_lbl.pack()
year_ent.pack()

theme_frm = tk.Frame(master=window)
theme_frm.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
theme_lbl = tk.Label(master=theme_frm,text="Theme")
theme_ent = tk.Entry(master=theme_frm)
theme_lbl.pack()
theme_ent.pack()

#Prediction dataframe
def prediction_dataset(pieces,figs,year,theme):
    #Random data of size 32 (prediction is better when running a full dataset)
    predict_df = pd.DataFrame({
        "Pieces" : np.random.randint(0,high=15000,size=1280),
        "Figures" : np.random.randint(0,high=300,size=1280),
        "Year" : np.random.randint(2000,high=2050,size=1280),
        "Theme" : 'BrickHeadz',
    })
    #Replace first random value with input
    predict_df.iloc[0,0] = np.int64(pieces)
    predict_df.iloc[0,1] = np.int64(figs)
    predict_df.iloc[0,2] = np.int64(year)
    #Random labels
    
    ds = tf.data.Dataset.from_tensor_slices(dict(predict_df))
    ds = ds.batch(32, drop_remainder=True)
    return ds

#Button click event
#Currently just returns pieces, will change to run prediction
def predictclick():
    #Check to make sure only numbers for numeric data inputs
    if pieces_ent.get().isnumeric() == False or figs_ent.get().isnumeric == False or year_ent.get().isnumeric() == False:
        label["text"] = "Invalid input, please only use numbers for Pieces, Minifigures, and Year."
    #Check to make sure no empty inputs
    elif pieces_ent.get() == "" or figs_ent.get() == "" or year_ent.get() == "" or theme_ent.get() == "":
        label["text"] = "Please make sure all input boxes are filled before pressing submit."
    else:
        data = prediction_dataset(pieces_ent.get(),figs_ent.get(),year_ent.get(),theme_ent.get())
        prediction = model.predict(data, verbose=0)
        label["text"] = "Your predicted price is $" + str(round(float(str(prediction[0])[1:-1]),2)) + '\n' "Feel free to enter another prediction!"

#Predict button
button_frm = tk.Frame(master=window)
button_frm.pack(side=tk.LEFT,fill=tk.BOTH)
button = tk.Button(master=button_frm,text="Predict",command=predictclick)
button.pack()


window.mainloop()