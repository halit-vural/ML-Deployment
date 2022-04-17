import datetime
import matplotlib as plt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
from numpy.random import sample
from PIL import Image

st.markdown("# Auto Scout Prediction Project") 
st.markdown("Auto Scout data which using for this project, " +
            "scraped from the on-line car trading company(https://www.autoscout24.com) " +
            "in 2019, contains many features of 9 different car models.")
st.image("https://i.ibb.co/MZfS9Cv/Auto-Scout24-1100.jpg", caption="Credit: www.autoscout24.it/")
st.write("In this dataset, " +
         "we used the data of below mentioned car models. " +
         "Our Machine Learning model was built upon the dataset displayed below."
         )

# #ingest data from my google drive sharing link
# url = "https://drive.google.com/file/d/1KJLNreWSEGg1B9eXPzxC5kaJa4XXR59t/view?usp=sharing"
# path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# #df = pd.read_pickle(path)
# df = pd.read_csv(path)
# table = df.make_model.value_counts()
# table

#instead we do this way to make faster loading
table = pd.read_csv("model_count.csv")
table 

st.write("If we display the counts in a chart, we could have the one below:")

cht_img = Image.open('./charts/chart01_model_counts.png')
st.image(cht_img)


#################################################
#               USER INPUT                      #
#################################################

st.markdown("## Predict your car's price")
make_slct = st.selectbox("Select your car:", table['make model'])
age_slct = st.slider("Age", 0,5,1)
km_slct = st.slider("Your car's milage (km)", 0,400000,1) #according to max value in dataset
hp_slct = st.slider("Horse Power value (kW)", 20,300, 1)
gear_slct = st.selectbox("Gearing Type", ['Automatic', 'Manual', 'Semi-automatic'])

#  "make_model": 'Audi A3',
#  "age": 2,
#  "km": 17000,
#  "hp_kW": 66,
#  "Gearing_Type": "Automatic"

new_dict = {
    "make_model": make_slct,
    "age": age_slct,
    "km": km_slct,
    "hp_kW": hp_slct,
    "Gearing_Type": gear_slct
}

print(new_dict)

df_dict = pd.DataFrame([new_dict])
df_dict = pd.get_dummies(df_dict)

df_col = pd.read_csv("columns.csv")
df_dict = df_dict.reindex(columns = df_col.columns, fill_value=0)

#####################################################
#                    LOAD MODEL                     #
#####################################################

model = pickle.load(open("final_model", "rb"))
scaler = pickle.load(open("final_scaler", "rb"))


#scale the values in the dictionary
df_dict = scaler.transform(df_dict)

#make a prediction
btn = st.button("Predict")
if btn:
    st.write("Calculating...")
    pred_result = model.predict(df_dict)
else:
    st.write(" ")

#####################################################
#                  PREDICT & REPORT                 #
#####################################################


st.markdown("## Auto Scout Prediction Report")
date_format = "%b %d, %Y - %H:%M"
st.write("Reported:", datetime.datetime.now().strftime(date_format))

st.markdown("# Predicted Value of Your Car")

if btn:
    st.markdown(f"## ${pred_result}:.3f")

#Disclaimer to the user
st.write("This prediction tends to have some bias. There is no guaranty that the exact" +
         " price of your car can be listed. You should refer to an expert in the field, since " +
         "our model cannot cover unseen situation of your car. Not all of your car's features" +
         " are included in this prediction. Therefore, do not rely on your prediction alone.")

## Bence bu yorumu biraz geliştirmek lazım, şahsi kanaatım, saha tecrübem yok DS alanında.
# ama ben olsam zayıf tarafına odaklanmak yerine modele odaklanrım:
# RMSE skoru, R2 skoru ve diğer metriklere göre şu kadar veri üzerinde test edild (X_test ten bahsediyorum)
# test sonucu modelim gerçeğe bu kadar yakındır demek lazım, subjektif yorumumdur,