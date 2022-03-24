import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
import os
from config.definitions import ROOT_DIR
from PIL import Image
import matplotlib.pyplot as plt

#Load models and scalers
GBR_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_GBR_NWC.joblib'))
GBR_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_GBR_NWC.pkl'),'rb'))
GBR_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_GBR_LWC.joblib'))
GBR_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_GBR_LWC.pkl'),'rb'))
k_red_NWC_GBR=0.97
k_red_LWC_GBR=0.98

LightGBM_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_NWC.joblib'))
LightGBM_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_NWC.pkl'),'rb'))
LightGBM_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_LWC.joblib'))
LightGBM_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_LWC.pkl'),'rb'))
k_red_NWC_LightGBM=0.96
k_red_LWC_LightGBM=1.00

CatBoost_NWC=joblib.load(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_NWC.joblib'))
CatBoost_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_NWC.pkl'),'rb'))
CatBoost_LWC=joblib.load(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_LWC.joblib'))
CatBoost_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_LWC.pkl'),'rb'))
k_red_NWC_CatBoost=1.00
k_red_LWC_CatBoost=1.00

st.header('Shear Resistance of Headed Studs in Solid Concrete Slabs Predicted by Machine Learning Models')

st.sidebar.header('User Input Parameters')

def user_input_features():
    concrete_type=st.sidebar.radio("Concrete Type",('Normal weight','Lightweight'))

    if concrete_type=='Normal weight': fcm=st.sidebar.selectbox("fcm (MPa)",(20,24,28,33,38,43,48,53,58,63,68,78,88,98))
    else: fcm=st.sidebar.selectbox("fcm (MPa)",(24,28,33,38,43,48,53,58))
	
    if concrete_type=='Normal weight': d=st.sidebar.slider("d (mm)",min_value=16, max_value=25, step=3)
    else: d=st.sidebar.slider("d (mm)",min_value=13, max_value=22, step=3)

    if concrete_type=='Normal weight': h_d=st.sidebar.slider("h/d",min_value=3.0, max_value=9.0, step=0.5)
    else: h_d=st.sidebar.slider("h/d",min_value=3.0, max_value=8.0, step=0.5)

    tensile_strength=st.sidebar.slider('fu (MPa)', min_value=450, max_value=600, step=50)

    data = {'Concrete Type': concrete_type,
	        'fcm (MPa)': fcm,
			'd (mm)': d,
        	'h_d': h_d,
            'fu (MPa)': tensile_strength}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

Concrete_Type=df['Concrete Type'].to_string(index=False)
fcm=df['fcm (MPa)'].values.item()
fck=fcm-8
d=df['d (mm)'].values.item()
h=d*df['h_d'].values.item()
h_d=h/d
fu=df['fu (MPa)'].values.item()

if d==13: ddo=17
elif d==16: ddo=21
elif d==19: ddo=23
elif d==22: ddo=29
elif d==25: ddo=31

if d==13: hw=3
elif d==16: hw=4.5
elif d==19: hw=6
elif d==22: hw=6
elif d==25: hw=7

st.subheader('Input Parameters')

input_parameters={'Concrete Type': Concrete_Type,
	              'fcm (MPa)': "{:.0f}".format(fcm),
			      'fu (MPa)': "{:.0f}".format(fu),
        	      'd (mm)': "{:.0f}".format(d),
				  'ddo (mm)': "{:.0f}".format(ddo),
				  'hw (mm)': "{:.1f}".format(hw),
				  'h (mm)': "{:.0f}".format(h)}
input_parameters_df=pd.DataFrame(input_parameters, index=[0])
st.dataframe(input_parameters_df)

X_ML_N=np.array([[fcm,fu,d,ddo,hw,h]])
X_ML_D=np.array([[fck,fu,d,ddo,hw,h]])

if Concrete_Type=='Normal weight':
    X_ML_GBR_N=GBR_scaler_NWC.transform(X_ML_N)
    X_ML_LightGBM_N=LightGBM_scaler_NWC.transform(X_ML_N)
    X_ML_CatBoost_N=CatBoost_scaler_NWC.transform(X_ML_N)
    
    X_ML_GBR_D=GBR_scaler_NWC.transform(X_ML_D)
    X_ML_LightGBM_D=LightGBM_scaler_NWC.transform(X_ML_D)
    X_ML_CatBoost_D=CatBoost_scaler_NWC.transform(X_ML_D)

    Pn_GBR=GBR_NWC.predict(X_ML_GBR_N).item()
    Pn_LightGBM=LightGBM_NWC.predict(X_ML_LightGBM_N).item()
    Pn_CatBoost=CatBoost_NWC.predict(X_ML_CatBoost_N).item()

    Pd_GBR=k_red_NWC_GBR*GBR_NWC.predict(X_ML_GBR_D).item()/1.25
    Pd_LightGBM=k_red_NWC_LightGBM*LightGBM_NWC.predict(X_ML_LightGBM_D).item()/1.25
    Pd_CatBoost=k_red_NWC_CatBoost*CatBoost_NWC.predict(X_ML_CatBoost_D).item()/1.25
elif Concrete_Type=='Lightweight':
    X_ML_GBR_N=GBR_scaler_LWC.transform(X_ML_N)
    X_ML_LightGBM_N=LightGBM_scaler_LWC.transform(X_ML_N)
    X_ML_CatBoost_N=CatBoost_scaler_LWC.transform(X_ML_N)
    
    X_ML_GBR_D=GBR_scaler_LWC.transform(X_ML_D)
    X_ML_LightGBM_D=LightGBM_scaler_LWC.transform(X_ML_D)
    X_ML_CatBoost_D=CatBoost_scaler_LWC.transform(X_ML_D)

    Pn_GBR=GBR_LWC.predict(X_ML_GBR_N).item()
    Pn_LightGBM=LightGBM_LWC.predict(X_ML_LightGBM_N).item()
    Pn_CatBoost=CatBoost_LWC.predict(X_ML_CatBoost_N).item()

    Pd_GBR=k_red_LWC_GBR*GBR_LWC.predict(X_ML_GBR_D).item()/1.25
    Pd_LightGBM=k_red_LWC_LightGBM*LightGBM_LWC.predict(X_ML_LightGBM_D).item()/1.25
    Pd_CatBoost=k_red_LWC_CatBoost*CatBoost_LWC.predict(X_ML_CatBoost_D).item()/1.25

st.subheader('Nominal Shear Resistance, Pn (kN)')
Pn={'GBR': "{:.1f}".format(Pn_GBR),
    'LightGBM': "{:.1f}".format(Pn_LightGBM),
    'CatBoost': "{:.1f}".format(Pn_CatBoost),
	}
Pn_df=pd.DataFrame(Pn, index=[0])
st.dataframe(Pn_df)

st.subheader('Design Shear Resistance, Pd (kN)')
Pd={'GBR': "{:.1f}".format(Pd_GBR),
    'LightGBM': "{:.1f}".format(Pd_LightGBM),
    'CatBoost': "{:.1f}".format(Pd_CatBoost),
	}
Pd_df=pd.DataFrame(Pd, index=[0])
st.dataframe(Pd_df)

image = Image.open(os.path.join(ROOT_DIR,'Stud.png'))
st.subheader('Dimensional Parameters of Studs')
st.image(image)

st.subheader('Nomenclature')
st.write('fcm is the mean compressive cylinder strength of the concrete at 28-days (characteristic compressive cylinder strength of the concrete fck = fcm – 8 MPa); fu is the tensile strength of stud; d is the stud diameter; h is the height of the stud after welding; ddo is the weld collar diameter according to ISO 13918; hw is the weld collar height according to ISO 13918; GBR is gradient boosting regressor; LightGBM is light gradient boosting machine; CatBoost is gradient boosting with categorical features support.')

st.subheader('Stud Resistance Plots as Functions of Design Variables')

if Concrete_Type=='Normal weight': 
    fcm1=np.array([20,24,28,33,38,43,48,53,58,63,68,78,88,98])
    fcm1=fcm1.reshape(len(fcm1),1)
    fck1=fcm1-8
    
    fu1=np.full((14,1),fu)
    d1=np.full((14,1),d)
    ddo1=np.full((14,1),ddo)
    hw1=np.full((14,1),hw)
    h1=np.full((14,1),h)
    
    X_ML_1_N=np.concatenate((fcm1, fu1, d1, ddo1, hw1, h1), axis=1)
    X_ML_1_D=np.concatenate((fck1, fu1, d1, ddo1, hw1, h1), axis=1)    
     
    X_ML_GBR_1_N=GBR_scaler_NWC.transform(X_ML_1_N)
    X_ML_LightGBM_1_N=LightGBM_scaler_NWC.transform(X_ML_1_N)
    X_ML_CatBoost_1_N=CatBoost_scaler_NWC.transform(X_ML_1_N)
    
    X_ML_GBR_1_D=GBR_scaler_NWC.transform(X_ML_1_D)
    X_ML_LightGBM_1_D=LightGBM_scaler_NWC.transform(X_ML_1_D)
    X_ML_CatBoost_1_D=CatBoost_scaler_NWC.transform(X_ML_1_D)    

    Pn_GBR_1=GBR_NWC.predict(X_ML_GBR_1_N)
    Pn_LightGBM_1=LightGBM_NWC.predict(X_ML_LightGBM_1_N)
    Pn_CatBoost_1=CatBoost_NWC.predict(X_ML_CatBoost_1_N)

    Pd_GBR_1=k_red_NWC_GBR*GBR_NWC.predict(X_ML_GBR_1_D)/1.25
    Pd_LightGBM_1=k_red_NWC_LightGBM*LightGBM_NWC.predict(X_ML_LightGBM_1_D)/1.25
    Pd_CatBoost_1=k_red_NWC_CatBoost*CatBoost_NWC.predict(X_ML_CatBoost_1_D)/1.25
    
else: 
    fcm1=np.array([24,28,33,38,43,48,53,58])
    fcm1=fcm1.reshape(len(fcm1),1)
    fck1=fcm1-8
    
    fu1=np.full((8,1),fu)
    d1=np.full((8,1),d)
    ddo1=np.full((8,1),ddo)
    hw1=np.full((8,1),hw)
    h1=np.full((8,1),h)
    
    X_ML_1_N=np.concatenate((fcm1, fu1, d1, ddo1, hw1, h1), axis=1)
    X_ML_1_D=np.concatenate((fck1, fu1, d1, ddo1, hw1, h1), axis=1)    
     
    X_ML_GBR_1_N=GBR_scaler_LWC.transform(X_ML_1_N)
    X_ML_LightGBM_1_N=LightGBM_scaler_LWC.transform(X_ML_1_N)
    X_ML_CatBoost_1_N=CatBoost_scaler_LWC.transform(X_ML_1_N)
    
    X_ML_GBR_1_D=GBR_scaler_LWC.transform(X_ML_1_D)
    X_ML_LightGBM_1_D=LightGBM_scaler_LWC.transform(X_ML_1_D)
    X_ML_CatBoost_1_D=CatBoost_scaler_LWC.transform(X_ML_1_D)    

    Pn_GBR_1=GBR_LWC.predict(X_ML_GBR_1_N)
    Pn_LightGBM_1=LightGBM_LWC.predict(X_ML_LightGBM_1_N)
    Pn_CatBoost_1=CatBoost_LWC.predict(X_ML_CatBoost_1_N)

    Pd_GBR_1=k_red_LWC_GBR*GBR_LWC.predict(X_ML_GBR_1_D)/1.25
    Pd_LightGBM_1=k_red_LWC_LightGBM*LightGBM_LWC.predict(X_ML_LightGBM_1_D)/1.25
    Pd_CatBoost_1=k_red_LWC_CatBoost*CatBoost_LWC.predict(X_ML_CatBoost_1_D)/1.25

fu2=np.array([450,500,550,600])
fu2=fu2.reshape(len(fu2),1)

fcm2=np.full((4,1),fcm)
fck2=fcm2-8
d2=np.full((4,1),d)
ddo2=np.full((4,1),ddo)
hw2=np.full((4,1),hw)
h2=np.full((4,1),h)

X_ML_2_N=np.concatenate((fcm2, fu2, d2, ddo2, hw2, h2), axis=1)
X_ML_2_D=np.concatenate((fck2, fu2, d2, ddo2, hw2, h2), axis=1)

if Concrete_Type=='Normal weight': 
    X_ML_GBR_2_N=GBR_scaler_NWC.transform(X_ML_2_N)
    X_ML_LightGBM_2_N=LightGBM_scaler_NWC.transform(X_ML_2_N)
    X_ML_CatBoost_2_N=CatBoost_scaler_NWC.transform(X_ML_2_N)
    
    X_ML_GBR_2_D=GBR_scaler_NWC.transform(X_ML_2_D)
    X_ML_LightGBM_2_D=LightGBM_scaler_NWC.transform(X_ML_2_D)
    X_ML_CatBoost_2_D=CatBoost_scaler_NWC.transform(X_ML_2_D)    

    Pn_GBR_2=GBR_NWC.predict(X_ML_GBR_2_N)
    Pn_LightGBM_2=LightGBM_NWC.predict(X_ML_LightGBM_2_N)
    Pn_CatBoost_2=CatBoost_NWC.predict(X_ML_CatBoost_2_N)

    Pd_GBR_2=k_red_NWC_GBR*GBR_NWC.predict(X_ML_GBR_2_D)/1.25
    Pd_LightGBM_2=k_red_NWC_LightGBM*LightGBM_NWC.predict(X_ML_LightGBM_2_D)/1.25
    Pd_CatBoost_2=k_red_NWC_CatBoost*CatBoost_NWC.predict(X_ML_CatBoost_2_D)/1.25
    
else: 
    X_ML_GBR_2_N=GBR_scaler_LWC.transform(X_ML_2_N)
    X_ML_LightGBM_2_N=LightGBM_scaler_LWC.transform(X_ML_2_N)
    X_ML_CatBoost_2_N=CatBoost_scaler_LWC.transform(X_ML_2_N)
    
    X_ML_GBR_2_D=GBR_scaler_LWC.transform(X_ML_2_D)
    X_ML_LightGBM_2_D=LightGBM_scaler_LWC.transform(X_ML_2_D)
    X_ML_CatBoost_2_D=CatBoost_scaler_LWC.transform(X_ML_2_D)    

    Pn_GBR_2=GBR_LWC.predict(X_ML_GBR_2_N)
    Pn_LightGBM_2=LightGBM_LWC.predict(X_ML_LightGBM_2_N)
    Pn_CatBoost_2=CatBoost_LWC.predict(X_ML_CatBoost_2_N)

    Pd_GBR_2=k_red_LWC_GBR*GBR_LWC.predict(X_ML_GBR_2_D)/1.25
    Pd_LightGBM_2=k_red_LWC_LightGBM*LightGBM_LWC.predict(X_ML_LightGBM_2_D)/1.25
    Pd_CatBoost_2=k_red_LWC_CatBoost*CatBoost_LWC.predict(X_ML_CatBoost_2_D)/1.25 
    
if Concrete_Type=='Normal weight': 
    d3=np.array([16,19,22,25])
    d3=d3.reshape(len(d3),1)

    fcm3=np.full((4,1),fcm)
    fck3=fcm3-8    
    fu3=np.full((4,1),fu)
    ddo3=np.full((4,1),ddo)
    hw3=np.full((4,1),hw)
    h3=np.full((4,1),h)
    
    X_ML_3_N=np.concatenate((fcm3, fu3, d3, ddo3, hw3, h3), axis=1)
    X_ML_3_D=np.concatenate((fck3, fu3, d3, ddo3, hw3, h3), axis=1)
     
    X_ML_GBR_3_N=GBR_scaler_NWC.transform(X_ML_3_N)
    X_ML_LightGBM_3_N=LightGBM_scaler_NWC.transform(X_ML_3_N)
    X_ML_CatBoost_3_N=CatBoost_scaler_NWC.transform(X_ML_3_N)
    
    X_ML_GBR_3_D=GBR_scaler_NWC.transform(X_ML_3_D)
    X_ML_LightGBM_3_D=LightGBM_scaler_NWC.transform(X_ML_3_D)
    X_ML_CatBoost_3_D=CatBoost_scaler_NWC.transform(X_ML_3_D)    

    Pn_GBR_3=GBR_NWC.predict(X_ML_GBR_3_N)
    Pn_LightGBM_3=LightGBM_NWC.predict(X_ML_LightGBM_3_N)
    Pn_CatBoost_3=CatBoost_NWC.predict(X_ML_CatBoost_3_N)

    Pd_GBR_3=k_red_NWC_GBR*GBR_NWC.predict(X_ML_GBR_3_D)/1.25
    Pd_LightGBM_3=k_red_NWC_LightGBM*LightGBM_NWC.predict(X_ML_LightGBM_3_D)/1.25
    Pd_CatBoost_3=k_red_NWC_CatBoost*CatBoost_NWC.predict(X_ML_CatBoost_3_D)/1.25
    
else: 
    d3=np.array([13,16,19,22])
    d3=d3.reshape(len(d3),1)

    fcm3=np.full((4,1),fcm)
    fck3=fcm3-8    
    fu3=np.full((4,1),fu)
    ddo3=np.full((4,1),ddo)
    hw3=np.full((4,1),hw)
    h3=np.full((4,1),h)
    
    X_ML_3_N=np.concatenate((fcm3, fu3, d3, ddo3, hw3, h3), axis=1)
    X_ML_3_D=np.concatenate((fck3, fu3, d3, ddo3, hw3, h3), axis=1)
     
    X_ML_GBR_3_N=GBR_scaler_LWC.transform(X_ML_3_N)
    X_ML_LightGBM_3_N=LightGBM_scaler_LWC.transform(X_ML_3_N)
    X_ML_CatBoost_3_N=CatBoost_scaler_LWC.transform(X_ML_3_N)
    
    X_ML_GBR_3_D=GBR_scaler_LWC.transform(X_ML_3_D)
    X_ML_LightGBM_3_D=LightGBM_scaler_LWC.transform(X_ML_3_D)
    X_ML_CatBoost_3_D=CatBoost_scaler_LWC.transform(X_ML_3_D)    

    Pn_GBR_3=GBR_LWC.predict(X_ML_GBR_3_N)
    Pn_LightGBM_3=LightGBM_LWC.predict(X_ML_LightGBM_3_N)
    Pn_CatBoost_3=CatBoost_LWC.predict(X_ML_CatBoost_3_N)

    Pd_GBR_3=k_red_LWC_GBR*GBR_LWC.predict(X_ML_GBR_3_D)/1.25
    Pd_LightGBM_3=k_red_LWC_LightGBM*LightGBM_LWC.predict(X_ML_LightGBM_3_D)/1.25
    Pd_CatBoost_3=k_red_LWC_CatBoost*CatBoost_LWC.predict(X_ML_CatBoost_3_D)/1.25
    
if Concrete_Type=='Normal weight': 
    h_d4=np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9])
    h_d4=h_d4.reshape(len(h_d4),1)
    h4=h_d4*d

    fcm4=np.full((13,1),fcm)
    fck4=fcm4-8    
    fu4=np.full((13,1),fu)
    d4=np.full((13,1),d)    
    ddo4=np.full((13,1),ddo)
    hw4=np.full((13,1),hw)
    
    X_ML_4_N=np.concatenate((fcm4, fu4, d4, ddo4, hw4, h4), axis=1)
    X_ML_4_D=np.concatenate((fck4, fu4, d4, ddo4, hw4, h4), axis=1)    
     
    X_ML_GBR_4_N=GBR_scaler_NWC.transform(X_ML_4_N)
    X_ML_LightGBM_4_N=LightGBM_scaler_NWC.transform(X_ML_4_N)
    X_ML_CatBoost_4_N=CatBoost_scaler_NWC.transform(X_ML_4_N)
    
    X_ML_GBR_4_D=GBR_scaler_NWC.transform(X_ML_4_D)
    X_ML_LightGBM_4_D=LightGBM_scaler_NWC.transform(X_ML_4_D)
    X_ML_CatBoost_4_D=CatBoost_scaler_NWC.transform(X_ML_4_D)    

    Pn_GBR_4=GBR_NWC.predict(X_ML_GBR_4_N)
    Pn_LightGBM_4=LightGBM_NWC.predict(X_ML_LightGBM_4_N)
    Pn_CatBoost_4=CatBoost_NWC.predict(X_ML_CatBoost_4_N)

    Pd_GBR_4=k_red_NWC_GBR*GBR_NWC.predict(X_ML_GBR_4_D)/1.25
    Pd_LightGBM_4=k_red_NWC_LightGBM*LightGBM_NWC.predict(X_ML_LightGBM_4_D)/1.25
    Pd_CatBoost_4=k_red_NWC_CatBoost*CatBoost_NWC.predict(X_ML_CatBoost_4_D)/1.25
    
else: 
    h_d4=np.array([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])
    h_d4=h_d4.reshape(len(h_d4),1)
    h4=h_d4*d

    fcm4=np.full((11,1),fcm)
    fck4=fcm4-8     
    fu4=np.full((11,1),fu)
    d4=np.full((11,1),d)    
    ddo4=np.full((11,1),ddo)
    hw4=np.full((11,1),hw)
    
    X_ML_4_N=np.concatenate((fcm4, fu4, d4, ddo4, hw4, h4), axis=1)
    X_ML_4_D=np.concatenate((fck4, fu4, d4, ddo4, hw4, h4), axis=1)    
     
    X_ML_GBR_4_N=GBR_scaler_LWC.transform(X_ML_4_N)
    X_ML_LightGBM_4_N=LightGBM_scaler_LWC.transform(X_ML_4_N)
    X_ML_CatBoost_4_N=CatBoost_scaler_LWC.transform(X_ML_4_N)
    
    X_ML_GBR_4_D=GBR_scaler_LWC.transform(X_ML_4_D)
    X_ML_LightGBM_4_D=LightGBM_scaler_LWC.transform(X_ML_4_D)
    X_ML_CatBoost_4_D=CatBoost_scaler_LWC.transform(X_ML_4_D)    

    Pn_GBR_4=GBR_LWC.predict(X_ML_GBR_4_N)
    Pn_LightGBM_4=LightGBM_LWC.predict(X_ML_LightGBM_4_N)
    Pn_CatBoost_4=CatBoost_LWC.predict(X_ML_CatBoost_4_N)

    Pd_GBR_4=k_red_LWC_GBR*GBR_LWC.predict(X_ML_GBR_4_D)/1.25
    Pd_LightGBM_4=k_red_LWC_LightGBM*LightGBM_LWC.predict(X_ML_LightGBM_4_D)/1.25
    Pd_CatBoost_4=k_red_LWC_CatBoost*CatBoost_LWC.predict(X_ML_CatBoost_4_D)/1.25   


f1 = plt.figure(figsize=(6.75,4), dpi=200)

ax1 = f1.add_subplot(2,2,1)
ax1.plot(fcm1, Pn_GBR_1, color='#e31a1c',linewidth=1.5, label='GBR Nominal',linestyle='solid')
ax1.plot(fcm1, Pn_LightGBM_1, color='#0070C0',linewidth=1.5, label='LightGBM Nominal',linestyle='solid')
ax1.plot(fcm1, Pn_CatBoost_1, color='#00B050',linewidth=1.5, label='CatBoost Nominal',linestyle='solid')
fcm_loc=np.where(fcm1==fcm)[0].item()
ax1.scatter(fcm,Pn_GBR_1[fcm_loc],marker='o',facecolors='#e31a1c')
ax1.scatter(fcm,Pn_LightGBM_1[fcm_loc],marker='o',facecolors='#0070C0')
ax1.scatter(fcm,Pn_CatBoost_1[fcm_loc],marker='o',facecolors='#00B050')
ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax1.set_ylabel('Pn (kN)', fontsize=10)
ax1.set_xlabel('fcm (MPa)', fontsize=10)

ax2 = f1.add_subplot(2,2,4)
ax2.plot(fu2, Pn_GBR_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax2.plot(fu2, Pn_LightGBM_2, color='#0070C0',linewidth=1.5, linestyle='solid')
ax2.plot(fu2, Pn_CatBoost_2, color='#00B050',linewidth=1.5, linestyle='solid')
fu_loc=np.where(fu2==fu)[0].item()
ax2.scatter(fu,Pn_GBR_2[fu_loc],marker='o',facecolors='#e31a1c')
ax2.scatter(fu,Pn_LightGBM_2[fu_loc],marker='o',facecolors='#0070C0')
ax2.scatter(fu,Pn_CatBoost_2[fu_loc],marker='o',facecolors='#00B050')
ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax2.set_ylabel('Pn (kN)', fontsize=10)
ax2.set_xlabel('fu (MPa)', fontsize=10)

ax3 = f1.add_subplot(2,2,2)
ax3.plot(d3, Pn_GBR_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax3.plot(d3, Pn_LightGBM_3, color='#0070C0',linewidth=1.5, linestyle='solid')
ax3.plot(d3, Pn_CatBoost_3, color='#00B050',linewidth=1.5, linestyle='solid')
d_loc=np.where(d3==d)[0].item()
ax3.scatter(d,Pn_GBR_3[d_loc],marker='o',facecolors='#e31a1c')
ax3.scatter(d,Pn_LightGBM_3[d_loc],marker='o',facecolors='#0070C0')
ax3.scatter(d,Pn_CatBoost_3[d_loc],marker='o',facecolors='#00B050')
ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax3.set_ylabel('Pn (kN)', fontsize=10)
ax3.set_xlabel('d (mm)', fontsize=10)

ax4 = f1.add_subplot(2,2,3)
ax4.plot(h_d4, Pn_GBR_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax4.plot(h_d4, Pn_LightGBM_4, color='#0070C0',linewidth=1.5, linestyle='solid')
ax4.plot(h_d4, Pn_CatBoost_4, color='#00B050',linewidth=1.5, linestyle='solid')
h_d_loc=np.where(h_d4==h_d)[0].item()
ax4.scatter(h_d,Pn_GBR_4[h_d_loc],marker='o',facecolors='#e31a1c')
ax4.scatter(h_d,Pn_LightGBM_4[h_d_loc],marker='o',facecolors='#0070C0')
ax4.scatter(h_d,Pn_CatBoost_4[h_d_loc],marker='o',facecolors='#00B050')
ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax4.set_ylabel('Pn (kN)', fontsize=10)
ax4.set_xlabel('h/d', fontsize=10)

f1.legend(ncol=3, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
f1.tight_layout()
st.pyplot(f1)

f2 = plt.figure(figsize=(6.75,4), dpi=200)

ax1 = f2.add_subplot(2,2,1)
ax1.plot(fck1, Pd_GBR_1, color='#e31a1c',linewidth=1.5, label='GBR Design',linestyle='solid')
ax1.plot(fck1, Pd_LightGBM_1, color='#0070C0',linewidth=1.5, label='LightGBM Design',linestyle='solid')
ax1.plot(fck1, Pd_CatBoost_1, color='#00B050',linewidth=1.5, label='CatBoost Design',linestyle='solid')
fck_loc=np.where(fck1==fck)[0].item()
ax1.scatter(fck,Pd_GBR_1[fck_loc],marker='o',facecolors='#e31a1c')
ax1.scatter(fck,Pd_LightGBM_1[fck_loc],marker='o',facecolors='#0070C0')
ax1.scatter(fck,Pd_CatBoost_1[fck_loc],marker='o',facecolors='#00B050')
ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax1.set_ylabel('Pd (kN)', fontsize=10)
ax1.set_xlabel('fck (MPa)', fontsize=10)

ax2 = f2.add_subplot(2,2,4)
ax2.plot(fu2, Pd_GBR_2, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax2.plot(fu2, Pd_LightGBM_2, color='#0070C0',linewidth=1.5, linestyle='solid')
ax2.plot(fu2, Pd_CatBoost_2, color='#00B050',linewidth=1.5, linestyle='solid')
fu_loc=np.where(fu2==fu)[0].item()
ax2.scatter(fu,Pd_GBR_2[fu_loc],marker='o',facecolors='#e31a1c')
ax2.scatter(fu,Pd_LightGBM_2[fu_loc],marker='o',facecolors='#0070C0')
ax2.scatter(fu,Pd_CatBoost_2[fu_loc],marker='o',facecolors='#00B050')
ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax2.set_ylabel('Pd (kN)', fontsize=10)
ax2.set_xlabel('fu (MPa)', fontsize=10)

ax3 = f2.add_subplot(2,2,2)
ax3.plot(d3, Pd_GBR_3, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax3.plot(d3, Pd_LightGBM_3, color='#0070C0',linewidth=1.5, linestyle='solid')
ax3.plot(d3, Pd_CatBoost_3, color='#00B050',linewidth=1.5, linestyle='solid')
d_loc=np.where(d3==d)[0].item()
ax3.scatter(d,Pd_GBR_3[d_loc],marker='o',facecolors='#e31a1c')
ax3.scatter(d,Pd_LightGBM_3[d_loc],marker='o',facecolors='#0070C0')
ax3.scatter(d,Pd_CatBoost_3[d_loc],marker='o',facecolors='#00B050')
ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax3.set_ylabel('Pd (kN)', fontsize=10)
ax3.set_xlabel('d (mm)', fontsize=10)

ax4 = f2.add_subplot(2,2,3)
ax4.plot(h_d4, Pd_GBR_4, color='#e31a1c',linewidth=1.5, linestyle='solid')
ax4.plot(h_d4, Pd_LightGBM_4, color='#0070C0',linewidth=1.5, linestyle='solid')
ax4.plot(h_d4, Pd_CatBoost_4, color='#00B050',linewidth=1.5, linestyle='solid')
h_d_loc=np.where(h_d4==h_d)[0].item()
ax4.scatter(h_d,Pd_GBR_4[h_d_loc],marker='o',facecolors='#e31a1c')
ax4.scatter(h_d,Pd_LightGBM_4[h_d_loc],marker='o',facecolors='#0070C0')
ax4.scatter(h_d,Pd_CatBoost_4[h_d_loc],marker='o',facecolors='#00B050')
ax4.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.5)
ax4.set_ylabel('Pd (kN)', fontsize=10)
ax4.set_xlabel('h/d', fontsize=10)

f2.legend(ncol=3, fontsize=10, bbox_to_anchor=(0.52, -0.07), loc='lower center')
f2.tight_layout()
st.pyplot(f2)

st.write('Note: Circle markers indicate input parameters selected by user.')


st.subheader('Reference')
st.write('Degtyarev, V.V., Hicks, S.J., Reliability-based design shear resistance of headed studs in solid slabs predicted by machine learning models')
#st.markdown('[engrXiv](https://doi.org/10.31224/osf.io/mezar)', unsafe_allow_html=True)
#st.markdown('[ResearchGate](https://www.researchgate.net/publication/356756613_Buckling_and_ultimate_load_prediction_models_for_perforated_steel_beams_using_machine_learning_algorithms)', unsafe_allow_html=True)

st.subheader('Source code')
st.markdown('[GitHub](https://github.com/vitdegtyarev/Streamlit_Studs_Solid)', unsafe_allow_html=True)
