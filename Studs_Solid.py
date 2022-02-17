import streamlit as st
import pandas as pd
from pickle import load
import joblib
import pickle
import numpy as np
import math
import os
from config.definitions import ROOT_DIR

#Load models and scalers
GBR_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_GBR_NWC.joblib'))
GBR_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_GBR_NWC.pkl'),'rb'))
GBR_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_GBR_LWC.joblib'))
GBR_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_GBR_LWC.pkl'),'rb'))
k_red_NWC_GBR=0.97
k_red_LWC_GBR=0.98

XGBoost_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_XGBoost_NWC.joblib'))
XGBoost_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_XGBoost_NWC.pkl'),'rb'))
XGBoost_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_XGBoost_LWC.joblib'))
XGBoost_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_XGBoost_LWC.pkl'),'rb'))
k_red_NWC_XGBoost=0.89
k_red_LWC_XGBoost=0.99

LightGBM_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_NWC.joblib'))
LightGBM_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_NWC.pkl'),'rb'))
LightGBM_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_LWC.joblib'))
LightGBM_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_LightGBM_LWC.pkl'),'rb'))
k_red_NWC_LightGBM=0.96
k_red_LWC_LightGBM=1.00

RF_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_RF_NWC.joblib'))
RF_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_RF_NWC.pkl'),'rb'))
RF_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_RF_LWC.joblib'))
RF_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_RF_LWC.pkl'),'rb'))
k_red_NWC_RF=0.94
k_red_LWC_RF=1.00

KNN_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_KNN_NWC.joblib'))
KNN_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_KNN_NWC.pkl'),'rb'))
KNN_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_KNN_LWC.joblib'))
KNN_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_KNN_LWC.pkl'),'rb'))
k_red_NWC_KNN=0.89
k_red_LWC_KNN=0.96

CatBoost_NWC=joblib.load(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_NWC.joblib'))
CatBoost_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_NWC.pkl'),'rb'))
CatBoost_LWC=joblib.load(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_LWC.joblib'))
CatBoost_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Stud_Solid_CatBoost_LWC.pkl'),'rb'))
k_red_NWC_CatBoost=1.00
k_red_LWC_CatBoost=1.00

DT_NWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_DT_NWC.joblib'))
DT_scaler_NWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_DT_NWC.pkl'),'rb'))
DT_LWC=joblib.load(os.path.join(ROOT_DIR,'Studs_Solid_DT_LWC.joblib'))
DT_scaler_LWC=pickle.load(open(os.path.join(ROOT_DIR,'Studs_Solid_DT_LWC.pkl'),'rb'))
k_red_NWC_DT=0.75
k_red_LWC_DT=0.95

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
d=df['d (mm)'].values.item()
h=d*df['h_d'].values.item()
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

X_ML=np.array([[fcm,fu,d,ddo,hw,h]])

if Concrete_Type=='Normal weight':
    X_ML_GBR=GBR_scaler_NWC.transform(X_ML)
    X_ML_XGBoost=XGBoost_scaler_NWC.transform(X_ML)
    X_ML_LightGBM=LightGBM_scaler_NWC.transform(X_ML)
    X_ML_RF=RF_scaler_NWC.transform(X_ML)
    #X_ML_KNN=KNN_scaler_NWC.transform(X_ML)
    X_ML_CatBoost=CatBoost_scaler_NWC.transform(X_ML)
    #X_ML_DT=DT_scaler_NWC.transform(X_ML)

    Pn_GBR=GBR_NWC.predict(X_ML_GBR).item()
    Pn_XGBoost=XGBoost_NWC.predict(X_ML_XGBoost).item()
    Pn_LightGBM=LightGBM_NWC.predict(X_ML_LightGBM).item()
    Pn_RF=RF_NWC.predict(X_ML_RF).item()
    #Pn_KNN=KNN_NWC.predict(X_ML_KNN).item()
    Pn_CatBoost=CatBoost_NWC.predict(X_ML_CatBoost).item()
    #Pn_DT=DT_NWC.predict(X_ML_DT).item()

    Pd_GBR=k_red_NWC_GBR*Pn_GBR/1.25
    Pd_XGBoost=k_red_NWC_XGBoost*Pn_XGBoost/1.25
    Pd_LightGBM=k_red_NWC_LightGBM*Pn_LightGBM/1.25
    Pd_RF=k_red_NWC_RF*Pn_RF/1.25
    #Pd_KNN=k_red_NWC_KNN*Pn_KNN/1.25
    Pd_CatBoost=k_red_NWC_CatBoost*Pn_CatBoost/1.25
    #Pd_DT=k_red_NWC_DT*Pn_DT/1.25
elif Concrete_Type=='Lightweight':
    X_ML_GBR=GBR_scaler_LWC.transform(X_ML)
    X_ML_XGBoost=XGBoost_scaler_LWC.transform(X_ML)
    X_ML_LightGBM=LightGBM_scaler_LWC.transform(X_ML)
    X_ML_RF=RF_scaler_LWC.transform(X_ML)
    #X_ML_KNN=KNN_scaler_LWC.transform(X_ML)
    X_ML_CatBoost=CatBoost_scaler_LWC.transform(X_ML)
    #X_ML_DT=DT_scaler_LWC.transform(X_ML)

    Pn_GBR=GBR_LWC.predict(X_ML_GBR).item()
    Pn_XGBoost=XGBoost_LWC.predict(X_ML_XGBoost).item()
    Pn_LightGBM=LightGBM_LWC.predict(X_ML_LightGBM).item()
    Pn_RF=RF_LWC.predict(X_ML_RF).item()
    #Pn_KNN=KNN_LWC.predict(X_ML_KNN).item()
    Pn_CatBoost=CatBoost_LWC.predict(X_ML_CatBoost).item()
    #Pn_DT=DT_LWC.predict(X_ML_DT).item()

    Pd_GBR=k_red_LWC_GBR*Pn_GBR/1.25
    Pd_XGBoost=k_red_LWC_XGBoost*Pn_XGBoost/1.25
    Pd_LightGBM=k_red_LWC_LightGBM*Pn_LightGBM/1.25
    Pd_RF=k_red_LWC_RF*Pn_RF/1.25
    #Pd_KNN=k_red_LWC_KNN*Pn_KNN/1.25
    Pd_CatBoost=k_red_LWC_CatBoost*Pn_CatBoost/1.25
    #Pd_DT=k_red_LWC_DT*Pn_DT/1.25

st.subheader('Nominal Shear Resistance, Pn (kN)')
Pn={'GBR': "{:.1f}".format(Pn_GBR),
    'XGBoost': "{:.1f}".format(Pn_XGBoost),
    'LightGBM': "{:.1f}".format(Pn_LightGBM),
    'RF': "{:.1f}".format(Pn_RF),
    #'KNN': "{:.1f}".format(Pn_KNN),
    'CatBoost': "{:.1f}".format(Pn_CatBoost),
    #'DT': "{:.1f}".format(Pn_DT)
	}
Pn_df=pd.DataFrame(Pn, index=[0])
st.dataframe(Pn_df)

st.subheader('Design Shear Resistance, Pd (kN)')
Pd={'GBR': "{:.1f}".format(Pd_GBR),
    'XGBoost': "{:.1f}".format(Pd_XGBoost),
    'LightGBM': "{:.1f}".format(Pd_LightGBM),
    'RF': "{:.1f}".format(Pd_RF),
    #'KNN': "{:.1f}".format(Pd_KNN),
    'CatBoost': "{:.1f}".format(Pd_CatBoost),
    #'DT': "{:.1f}".format(Pd_DT)
	}
Pd_df=pd.DataFrame(Pd, index=[0])
st.dataframe(Pd_df)