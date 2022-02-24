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
    X_ML_LightGBM=LightGBM_scaler_NWC.transform(X_ML)
    X_ML_CatBoost=CatBoost_scaler_NWC.transform(X_ML)

    Pn_GBR=GBR_NWC.predict(X_ML_GBR).item()
    Pn_LightGBM=LightGBM_NWC.predict(X_ML_LightGBM).item()
    Pn_CatBoost=CatBoost_NWC.predict(X_ML_CatBoost).item()

    Pd_GBR=k_red_NWC_GBR*Pn_GBR/1.25
    Pd_LightGBM=k_red_NWC_LightGBM*Pn_LightGBM/1.25
    Pd_CatBoost=k_red_NWC_CatBoost*Pn_CatBoost/1.25
elif Concrete_Type=='Lightweight':
    X_ML_GBR=GBR_scaler_LWC.transform(X_ML)
    X_ML_LightGBM=LightGBM_scaler_LWC.transform(X_ML)
    X_ML_CatBoost=CatBoost_scaler_LWC.transform(X_ML)

    Pn_GBR=GBR_LWC.predict(X_ML_GBR).item()
    Pn_LightGBM=LightGBM_LWC.predict(X_ML_LightGBM).item()
    Pn_CatBoost=CatBoost_LWC.predict(X_ML_CatBoost).item()

    Pd_GBR=k_red_LWC_GBR*Pn_GBR/1.25
    Pd_LightGBM=k_red_LWC_LightGBM*Pn_LightGBM/1.25
    Pd_CatBoost=k_red_LWC_CatBoost*Pn_CatBoost/1.25

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
st.write('fcm is the mean compressive strength of concrete; fu is the tensile strength of stud; d is the stud diameter; h is the stud height; ddo is the weld colar diameter; hw is the weld colar height; GBR is gradient boosting regressor; LightGBM is light gradient boosting machine; CatBoost is gradient boosting with categorical features support.')

st.subheader('Reference')
st.write('Degtyarev, V.V., Hicks, S.J., Reliability-based design shear resistance of headed studs in solid slabs predicted by machine learning models')
#st.markdown('[engrXiv](https://doi.org/10.31224/osf.io/mezar)', unsafe_allow_html=True)
#st.markdown('[ResearchGate](https://www.researchgate.net/publication/356756613_Buckling_and_ultimate_load_prediction_models_for_perforated_steel_beams_using_machine_learning_algorithms)', unsafe_allow_html=True)

st.subheader('Source code')
st.markdown('[GitHub](https://github.com/vitdegtyarev/Streamlit_Studs_Solid)', unsafe_allow_html=True)