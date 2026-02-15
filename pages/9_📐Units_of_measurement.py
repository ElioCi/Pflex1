import streamlit as st
import pandas as pd

st.title('📐 Units of measurement')
#leggi dati di input da DatiGenerali.csv
with open('files/units.csv') as file_units:
    dfunits = pd.read_csv(file_units, delimiter= ',', encoding="utf-8")   # lettura file e creazione
    #dfunits.drop(dfunits.columns[dfunits.columns.str.contains('unnamed', case= False)], axis=1, inplace= True)
    dfunits.columns = dfunits.columns.str.replace(r"[^a-zA-Z0-9_ ]", "", regex=True)  # Rimuove caratteri non alfanumerici

#st.write(dfunits.columns.tolist())

dfunits.drop(columns=['id'], inplace=True)
st.dataframe(dfunits, hide_index= True, height= 450, width= 500)

