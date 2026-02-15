import streamlit as st
import pandas as pd
import csv
import os

from generaFileUnito import SalvaDati

# Titolo dell'applicazione
st.set_page_config(page_title="Save_Project")
st.title('💾Save Project')

with st.expander("🆘 Help"):
    st.markdown("""
    - **What to do here?**  
      In this module you can choose a personal file name and download it in your local drive. 
      The file include all project data and you can recall it when you need by ticking the option "stored project" in the main module. 
                                               

    """)

#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()


sessionDir = "sessions" 
fileDatiGenerali = os.path.join(sessionDir, f"DatiGenerali_{st.session_state.session_id}.csv")
fileDatiGeo = os.path.join(sessionDir, f"DatiGeo_{st.session_state.session_id}.csv")
fileConcrete = os.path.join(sessionDir, f"concrete_{st.session_state.session_id}.csv")
fileSteel = os.path.join(sessionDir, f"steel_{st.session_state.session_id}.csv")
fileReinfLayers = os.path.join(sessionDir, f"Reinf_layers_{st.session_state.session_id}.csv")
fileDatiLoad = os.path.join(sessionDir, f"DatiLoad_{st.session_state.session_id}.csv")
fileLoadsMR = os.path.join(sessionDir, f"loadsMR_{st.session_state.session_id}.csv")



dati_generali = pd.read_csv(fileDatiGenerali)
dati_geo = pd.read_csv(fileDatiGeo)
dati_concrete = pd.read_csv(fileConcrete, sep= ";")
dati_steel = pd.read_csv(fileSteel, sep= ";")
dati_reinf_layers = pd.read_csv(fileReinfLayers)
dati_load = pd.read_csv(fileDatiLoad)
dati_loadsMR = pd.read_csv(fileLoadsMR)


#dati_temperatures["TempC"] = pd.to_numeric(dati_temperatures["TempC"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
#dati_temperatures["TempF"] = pd.to_numeric(dati_temperatures["TempF"].astype(str).str.replace(",", ".", regex=False), errors="coerce")


# Dizionario con i tuoi DataFrame
dati_dict = {
    "General Data": dati_generali,
    "Geometry Data": dati_geo,
    "Concrete Data": dati_concrete,
    "Steel Data": dati_steel,
    "Reinforcement Layers Data": dati_reinf_layers,
    "Load Data": dati_load,
    "LoadsMR Data": dati_loadsMR

}

# Lista dei DataFrame vuoti
vuoti = [nome for nome, df in dati_dict.items() if df.empty]

# Controllo e messaggi
if vuoti:
    st.warning("⚠️ Some data files are empty!")
    for nome in vuoti:
        st.error(f"❌ {nome} is empty!")
else:
    #st.success("✅ All data files are complete!")
    
    # Mostra un'anteprima dei dati separati
    st.write("General Data:")
    st.dataframe(dati_generali)
    
    st.write("Geometry Data:")
    st.dataframe(dati_geo)

    st.write("Concrete Data:")
    st.dataframe(dati_concrete)

    st.write("Steel Data:")
    st.dataframe(dati_steel)

    st.write("Reinforcement Layers Data:")
    st.dataframe(dati_reinf_layers)

    st.write("Load Data:")
    st.dataframe(dati_load) 

    st.write("Loads MR Data:")
    st.dataframe(dati_loadsMR)

    
     
    SalvaDati()






