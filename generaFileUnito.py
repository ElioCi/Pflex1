import streamlit as st
import pandas as pd
import os
from datetime import datetime

def SalvaDati():
    # Carica i file CSV contenenti i dati
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
    dati_concrete = pd.read_csv(fileConcrete)
    dati_steel = pd.read_csv(fileSteel)
    dati_reinf_layers = pd.read_csv(fileReinfLayers)
    dati_load = pd.read_csv(fileDatiLoad)
    dati_loadsMR = pd.read_csv(fileLoadsMR)


    # ✅ Conversione della colonna "rating" in interi
    #if "rating" in dati_components.columns:
    #    dati_components["Rating"] = pd.to_numeric(dati_components["Rating"], errors="coerce").fillna(0).astype(int)


    # Crea una prima colonna vuota per info generali "colonna_app" e poi le colonne di seprazione
    
    colonna_app = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
  
    # Inserisci l'intestazione personalizzata nella colonna vuota
    colonna_app.columns = [f"Pflex1 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]

    colonna_vuota1 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
    colonna_vuota2 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
    colonna_vuota3 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
    colonna_vuota4 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
    colonna_vuota5 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})
    colonna_vuota6 = pd.DataFrame({'': [None] * max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))})

    colonna_vuota1.columns = ["separator_1"]
    colonna_vuota2.columns = ["separator_2"]
    colonna_vuota3.columns = ["separator_3"]    
    colonna_vuota4.columns = ["separator_4"]    
    colonna_vuota5.columns = ["separator_5"]
    colonna_vuota6.columns = ["separator_6"]  


    # Allinea le righe dei due file in modo che abbiano la stessa lunghezza
    dati_generali = dati_generali.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_geo = dati_geo.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_concrete = dati_concrete.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_steel = dati_steel.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_reinf_layers = dati_reinf_layers.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_load = dati_load.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))
    dati_loadsMR = dati_loadsMR.reindex(range(max(len(dati_generali), len(dati_geo), len(dati_concrete), len(dati_steel), len(dati_reinf_layers), len(dati_load), len(dati_loadsMR))))

    # Unisci i due DataFrame con la colonna vuota tra di loro
    dati_uniti = pd.concat([
        colonna_app, dati_generali, colonna_vuota1, dati_geo, colonna_vuota2, dati_concrete, colonna_vuota3, 
        dati_steel, colonna_vuota4, dati_reinf_layers, colonna_vuota5, dati_load, colonna_vuota6, dati_loadsMR
        ],
    axis=1)


    # Permetti all'utente di scaricare il file

    csv = dati_uniti.to_csv(index=False)

    fileName = st.text_input("Insert file name (without extension):", value="MyProject")
    # Rimuove eventuali spazi e qualsiasi estensione già presente
    if fileName:
        base_name, _ = os.path.splitext(fileName.strip())
        fileName = base_name + ".csv"
        st.markdown(
            f"Data file name set to: <span style='color:blue;'>**{fileName}**</span>",
            unsafe_allow_html=True
        )
        st.download_button(label=f"💾 Download: {fileName}", data=csv, file_name=f"{fileName}", mime="text/csv", help= '***click here to save data in your personal drive***')


    else:
        st.warning("Please enter a valid file name.") 

    
    