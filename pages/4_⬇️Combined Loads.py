import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

# --- CONFIGURAZIONE ---
st.set_page_config(
    page_title="Combined Loads",
    #layout="wide"
    )
st.title("⬇️ Combined Loads")
st.subheader("Units: kN, kN·m ")

with st.expander("🆘 Help"):
    st.markdown("""
    - **What to do here?**  
      In the table below You can see the stored combined loads. By clicking the check box '***Activate/Deactivate***' You can enter in **modification mode**.
               
    - **Before to go on ...**  
      After modifications or adding of new loads You have to click on the button '💾***Save load***', that appears below, to apply and save modifications . 
      
    - **further information:**  
      click on ℹ️ info button menu]
    """)


#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()

if 'datiLoad' not in st.session_state:
    st.session_state['datiLoad'] = []

if "axial" not in st.session_state:
    st.session_state["axial"] = 0.0  # default axial load
if "momentX" not in st.session_state:
    st.session_state["momentX"] = 0.0  # default moment around X axis
if"momentY" not in st.session_state:
    st.session_state["momentY"] = 0.0  # default moment around Y axis

    

#leggi dati di input da FileDatiGeo.csv
sessionDir = "sessions"
fileDatiLoad = f"{sessionDir}/DatiLoad_{st.session_state.session_id}.csv"

# ---------------------------------------------------
#  Combined Loads INPUT
# ---------------------------------------------------       

# === Parametro: numero massimo di carichi combinati ===
nloads_max = 20

# === Caricamento carichi esistenti se presenti ===

if 'num_loads' not in st.session_state:
    st.session_state.num_loads = 0

if os.path.exists(fileDatiLoad):
    df_loads_caricati = pd.read_csv(fileDatiLoad)
    num_loads = len(df_loads_caricati)
    gruppi_loads = df_loads_caricati.to_dict(orient="records")
    st.markdown("### Existing Combined Loads") 
    st.dataframe(df_loads_caricati, hide_index=True) 
else:
    df_loads_caricati = pd.DataFrame()
    st.info("No existing combined loads found.")
    num_loads = 0

st.session_state.num_loads = num_loads

col1, col2 = st.columns([1,1])

# === Pulsante per cancellare tutto e partire da zero ===
if col2.button("🗑️ Delete all combined loads"):
    # tieni solo le intestazioni (DataFrame vuoto con le stesse colonne)
    df_empty = df_loads_caricati.iloc[0:0]
    # salva di nuovo il file con sole intestazioni
    df_empty.to_csv(fileDatiLoad, index=False)
    #df_loads_caricati = pd.DataFrame()
    st.rerun()

checked = col1.checkbox(
    ":red[**A**ctivate] / :gray[**D**eactivate] ➡️ Input/Modification"
)

# === Sezione modifica gruppi (compare solo dopo click) ===


if checked == True:
    
    st.markdown("### Modify / Add Load Combinations")

    # Numero di strati o layers
    
    if not df_loads_caricati.empty:
        num_loads = st.number_input(
            "🔢 Adjust Number of load combinations",
            min_value=1, max_value=nloads_max,
            #value=len(df_loads_caricati), step=1
            value= st.session_state.num_loads, step =1
        )
        #print("layers presenti")
        
    else:
        num_loads = st.number_input(
            "🔢 Number of load combinations",
            min_value=1, max_value=nloads_max, value=1, step=1
        )

    
    gruppi_loads = []
    # Input completo dei loads
    for i in range(int(num_loads)):
        with st.expander(f"➕ Load {i+1}"):
            load = {}
            load["Load"] = f"CL_{i+1}"

            # Precompilazione se esiste
            load_esiste = (not df_loads_caricati.empty) and (i < len(df_loads_caricati))

            if load_esiste:
                saved = df_loads_caricati.iloc[i]   # Pandas Series
            else:
                saved = None                         # Load nuovo

            
            col1, col2, col3 = st.columns(3)

            load["axial"] = col1.number_input(
                f"Axial Force (Load {i+1})",
                value=0.00 if saved is None else saved["axial"],
                key=f"axial_{i}",
                #on_change=update_layer_values,
                args=(i,)
            
            )

            load["momentX"] = col2.number_input(
                f"Moment X (Load {i+1})",
                value=0.00 if saved is None else saved["momentX"],
                key=f"momentX_{i}",
                #on_change=update_layer_values,
                args=(i,)
            
            )

            load["momentY"] = col3.number_input(
                f"Moment Y (Load {i+1})",
                value=0.00 if saved is None else saved["momentY"],
                key=f"momentY_{i}",
                #on_change=update_layer_values,
                args=(i,)
            
            )


            gruppi_loads.append(load)
     

    # Tabella finale solo in questa sezione
    st.markdown("### Combined Loads Summary")
    dfLoads_finale = pd.DataFrame(gruppi_loads)
    st.dataframe(dfLoads_finale, hide_index=True)

    #df_finale.to_csv(fileReinfLayers, index=False)
    if st.button("💾 Save Combined Loads"):
        dfLoads_finale.to_csv(fileDatiLoad, index=False)

    #    
        st.success("Combined loads saved successfully!")



