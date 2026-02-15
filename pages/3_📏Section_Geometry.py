import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

# --- CONFIGURAZIONE ---
st.set_page_config(
    page_title="Section Geometry",
    #layout="wide"
    )

st.title("📏 Section Geometry")
#st.subheader("acc. to ASME B16.34 and B16.5 - 2025 ")

with st.expander("🆘 Help"):
    st.markdown("""
    - **What to do here?**  
        In this module you have to confirm or change the geometric dimensions od the reinforced concrete section.
        You can also add or modify the reinforcement layers, by clicking on the checkbox to activate the input section
                           
    - **Before to go on ...**  
        After any geometric change or addition of new reinforcement layers it is strictly required to click the buttons '💾***Save Geometry Data***',
        and '💾***Save Reinforcement Layers***' to apply and save your changes. 
      
    - **further information:**  
        click on ℹ️ info button menu]
    """)


#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()

if 'datiGeo' not in st.session_state:
    st.session_state['datiGeo'] = []

if "base" not in st.session_state:
    st.session_state["base"] = 300.0  # default width
if "height" not in st.session_state:
    st.session_state["height"] = 500.0  # default height

if"cover" not in st.session_state:
    st.session_state["cover"] = 30.0  # default cover

if "stirrups_dia" not in st.session_state:
    st.session_state["stirrups_dia"] = 8  # default stirrups diameter

    
def update_layer_values(layer_id):
    try:
        n_bars = int(st.session_state[f"num_bars_{layer_id}"])
        dia = float(st.session_state[f"dia_{layer_id}"])
        b = float(st.session_state["b"])
        cover = float(st.session_state["cover"])

        # y automatico
        y_auto = cover + dia/2
        st.session_state[f"y_layer_{layer_id}"] = y_auto

        # primo ferro
        x_first = cover + dia/2
        st.session_state[f"x_first_{layer_id}"] = x_first

        # spacing
        if n_bars > 1:
            x_last = b - cover - dia/2
            spacing = (x_last - x_first) / (n_bars - 1)
        else:
            spacing = 0

        st.session_state[f"rebar_spacing_{layer_id}"] = spacing

    except Exception as e:
        st.warning(f"Error updating layer {layer_id}: {e}")


#leggi dati di input da FileDatiGeo.csv
sessionDir = "sessions"
fileDatiGeo = f"{sessionDir}/DatiGeo_{st.session_state.session_id}.csv"

if os.path.exists(fileDatiGeo):
    with open(fileDatiGeo) as file_input_geo:
        dfgeo = pd.read_csv(file_input_geo)   # lettura file e creazione
        dfgeo.drop(dfgeo.columns[dfgeo.columns.str.contains('unnamed', case= False)], axis=1, inplace= True)

    st.session_state.base = dfgeo.loc[0,'base']
    st.session_state.height = dfgeo.loc[0,'height']
    st.session_state.cover = dfgeo.loc[0,'cover']
    st.session_state.stirrups_dia = dfgeo.loc[0,'stirrups_dia']


#st.markdown(
#    "<h6 style='color:red; font-weight:normal;'>=== Read-only access for viewing and querying ===</h6>",
#    unsafe_allow_html=True
#)

# === Caricamento file dati statici ===
df_DiametriArmature = pd.read_csv("files/diametri_armature.csv", sep=";")

# Definizione della Geometria sezione rettangolare ---
# impostazione di: base, altezza, copriferro

col1, col2 = st.columns(2)
with col1:
    b = st.number_input("Width b [mm]", min_value=50.0, max_value=2000.0, value=st.session_state["base"], step=50.0)
with col2:
    h = st.number_input("Height h [mm]", min_value=50.0, max_value=2000.0, value=st.session_state["height"], step=50.0)

with col1:
    cover = st.number_input("Concrete cover (side and bottom) [mm]", min_value=10.0, max_value=100.0, value=st.session_state["cover"], step=5.0)
with col2:
    stirrups_dia = st.number_input("Stirrups Diameter [mm]", min_value=4, max_value=14, value=int(st.session_state["stirrups_dia"]), step=2)


st.session_state["datiGeo"] = [{
    "base": b,
    "height": h,
    "cover": cover,
    "stirrups_dia": stirrups_dia
    }]

dfgeo = pd.DataFrame(st.session_state['datiGeo'])
#st.subheader('Summary of geometric data')
#st.dataframe(dfgeo, hide_index= True)

if st.button("💾 Save Geometry Data"):
    dfgeo.to_csv(fileDatiGeo, index=False)   # salva dati generali su sessions/DatiGenerali_idUser.csv
    st.success("Geometry data saved successfully!") 


# ---------------------------------------------------
#  REINFORCEMENT LAYERS INPUT
# ---------------------------------------------------       

# === Parametro: numero massimo di layers ===
n_max = 10

# === Caricamento layers di armatura esistenti se presenti ===
#sessionDir = "sessions"
fileReinfLayers = f"{sessionDir}/Reinf_layers_{st.session_state.session_id}.csv"

if 'num_layers' not in st.session_state:
    st.session_state.num_layers = 0

if os.path.exists(fileReinfLayers):
    df_layers_caricati = pd.read_csv(fileReinfLayers)
    num_layers = len(df_layers_caricati)
    #st.markdown("### Existing Reinforcement Layers")
    #st.dataframe(df_layers_caricati, hide_index=True)
    gruppi_dati = df_layers_caricati.to_dict(orient="records")  
else:
    df_layers_caricati = pd.DataFrame()
    st.info("No existing reinforcement layer found.")
    num_layers = 0

st.session_state.num_layers = num_layers

col1, col2 = st.columns([1,1])

# === Pulsante per cancellare tutto e partire da zero ===
if col2.button("🗑️ Delete all reinforcement layers"):
    # tieni solo le intestazioni (DataFrame vuoto con le stesse colonne)
    df_empty = df_layers_caricati.iloc[0:0]
    # salva di nuovo il file con sole intestazioni
    df_empty.to_csv(fileReinfLayers, index=False)
    #df_layers_caricati = pd.DataFrame()
    st.rerun()

checked = col1.checkbox(
    ":red[**A**ctivate] / :gray[**D**eactivate] ➡️ Input/Modification"
)

# === Sezione modifica gruppi (compare solo dopo click) ===


if checked == True:
    
    st.markdown("### Modify / Add Reinforcement Layers")

    # Numero di strati o layers
    
    if not df_layers_caricati.empty:
        num_layers = st.number_input(
            "🔢 Adjust Number of reinforcement layers",
            min_value=1, max_value=n_max,
            #value=len(df_layers_caricati), step=1
            value= st.session_state.num_layers, step =1
        )
        #print("layers presenti")
        
    else:
        num_layers = st.number_input(
            "🔢 Number of reinforcement layers",
            min_value=1, max_value=n_max, value=1, step=1
        )


    #diametri_armature = df_DiametriArmature["Diam"].dropna().unique()
    diametri_armature = df_DiametriArmature["Diam"].astype(float).dropna().unique()
    
    gruppi_dati = []
    # Input completo dei layers
    for i in range(int(num_layers)):
        with st.expander(f"➕ Layer {i+1}"):
            layer = {}
            layer["Layer"] = f"L_{i+1}"

            # Precompilazione se esiste
            layer_esiste = (not df_layers_caricati.empty) and (i < len(df_layers_caricati))

            if layer_esiste:
                saved = df_layers_caricati.iloc[i]   # Pandas Series
            else:
                saved = None                         # Layer nuovo

            
            col1, col2, col3 = st.columns(3)

            layer["num_bars"] = col1.number_input(
                f"num_bars (Layer {i+1})",
                value=2 if saved is None else saved["num_bars"],
                key=f"num_bars_{i}",
                #on_change=update_layer_values,
                args=(i,)
            
            )


            layer["dia"] = col2.selectbox(
                f"Diameter (Layer {i+1})",
                diametri_armature,
                index=6 if saved is None else list(diametri_armature).index(saved["dia"]),
                key=f"dia_{i}",
                #on_change=update_layer_values,
                args=(i,),
                )




            x_first_auto = cover +  layer["dia"]/2 + stirrups_dia
            
            if (layer["num_bars"] - 1) != 0:
                spacing_auto = (b - 2*cover - layer["dia"]-2*stirrups_dia)/(layer["num_bars"] - 1)
            else:
                spacing_auto = 0.0    


            #st.write("min horizontal offset from left edge = ", cover + layer["dia"]/2, "mm")
            #st.write("rebar equal spacing = ", (b - 2*cover - layer["dia"])/(layer["num_bars"] - 1), "mm")

            
            layer["y_layer"] = col3.number_input(
                f"y_layer mm (Layer {i+1})",
                value=0.0 if saved is None else saved["y_layer"],
                key=f"y_layer_{i}",
                
            )

            col3.markdown(
                f"<span style='font-size:1.0rem; color:gray; font-style:italic;'>"
                f"-> offset from: {cover + layer['dia']/2 + stirrups_dia} to {h - cover - layer['dia']/2 - stirrups_dia}"
                f"</span>",
                unsafe_allow_html=True
            )

  
            #col3.write(f"offset from: {cover + layer['dia']/2} to {h - cover -layer['dia']/2}")


            layer["x_first"] = x_first_auto
            #layer["x_first"] = st.number_input(
            #    f"distance from left side of first rebar mm (Layer {i+1})",
            #    key=f"x_first_{i}",
            #    value=cover + layer["dia"]/2 if saved is None else saved["x_first"]
            #)

            layer["rebar_spacing"] = spacing_auto
            #layer["rebar_spacing"] = st.number_input(
            #    f"rebar spacing mm (Layer {i+1})",
            #    key=f"rebar_spacing_{i}",
            #    value=0.0 if saved is None else saved["rebar_spacing"]
            #)


            gruppi_dati.append(layer)
     

    # Tabella finale solo in questa sezione
    st.markdown("### Reinforcement Layers Summary")
    df_finale = pd.DataFrame(gruppi_dati)
    st.dataframe(df_finale, hide_index=True)

    #df_finale.to_csv(fileReinfLayers, index=False)
    if st.button("💾 Save Reinforcement Layers"):
        df_finale.to_csv(fileReinfLayers, index=False)

        dfgeo.to_csv(fileDatiGeo, index=False)   # salva anche dati geometrici per sicurezza nel caso l'utente non lo abbia fatto prima 
    #    
        st.success("Reinforcement layers saved successfully!")



# ---------------------------------------------------
#  DRAW SECTION
# ---------------------------------------------------
st.header("Section View")

fig, ax = plt.subplots(figsize=(6,5), dpi= 300)
# ax.set_position([0.25, 0.25, 0.5, 0.5])
# Riduci e centra il grafico ax.set_position([0.2, 0.2, 0.6, 0.6])
ax.set_position([0.2, 0.2, 0.5, 0.5])
#plt.subplots_adjust(left=0.25, right=1.75, top=0.75, bottom=0.25)

sezione_cls = patches.Rectangle([0, 0], width=b, height=h, fill=True, facecolor="lightgray", linewidth=0.5, edgecolor='darkgray')

ax.add_patch(sezione_cls)


#impostazione scala
ax.set_xlim(-50, b+10)
ax.set_ylim(-50, h+50)

# stesso rapporto di scala su assi x e y
ax.set_aspect('equal', 'box')  # Rapporto 1:1 tra gli assi

ax.set_frame_on(False)   # Disabilita il rettangolo che racchiude il grafico

# Impostare solo le etichette delle quote dei due lati
ax.set_xlabel('X (cm)', fontsize= 5, color='red')
ax.set_ylabel('Y (cm)', fontsize= 5, color='red')

# Tick solo agli estremi dell’asse X ax.set_xticks([0, b]) # Tick solo agli estremi dell’asse Y ax.set_yticks([0, h])
# Tick solo agli estremi
ax.set_xticks([0, b])
ax.set_yticks([0, h])

# Dimensione e stile dei numeri
ax.tick_params(axis='both', labelsize=4, color='gray')
ax.tick_params(axis='both', width=0.2)


#ax.invert_yaxis()
#ax.set_position([0.2, 0.2, 0.4, 0])

# Draw bars for each layer

for nlay in range(num_layers):
    layer = gruppi_dati[nlay]
    x_start = layer["x_first"]
    y = layer["y_layer"]
    dia = layer["dia"]
    n_bars = int(layer["num_bars"])
    spacing = layer["rebar_spacing"]
    for i in range(n_bars):
        xi = x_start + i * spacing
        if xi + dia / 2 > b:
            continue
        ax.add_patch(plt.Circle((xi, y), dia / 2, color='red', linewidth=0.1 , fill=True))
    ax.text(b + 10, y, f"⌀{int(dia)} @ {int(spacing)} mm", fontsize=5, va='center', color='black')

# Cover lines (for clarity)

#perimetro_copriferro= patches.Rectangle([cover, cover], width= b - 2*cover, height= h - 2*cover, fill=False, linestyle='--', linewidth=0.3, color='darkblue')
#ax.add_patch(perimetro_copriferro)

# staffa
perimetroEsterno_staffa= patches.Rectangle([cover, cover], width= b - 2*cover, height= h - 2*cover, fill=False, linestyle='--', linewidth=0.3, color='darkblue')
perimetroInterno_staffa= patches.Rectangle([cover+stirrups_dia, cover+stirrups_dia], width= b - 2*cover - 2*stirrups_dia, height= h - 2*cover - 2*stirrups_dia, fill=False, linestyle='--', linewidth=0.3, color='darkblue')
ax.add_patch(perimetroEsterno_staffa)
ax.add_patch(perimetroInterno_staffa)

# QUOTA ORIZZONTALE
ax.plot([-10, b+10], [h+40, h+40], color='black', linewidth=0.2)

# Tacche verticali agli estremi
ax.plot([0, 0], [h+24, h+64], color='black', linewidth=0.5)
ax.plot([b, b], [h+24, h+64], color='black', linewidth=0.5)
# Testo della quota
ax.text(b/2, h+30, f"{b}", ha='center', va='center', fontsize=5, color='black')

# QUOTA VERTICALE
ax.plot([-40, -40], [-10, h+10], color='black', linewidth=0.3)
# Tacche orizzontali agli estremi
ax.plot([-60, -20], [0, 0], color='black', linewidth=0.5)
ax.plot([-60, -20], [h, h], color='black', linewidth=0.5)
# Testo della quota
ax.text(-60, h/2, f"{h}", ha='center', va='center', fontsize=5, color='black', rotation=90)


ax.invert_yaxis()
ax.set_xlabel("Width [mm]")
ax.set_ylabel("Height [mm]")

ax.set_title(f"Section: {b}x{h}", fontsize=5, color="black", pad= 10)



#ax.set_title("Section with Reinforcement")

# Salvataggio su file PNG
fileImgGeo = os.path.join(sessionDir, f"imgGeo_{st.session_state.session_id}.png")
fig.savefig(fileImgGeo, dpi=300, bbox_inches="tight")

#st.pyplot(fig)
st.pyplot(fig, use_container_width=False)






