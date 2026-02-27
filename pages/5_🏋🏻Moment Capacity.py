from ast import For
import streamlit as st
import pandas as pd
from reportPDF import UpdateReportPdf
import json
import os
import base64
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()


# -------------------------------
# Inizializzazione
# -------------------------------
if "export_results" not in st.session_state:
    st.session_state.export_results = []
if "auto_calc_done" not in st.session_state:
    st.session_state.auto_calc_done = False

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

if "layers" not in st.session_state:
    st.session_state["layers"] = []

if "axial" not in st.session_state:
    st.session_state["axial"] = 0.0  # default axial load
if "momentX" not in st.session_state:
    st.session_state["momentX"] = 0.0  # default moment around X axis
if"momentY" not in st.session_state:
    st.session_state["momentY"] = 0.0  # default moment around Y axis

if "Nmin" not in st.session_state:
    st.session_state["Nmin"] = 0.0
if "Mxmax" not in st.session_state:
    st.session_state["Mxmax"] = 0.0


# --------------------------------------------------------------------
# Lettura dati dinamici dalla sessione ID utente e statici da files
# --------------------------------------------------------------------
sessionDir = "sessions"
fileConcrete = os.path.join(sessionDir, f"concrete_{st.session_state.session_id}.csv")
fileSteel = os.path.join(sessionDir, f"steel_{st.session_state.session_id}.csv")
fileDatiGeo = f"{sessionDir}/DatiGeo_{st.session_state.session_id}.csv"
fileDatiLoads = f"{sessionDir}/DatiLoad_{st.session_state.session_id}.csv"

if os.path.exists(fileConcrete) and os.path.exists(fileSteel):
    df_concrete = pd.read_csv(fileConcrete, sep=',')
    df_steel = pd.read_csv(fileSteel, sep=',')
else:
    st.warning("No Material Data defined. Please go to the 'Material_Properties' to define them.") 
    st.stop()

st.session_state.fcd = df_concrete.loc[0,'fcd']
st.session_state.fyd = df_steel.loc[0,'fyd']

fcd = st.session_state.fcd
fyd = st.session_state.fyd  

if os.path.exists(fileDatiGeo):
    with open(fileDatiGeo) as file_input_geo:
        dfgeo = pd.read_csv(file_input_geo)   # lettura file e creazione
        dfgeo.drop(dfgeo.columns[dfgeo.columns.str.contains('unnamed', case= False)], axis=1, inplace= True)
    # Lettura variabili geometriche
    st.session_state.base = dfgeo.loc[0,'base']
    st.session_state.height = dfgeo.loc[0,'height']
    st.session_state.cover = dfgeo.loc[0,'cover']
    st.session_state.stirrups_dia = dfgeo.loc[0,'stirrups_dia']

else:
       st.warning("No Geomety Data defined. Please go to the 'Section_Geometry' to define them.") 
       st.stop()

b = st.session_state.base
h = st.session_state.height
c = st.session_state.cover
s_dia = st.session_state.stirrups_dia

# Reinforcement Layers
# === Caricamento layers di armatura esistenti se presenti ===

fileReinfLayers = f"{sessionDir}/Reinf_layers_{st.session_state.session_id}.csv"

if os.path.exists(fileReinfLayers):
    df_layers_caricati = pd.read_csv(fileReinfLayers)
    num_layers = len(df_layers_caricati)
    #st.markdown("### Existing Reinforcement Layers")
    #st.dataframe(df_layers_caricati, hide_index=True)
    gruppi_dati = df_layers_caricati.to_dict(orient="records")  
else:
    #df_layers_caricati = pd.DataFrame()
    st.info("No existing reinforcement layer found. Please go to the 'Section_Geometry' to define reinforcemente layers.")
    #num_layers = 0
    st.stop()

#Lettura dati di carico
if os.path.exists(fileDatiLoads):
    df_loads = pd.read_csv(fileDatiLoads, sep=',')
else:
    df_loads = pd.DataFrame()

# df_loads = pd.read_csv(fileDatiLoads, sep=',') if os.path.exists(fileDatiLoads) else pd.DataFrame()

if df_loads.columns.size > 0:
    df_loads.columns = df_loads.columns.str.strip() # Rinomina eventuali colonne con spazi o maiuscole df.columns = df.columns.str.strip()
    Nmin = df_loads['axial'].min() if 'axial' in df_loads.columns else 0.0
    Nmax = df_loads['axial'].max() if 'axial' in df_loads.columns else 0.0
    Mx_min = df_loads['momentX'].min() if 'momentX' in df_loads.columns else 0.0
    Mx_max = df_loads['momentX'].max() if 'momentX' in df_loads.columns else 0.0  
    My_min = df_loads['momentY'].min() if 'momentY' in df_loads.columns else 0.0
    My_max = df_loads['momentY'].max() if 'momentY' in df_loads.columns else 0.0

    Mx_abs_max = max(abs(Mx_min), abs(Mx_max))

else:
    Nmin = 0.0
    Mx_abs_max = 0.0


st.session_state.Nmin = Nmin
st.session_state.Mxmax = Mx_abs_max

# -------------------------------
# Interfaccia Streamlit
# -------------------------------
st.title("🏋🏻 Bending Moment Capacity")
with st.expander("🆘 Help"):
    st.markdown("""
    - **What to do here?**  
      This module gives you the value of the bending moment capacity on the basis of the minimum Axial Load and maximum Moment_X. 
      You can change these values manually, and runnning again the calculation. 
                                                         
    """)

# memorizzazione carichi usati per calcolo MR in "sessions"
file_LoadsMR = f"{sessionDir}/loadsMR_{st.session_state.session_id}.csv"

col1, col2 = st.columns(2)
with col1:
    Nd_kN = st.number_input("Axial Load [kN] (+ compression)", value=st.session_state.Nmin, step=1.0)

with col2:
    Md_kNm = st.number_input("Bending Moment [kN·m] (+ compressed upper fibres )", value=st.session_state.Mxmax, step=1.0)

df_loadsMR = pd.DataFrame({
    "Nd": [Nd_kN],
    "Md": [Md_kNm]

})
df_loadsMR.to_csv(file_LoadsMR, index=False)

st.session_state["layers"] = []

for idx, row in df_layers_caricati.iterrows():
    #st.markdown(f"### Layer {idx + 1}")
    dia = row['dia']
    num_bars = row['num_bars']  
    x_first = row['x_first']
    y_layer = row['y_layer']
    rebar_spacing = row['rebar_spacing']
    As_bar = math.pi * (dia ** 2) / 4
    As_layer = As_bar * num_bars

    #st.write(f"- Diameter: {dia} mm")
    #st.write(f"- Number of bars: {num_bars}")
    #st.write(f"- Area of steel in layer: {As_layer:.2f} mm²")       
    #st.write(f"- y_layer: {y_layer} mm from the top side")


    # add to st.session_state.layer
    st.session_state["layers"].append({
        "x_start": x_first,
        "y_layer": y_layer,
        "n_bars": num_bars,
        "diameter": dia,
        "spacing": rebar_spacing,
        "As_bar [mm²]": round(As_bar, 0),
        "As_layer [mm²]": round(As_layer, 0)
        })

st.markdown("Reinforcement Layers Summary") 
st.table(st.session_state["layers"])

# 🔹 Calcolo del valore massimo di y
y_max = df_layers_caricati["y_layer"].max()
st.write(f"**d = {y_max:.1f} mm**")

total_As = sum(layer["As_layer [mm²]"] for layer in st.session_state["layers"])
st.write(f"Total steel area As = {total_As:.2f} mm²")

if y_max < c + s_dia / 2:
    st.error("Warning: Maximum reinforcement layer is below the concrete cover. Please check reinforcement layers definition.")
    st.stop()
    
if total_As == 0:
    st.error("Warning no steel area defined. Please define at least one reinforcement layer with bars.")
    st.stop()   



# ---------------------------------------------------
# 6️⃣ CALCULATION
# ---------------------------------------------------

Md = Md_kNm * 1000      # Newton*m
Nd = Nd_kN * 1000       # Newton 

# Calcolo Compressione massima
NdmaxC = fcd * b * h  # in Newton
if Nd > NdmaxC:
    st.error(f"Axial Load **Nd = {Nd/1000:.2f} kN** exceeds maximum compression capacity **NdmaxC = {NdmaxC/1000:.2f} kN**")
    st.stop()   

# Calolo trazione massima
NdmaxT = - fyd * sum(layer["As_layer [mm²]"] for layer in st.session_state["layers"])  # in Newton
if Nd < NdmaxT:
    st.error(f"Axial Load **Nd = {Nd/1000:.2f} kN** exceeds maximum tension capacity **NdmaxT = {NdmaxT/1000:.2f} kN**")
    st.stop()   

df_loadsMR = pd.DataFrame({
    "Nd_calc": [Nd_kN],
    "Md_calc": [Md_kNm],
    "Nd_maxC": [NdmaxC/1000],
    "Nd_maxT": [NdmaxT/1000]

})



ec = 0.35  # Maximum concrete strain
esy = 0.2  # Steel yield strain
d = y_max

Xn1 = 0
Xn2 = 3000
cicli = 0

# ricalcola

while cicli < 51:
       
    Xn = (Xn1 + Xn2) / 2    # valore in mm
    #print("Ciclo n. , Xn", cicli, Xn)
    # calcolo deformazioni
    if Md >= 0:
        # calcolo delle deformazioni per i tre strati (caso positivo di Md)
        # Protezione: se Xn == 0 evito divisione per zero
        if Xn == 0:
            raise ValueError("Xn = 0 -> division by zero nei calcoli delle deformazioni.")
        
        for layer in st.session_state["layers"]:   
            y = layer["y_layer"]
            Af_lay = layer["As_layer [mm²]"]
            es_lay = ec * (Xn - y) / Xn
            if Af_lay == 0:
                es_lay = 0

            layer["es_layer"] = es_lay 
            #print("layer es_layer", y, layer["es_layer"])

    else:
        # caso Md < 0
        # Nota: nel tuo snippet originale usavi (H - Xn - d1)/Xn
        # ho usato h (minuscolo) e mantengo la stessa formula.
        # Se la formula di Xn cambia per Md<0, calcola Xn qui diversamente.
        if Xn == 0:
            raise ValueError("Xn = 0 -> division by zero nei calcoli delle deformazioni.")

        for layer in st.session_state["layers"]:
            y = layer["y_layer"]
            Af_lay = layer["As_layer [mm²]"]
            es_lay = -ec * (h - Xn - y) / Xn
            if Af_lay == 0:
                es_lay = 0
            
            layer["es_layer"] = es_lay

    # Calcolo tensioni acciaio
    NR1 = 0.0
    es_lay = 0.0
    Af_lay= 0.0 
    fs_lay= 0.0
    NAf_lay = 0.0
    Af_comp = 0.0 
    
    for layer in st.session_state["layers"]:
        es_lay = layer["es_layer"]
        Af_lay = layer["As_layer [mm²]"]
        y = layer["y_layer"]

        if abs(es_lay) > esy :
            fs_lay = fyd
        else:
            fs_lay = fyd * es_lay / esy

        if es_lay > 0:
            fs_lay = abs(fs_lay)
        else:
            fs_lay = -abs(fs_lay)

            
        # calcolo N parziali

        if Xn < 1.25 * h :
            Ncls = fcd * 0.8 * Xn * b  # in Newton
        else:
            Ncls = fcd * h * b   # in Newton    


        NAf_lay = Af_lay * fs_lay

        NR1 = NR1 + NAf_lay

        layer["NAf_layer"] = NAf_lay
        


    NR = Ncls + NR1

    #print("fcd, h, b, Ncls", fcd, h, b, fcd*h*b)
    #print("cicli, Ncls, NR1, NR, Nd", cicli, Ncls, NR1, NR, Nd)

    if NR > Nd :
        Xn1 = Xn1
        Xn2 = Xn
    else:
        Xn1 = Xn
        Xn2 = Xn2

    cicli = cicli + 1
  
# Calcola M parziali rispetto ad H/2

if Xn > 1.25 * h :
    Mcls = 0
else:
    if Md >= 0 :
        Mcls = Ncls * ((h / 2 - 0.4 * Xn)) / 1000
    else:
        Mcls = -Ncls * ((h / 2 - 0.4 * Xn)) / 1000

MAf = 0.0
for layer in st.session_state["layers"]:
    NAf_lay = layer["NAf_layer"]
    y_lay = layer["y_layer"]
    
    
    MAf_lay = NAf_lay * ((h / 2) - y_lay) / 1000
    MAf = MAf + MAf_lay
    #print("NAf_lay, MAf_lay", NAf_lay, MAf_lay, MAf)

MR = Mcls + MAf

# Trova il layer con la massima tensione nel ferro
layer_min = min(st.session_state["layers"], key=lambda layer: layer["NAf_layer"])
# Estrai i campi massima tensione ferro
NAf_lay_min = layer_min["NAf_layer"]        # minimo NAf_layer in N (tensione)
As_Nmin = layer_min["As_layer [mm²]"]       # area corrispondente al layer con Naf min (tensione)
sigma_s_maxTens = NAf_lay_min / As_Nmin     # in MPa

# Trova il layer con la massima compressione nel ferro
layer_max = max(st.session_state["layers"], key=lambda layer: layer["NAf_layer"])
# Estrai i campi massima compressione ferro
NAf_lay_max = layer_max["NAf_layer"]        # massimo NAf_layer in N (compressione)
As_Nmax = layer_max["As_layer [mm²]"]       # area corrispondente al layer con Naf max (compressione)
sigma_s_maxComp = NAf_lay_max / As_Nmax     # in MPa

#print ("---")
#print ("Mcls, MAf, MR", Mcls, MAf, MR)

st.write(f"Neutral axis: **Xn = {Xn:.2f}** mm from top side")

st.write(f"Maximum compression in steel layer: **Nsc = {abs(NAf_lay_max):.2f}** N")
st.write(f"Maximum compression stress in steel: **fsc = {abs(sigma_s_maxComp):.2f}** MPa")

st.write(f"Maximum tension in steel layer: **Nst = {abs(NAf_lay_min):.2f}** N")
st.write(f"Maximum tension stress in steel: **fst = {abs(sigma_s_maxTens):.2f}** MPa")

df_loadsMR["Xn"] = Xn
df_loadsMR["fcc"] = fcd
df_loadsMR["Nsc"] = - abs(NAf_lay_max)
df_loadsMR["fsc"] = -abs(sigma_s_maxComp)
df_loadsMR["Nst"] = abs(NAf_lay_min)   
df_loadsMR["fst"] = abs(sigma_s_maxTens) 
df_loadsMR["MR"] = MR/1000  

df_loadsMR.to_csv(file_LoadsMR, index=False)
#st.write(f"Calculated Moment  **MR= {MR/1000:.2f}** kN*m")

#st.markdown(f"Bending Moment Capacity: **MR = {MR/1000:.2f}** kN·m")
#st.markdown(
#    f"Bending Moment Capacity: **<span style='color:red;'>MR = {MR/1000:.2f} kN·m</span>**",
#    unsafe_allow_html=True
#)
st.markdown(
    f"""
    <div style="
        background-color:#ffe6e6;
        padding:10px 15px;
        border-left:6px solid #cc0000;
        border-radius:5px;
        font-size:18px;
        font-weight:bold;
        color:#990000;">
        Bending Moment Capacity: MR = {MR/1000:.2f} kN·m
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# 7️⃣ VISUALIZZAZIONE: ASSE NEUTRO E DIAGRAMMA TENSIONI
# ---------------------------------------------------

st.header("Stress Diagram")

# Imposta figura con due sottotrame affiancate, ordinate condivise
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
plt.subplots_adjust(wspace=0.25)

# Limiti verticali comuni
ax1.set_ylim(0, h)
ax2.set_ylim(0, h)

# ===================================================
# 🔸 SEZIONE SINISTRA — vista frontale con campitura verde
# ===================================================

ax1.set_aspect('equal')
#ax2.set_aspect('equal')
#ax1.set_adjustable('box')   # aggiunto per mantenere proporzioni corrette
#ax2.set_adjustable('box')   # aggiunto per mantenere proporzioni corrette


ax1.set_xlim(-50, b + 50)

# Contorno sezione
ax1.add_patch(patches.Rectangle((0, 0), b, h, fill=False, linewidth=1.5, edgecolor='black'))

# Area compressa del calcestruzzo (0.8 * Xn, tutta la larghezza)
rect_height = 0.8 * Xn
ax1.add_patch(patches.Rectangle((0, 0), b, rect_height,
                                facecolor='lightgreen', alpha=0.6, edgecolor='green'))

# Barre di armatura
for layer in st.session_state["layers"]:
    x_start = layer["x_start"]
    y = layer["y_layer"]
    dia = layer["diameter"]
    n_bars = int(layer["n_bars"])
    spacing = layer["spacing"]
    for i in range(n_bars):
        xi = x_start + i * spacing
        if xi + dia / 2 > b:
            continue
        ax1.add_patch(plt.Circle((xi, y), dia / 2, color='red'))

# Asse neutro (sporgente)
if Md >= 0:
    ax1.plot([-30, b + 30], [Xn, Xn], linestyle='--', color='blue', linewidth=2)
    ax1.text(b / 2, Xn - 1, f"Neutral Axis\nXn = {Xn:.1f} mm", color='blue',
            ha='center', va='bottom', fontsize=9)
if Md < 0:
    ax1.plot([-30, b + 30], [h-Xn, h-Xn], linestyle='--', color='blue', linewidth=2)
    ax1.text(b / 2, (h-Xn) - 1, f"Neutral Axis\nXn = {h-Xn:.1f} mm", color='blue',
            ha='center', va='bottom', fontsize=9)

# Linee di copriferro (facoltative)
ax1.add_patch(patches.Rectangle((c, c), b - 2 * c, h - 2 * c,
                                fill=False, linestyle='--', color='gray'))

# Niente assi o ticks
ax1.axis('off')
ax1.set_title("Section View (Concrete compression in green)")

# ===================================================
# 🔸 SEZIONE DESTRA — diagramma delle tensioni
# ===================================================
xmax = max(abs(fyd), abs(fcd)) * 1.2
ax2.set_xlim(-xmax, xmax)

# Area compressa del calcestruzzo (compressione negativa)
ax2.add_patch(patches.Rectangle((-fcd, 0), fcd, rect_height,
                                facecolor='lightgreen', alpha=0.6, edgecolor='green'))
#ax2.text(-fcd / 2, rect_height / 2, "Concrete\ncompression", color='green',ha='center', va='center', fontsize=9)

ax2.text(-fcd-20, rect_height / 2, f"{-fcd:.2f}", color='green',
         ha='center', va='center', fontsize=8)

# Frecce per le tensioni nelle barre
for layer in st.session_state["layers"]:
    y = layer["y_layer"]
    sigma_s = layer["NAf_layer"] / layer["As_layer [mm²]"]

    # Convenzione segno: trazione positiva (→), compressione negativa (←)
    ax2.arrow(0, y, -sigma_s, 0, head_width=6, head_length=xmax / 15,
              fc='red', ec='red', alpha=0.8, length_includes_head=True)
    ax2.text(-sigma_s * 0.7, y - 7, f"{-sigma_s:.1f}",
             color='red', fontsize=8,
             ha='right' if sigma_s < 0 else 'left', va='center')

# Asse neutro
ax2.axhline(Xn, color='blue', linestyle='--', linewidth=1.5)
ax2.text(0, Xn - 1, "Neutral axis", color='blue', ha='center', va='bottom', fontsize=9)

# Asse delle tensioni
ax2.axvline(0, color='black', linewidth=0.8)

# Impostazioni grafiche
ax2.invert_yaxis()
ax2.set_xlabel("Stress [MPa]")
ax2.set_ylabel("Height [mm]")
ax2.set_title("Stress Diagram (tension →, compression ←)")

fig2.align_ylabels()
plt.tight_layout()

# Salvataggio su file PNG o JPG
fileImg1 = os.path.join(sessionDir, f"img1_{st.session_state.session_id}.png")
fig2.savefig(fileImg1, dpi=300, bbox_inches="tight")
# oppure JPG
#fig2.savefig("stress_diagram.jpg", dpi=300, bbox_inches="tight")


st.pyplot(fig2)



