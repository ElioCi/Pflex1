import streamlit as st
import pandas as pd
import csv

# Titolo dell'applicazione
#st.markdown(
#    "<hr style='border:1px solid #007BFF;margin:5px 0;'>",
#    unsafe_allow_html=True
#)

st.set_page_config(page_title="Materials")
st.title('🧱Materials')
#st.subheader('acc. to ASME B31.3 - 2024')
#st.markdown(
#    "<hr style='border:1px solid #007BFF;margin:5px 0;'>",
#    unsafe_allow_html=True
#)

st.markdown("---")



#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()
    

#st.markdown(
#    "<h6 style='color:red; font-weight:normal;'>=== Read-only access for viewing and querying ===</h6>",
#    unsafe_allow_html=True
#)


# Carica file csv concrete e steel
df_concrete = pd.read_csv("files/concrete.csv", sep=";")
df_steel = pd.read_csv("files/steel.csv", sep=";")


#col1, col2, col3 = st.columns([2,1,1])


# --- STEP 1: Seleziona materiale concrete---
#st.header("1️⃣ Concrete Properties")
concrete = df_concrete["Class"].dropna().unique()
concrete_scelto = st.selectbox("1️⃣ Select concrete class:", concrete, index=1)

# --- STEP 2: Estrai la riga corrispondente ---
riga = df_concrete[df_concrete["Class"] == concrete_scelto]

fck = riga["fck"].values[0]
ec1 = riga["ec1"].values[0]
ecu = riga["ecu"].values[0]

# coefficienti gamma_c e alfa_cc

#gamma_c = st.markdown("$\gamma$c = 1.5")
#alfa_cc = st.markdown("$\\alpha$cc = 0.85")


# calcolo fcd
gamma_c = 1.5
alpha_cc = 0.85
fcd = alpha_cc*fck / gamma_c 

Ec = 22000 * ((8+fck)/10)**0.3  # modulo di elasticità del calcestruzzo in MPa  


#st.markdown(f"$f$cd = {fcd:.2f} MPa")  
#st.markdown(f"$E$c = {Ec:.2f} MPa")  

# memorizzazione riga con concrete selezionato in "sessions"
sessionDir = "sessions"
file_concrete = f"{sessionDir}/concrete_{st.session_state.session_id}.csv"

# --- Creazione DataFrame con valori originali + calcolati ---
ec1 = ec1.replace(",",".")
ecu = ecu.replace(",",".")  
df_out = pd.DataFrame({
    "Class": [concrete_scelto],
    "Rck": [riga["Rck"].values[0]],
    "fck": [fck],
    "ec1": [ec1],
    "ecu": [ecu],
    "gamma_c": [gamma_c],
    "alpha_cc": [alpha_cc],
    "fcd": [fcd],
    "Ec": [Ec]
})

st.markdown ("")
st.markdown("Selected concrete class and associated properties:")
st.dataframe(df_out, hide_index=True)
# --- Salvataggio nel file CSV della sessione ---
df_out.to_csv(file_concrete, sep=",", index=False)


st.markdown ("---")
# --- STEP 2: Seleziona materiale steel---
# Steel properties 

steel = df_steel["Tag"].dropna().unique()
steel_scelto = st.selectbox("2️⃣ Select steel type:", steel, index= 3 )

# --- Estrai la riga corrispondente ---
riga_st = df_steel[df_steel["Tag"] == steel_scelto]

ftk = riga_st["ftk"].values[0]
fyk = riga_st["fyk"].values[0]
Es = riga_st["Es"].values[0]
esd = riga_st["esd"].values[0]

# coefficienti gamma_s
gamma_s = 1.15

# calcolo fyd
fyd = fyk / gamma_s

# memorizzazione riga con steel selezionato in "sessions"
file_steel = f"{sessionDir}/steel_{st.session_state.session_id}.csv"

# --- Creazione DataFrame con valori originali + calcolati ---
esd = esd.replace(",",".")  
df_steel_out = pd.DataFrame({
    "Tag": [steel_scelto],
    "ftk": [ftk],
    "fyk": [fyk],
    "Es": [Es],
    "esd": [esd],
    "gamma_s": [gamma_s],
    "fyd": [fyd]
})

st.markdown ("")
st.markdown("Selected steel type and associated properties:")
st.dataframe(df_steel_out, hide_index=True)
# --- Salvataggio nel file CSV della sessione ---
df_steel_out.to_csv(file_steel, sep=",", index=False)
