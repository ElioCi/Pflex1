import streamlit as st
import json
import uuid
import shutil
import os
import time, glob

from urllib.parse import urlparse, parse_qs
import requests
import jwt
from jwt import PyJWTError


st.set_page_config(page_title="Pflex01 App", page_icon="🎛️")

def cleanup_sessions(session_dir="sessions", max_age_hours=6):   #24
    now = time.time()
    for f in glob.glob(f"{session_dir}/*"):
        if os.stat(f).st_mtime < now - max_age_hours * 3600:   #*3600   
            os.remove(f)


PAGE_KEY = "visited_immissione_dati"

if st.session_state.get(PAGE_KEY):
    st.info("🎛️ Pflex: Reinforced concrete MN-Domain     --- * Please select a menu item * --- ")
    st.info(f"Session assigned ID = {st.session_state.session_id}")
    st.stop()

st.session_state[PAGE_KEY] = True


# generazione di un ID sessione di lavoro che resta fisso per tutta la durata della sessione
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
#print ("id = ", st.session_state.session_id)

if 'prot' not in st.session_state or not st.session_state.prot:
    st.session_state.prot = False
    
# pulisci cartella sessions ogni 6 ore
cleanup_sessions()


# 👮🪪 Parte relativa alla gestione del Token: lettura da indirizzo e memorizzazione in file json
# Chiave segreta utilizzata per firmare il token
SECRET_KEY = 'EC1'

# Funzione per verificare il token
def verify_token(token):
    try:
        # Decodifica e verifica il token
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token
    except PyJWTError as e:
        # Token non valido o scaduto
        st.error("Token invalid or expired: " + str(e))
        return None

# Acquisisci il token dai parametri dell'URL
# query_params = st.experimental_get_query_params()
query_params = st.query_params

#token = query_params.get("token", [None])[0]
token = query_params.get("token", [None])

if token:
    # Verifica il token
    sessionDir = 'sessions'
    fileProt = f"{sessionDir}/prot_status_{st.session_state.session_id}.json"
    decoded_token = verify_token(token)
    if decoded_token:
        st.success("Authorized access!")
        st.write("Token decoded:", decoded_token)
        st.session_state.prot = True
        with open(fileProt, "w") as file:
            json.dump({"prot": st.session_state.prot}, file)
            # Inserisci qui il codice dell'applicazione Streamlit
    else:
        st.error("Access denied: token invalid or expired.")
        st.info("Go back to the 'enginapps' blog and launch again the app")
        st.session_state.prot = False
        with open(fileProt, "w") as file:
            json.dump({"prot": st.session_state.prot}, file)
        st.stop()
else:
    st.error("No token provided, access denied.")
    with open(fileProt, "w") as file:
            json.dump({"prot": st.session_state.prot}, file)
    st.stop()

# - 🪪 -------------------Fine gestione Token---------------------------


st.markdown("---")
st.markdown("<h1 style='text-align: center;'>🎛️ PFlex: Reinforced concrete section</h1>", unsafe_allow_html=True)
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.page_link("https://enginapps.it", label="www.enginapps.it", icon="🏠")
#st.write("Application Start")

# Salva in un file temporaneo
sessionDir = "sessions"

st.session_state.prot = True
fileProt = f"{sessionDir}/prot_status_{st.session_state.session_id}.json"
with open(fileProt, "w") as file:
    json.dump({"prot": st.session_state.prot}, file)

# copia i file base nella cartella sessions solo la prima volta

# File sorgenti (templates)
src_files = [
    "templates/DatiGenerali.csv"
#    "templates/piping_groups.csv",
#    "templates/piping_report.csv",
#    "templates/pressures_by_temperature.csv",
#    "templates/Temperatures.csv",
#    "templates/groupRatingInfo.json"
]

# File di destinazione per la sessione
dst_files = [
    f"{sessionDir}/DatiGenerali_{st.session_state.session_id}.csv"
#    f"{sessionDir}/piping_groups_{st.session_state.session_id}.csv",
#    f"{sessionDir}/piping_report_{st.session_state.session_id}.csv",
#    f"{sessionDir}/pressures_by_temperature_{st.session_state.session_id}.csv",
#    f"{sessionDir}/Temperatures_{st.session_state.session_id}.csv",
#    f"{sessionDir}/groupRatingInfo_{st.session_state.session_id}.json"
]

# Copia i file solo se non esistono già (evita doppioni alla ricarica)
for src, dst in zip(src_files, dst_files):
    if not os.path.exists(dst):
        shutil.copy(src, dst)


# con la seguente riga aggiunta il programma non eseguirà mai button_html... . 
st.switch_page("pages/0_🗂️Main.py")  # al primo run vai sulla pagina main, al successivo fermati 

# Utilizza un link HTML per fare il redirect a "pages/main.py"
# Codice HTML e CSS per il pulsante con hover
button_html = """
    <style>
    .button-container {
        display: flex;
        justify-content: center; /* Centra il pulsante orizzontalmente */
        align-items: center; /* Centra il pulsante verticalmente, se necessario */
        height: 20vh; /* Imposta l'altezza del contenitore al 30% dell'altezza della vista */
    }
    .hover-button {
        padding: 10px 30px;
        font-size: 16px;
        cursor: pointer;
        color: #000000;
        background-color: #4CAF49; /* Colore iniziale del pulsante */
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s ease, transform 0.3s ease; /* Transizioni per un effetto fluido */
    }

    /* Effetto hover */
    .hover-button:hover {
        background-color: #5CD458; /* Colore al passaggio del mouse */
        color: white; 
        transform: scale(1.05); /* Leggero ingrandimento */
    }
    </style>
    <div class="button-container">
    <a href="Main" target="_self">
        <button class="hover-button">GoTo main page</button>
    </a>
"""

#st.session_state.update({'prot': True})
#st.rerun()

# Inserisci il bottone nella pagina
st.markdown(button_html, unsafe_allow_html=True)
 
print ('prot', st.session_state.prot)

