import streamlit as st

import pandas as pd
import csv
import os

# inizializza session_state
if 'newFlag' not in st.session_state:
    st.session_state.newFlag = 'none'
if 'dataGen' not in st.session_state:
    st.session_state['dataGen'] = []
if 'JAccount' not in st.session_state:
    st.session_state.JAccount = ""
if 'Project' not in st.session_state:
    st.session_state.Project = ""
if 'Location' not in st.session_state:
    st.session_state.Location = ""


if 'SectTag' not in st.session_state:
    st.session_state.SectTag = ""
if 'Item' not in st.session_state:
    st.session_state.Item = ""

   
if 'dataConfirmed' not in st.session_state:
    st.session_state.dataConfirmed = False  

if 'flagChanges' not in st.session_state:
    st.session_state.flagChanges = False  

# functions

#print ('newFlag = ', st.session_state.newFlag)
#---in caso non hai fatto alcuna scelta torna alla main page ---
if st.session_state.newFlag == "none":
    pagina = 'pages/0_🗂️Main.py'
    st.switch_page(pagina)
#---------------------------------------------

st.sidebar.info(st.session_state.newFlag)

if st.session_state.newFlag == "stored":
    # Titolo dell'applicazione
    st.title('📝 General Data - Stored project')
    st.session_state.flagChanges = False
    #st.subheader('Environmental data')

elif st.session_state.newFlag == "new":
    st.title('📝 General Data - New project')
    st.session_state.flagChanges = True


with st.expander("🆘 Help"):
    st.markdown("""
    - **What to do here?**  
      Input or change general data as: Job account, title and location, section tag, item.
               
    - **Before to go on ...**  
      Be sure the introduced values are correct and confirm them by ticking checkbox ***Data confirmed***. 
      
    - **further information:**  
      click on ℹ️ info button menu]
    """)

#leggi dati di input da DatiGenerali.csv
sessionDir = "sessions"
fileDatiGen = f"{sessionDir}/DatiGenerali_{st.session_state.session_id}.csv"

with open(fileDatiGen) as file_input:
    dfgen = pd.read_csv(file_input)   # lettura file e creazione
    dfgen.drop(dfgen.columns[dfgen.columns.str.contains('unnamed', case= False)], axis=1, inplace= True)

st.session_state.JAccount = dfgen.loc[0,'JAccount']
st.session_state.Project = dfgen.loc[0,'Project']
st.session_state.Location = dfgen.loc[0,'Location']
st.session_state.SectTag = dfgen.loc[0,'SectTag']
st.session_state.Item = dfgen.loc[0,'Item']

#col1, col2 = st.columns([1,3])
JAccount = st.text_input('Job Account', value = st.session_state.JAccount)
Project = st.text_input('Project', value = st.session_state.Project)
Location = st.text_input('Location', value = st.session_state.Location)
SectTag = st.text_input('Section Tag', value= st.session_state.SectTag )
Item = st.text_input('Item', value= st.session_state.Item )


col1, col2, col3 = st.columns([1,1,1])
checkbox_state = col1.checkbox('Data confirmed', value=st.session_state.dataConfirmed)
#checkbox_state = col1.checkbox('Data confirmed')
# Pulsante per acquisire i dati
if checkbox_state:
    #os.remove("files/DatiGenerali.csv")
    st.session_state['dataGen'] = [{
        'JAccount': JAccount,
        'Project': Project,
        'Location': Location,
        'SectTag': SectTag,
        'Item': Item

    }]
    st.session_state.dataConfirmed = True
    st.success('Data confirmed and stored successfully! Double click to untick checkbox.')
    df = pd.DataFrame(st.session_state['dataGen'])
    st.subheader('Summary of general data')
    st.dataframe(df, hide_index= True)
    
    df.to_csv(fileDatiGen)   # salva dati generali su sessions/DatiGenerali_idUser.csv 
       

else:
    
    st.session_state.dataConfirmed = False
    st.warning('Data not confirmed! Double click to tick checkbox and confirm input data.')
    
    #df = pd.DataFrame(st.session_state['dataGen'])
    #st.subheader('Summary of general data')
    #st.dataframe(df, hide_index= True)
    #df.to_csv("files/DatiGenerali.csv")   # salva dati su DatiPiping      
# Ritorno a Main
if col3.button('Back to Main'):
    st.switch_page('pages/0_🗂️Main.py')
# Ritorno a Main




