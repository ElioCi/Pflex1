import streamlit as st
import csv
import pandas as pd

def indicePosizione(filecsv, colonna, valore_cercato):
    # Carica il CSV in un DataFrame

    df = pd.read_csv(filecsv, delimiter= ';')

    # Specifica la colonna e il valore che stai cercando
    #colonna = 'NomeColonna'
    #valore_cercato = 'ValoreDaCercare'

    # Trova l'indice della riga in cui il valore è presente nella colonna
    indici = df.index[df[colonna] == valore_cercato].tolist() # solo il primo indice
    
    indicePosizione = indici[0] # solo il primo indice

    # Visualizza gli indici trovati
    if indici:
        print(f"Indice/i trovato/i per '{valore_cercato}' nella colonna '{colonna}': {indici}")
    else:
        print(f"Valore '{valore_cercato}' non trovato nella colonna '{colonna}'.")

    # routine per trovare l'indice associato ad un elemento in una colonna di un file csv
    return indicePosizione

def trovaDesc(fileInscsv, codice):
    df = pd.read_csv(fileInscsv, delimiter=';')
    
    # Cerca la descrizione corrispondente
    if codice:
        description = df.loc[df['code'] == codice, 'Desc'].values[0]

    return description


# Definire una funzione per applicare il CSS
def change_widget_style(bg_color1, bg_color2):
    # Stili personalizzati con il colore di background
    st.markdown(
        f"""
        <style>
        /* Cambiare il colore di background per text_input e number_input */
        input[type="text"] {{
            background-color: {bg_color1};
            color: black;
        }}
        input[type="number"] {{
            background-color: {bg_color2};
            color: black;

        }}
        /* Cambiare il colore di background per selectbox */
        .stSelectbox > div[data-baseweb="select"] > div {{
            background-color: {bg_color2};
            color: black;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )
    return

def reset_widget_style():
    st.markdown(
        """
        <style>
        /* Reimposta il colore di sfondo su quello predefinito */
        input[type="text"], input[type="number"], .stSelectbox > div[data-baseweb="select"] > div {
            background-color: initial; /* Reimposta al valore predefinito */
            color: initial;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return
# Bottone per cambiare colore dei widget
#if st.button("Cambia colore widget"):
#    change_widget_style("#FF6347")  # Colore Tomato
