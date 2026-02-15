import streamlit as st
import pandas as pd
import csv
from fpdf import FPDF
from create_table_fpdf2 import PDF
import re
import json
import os
from datetime import datetime

#from table_function import create_table

def UpdateReportPdf():
    # create FPDF object
    # layout ('P', 'L')
    # unit ('mm', 'cm', 'in')
    # format ('A3', 'A4 (default), 'A5', 'Letter', (100, 150))

    # Definizione dei parametri
    title = "Reinforced Concrete Section Check"
    orientation = 'P'
    unit = 'mm'
    format = 'A4'
    
    '''
    class PDF(FPDF):
        def header(self):
            #logo
            self.image('assets/rep.jpg', 15,5,12 )
            self.set_font('helvetica', 'B', 15)
            # padding
            #self.cell(80)
            # title
            
            title_w = self.get_string_width(title)
            doc_w = self.w
            
            self.set_x((doc_w - title_w)/2)

            self.cell(10,10, title, border=False, new_y= 'NEXT', align='L')
            # line breack
            self.set_draw_color(255,0,0)
            self.line(10,5,200,5)
            self.line(10,25,200,25)

        # page footer
        def footer(self):
            # set position of the footer 
            doc_w = self.w 
            self.set_xy(doc_w/2,-15)
            self.set_font('helvetica', 'I', 10)
            # page number
            self.cell(10, 10, f'Page {self.page_no()}/{{nb}}')  

    '''

    # Creazione del PDF con parametri passati
    pdf = PDF(orientation=orientation, unit=unit, format=format, title=title)
    #pdf = PDF('L', 'mm', 'A3')

    #get total pages
    pdf.alias_nb_pages()

    # Add a page
    pdf.add_page()
    # specify font ('times', 'courier', 'helvetica', symbol', zpfdingbats')
    # 'B'(bold), 'U'(underlined), 'I'(italic), ''(regular), combination(i.e.,('BU'))
    #pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(0,0,0)

    # add text
    # w= width
    # h= height
    # txt = your text
    # ln (0 False; 1 True - move cursor down to next line)
    # border (0 False; 1 True - add border around cell)
    pdf.set_xy(10, 30)
    #pdf.cell(120, 100, 'Hello World', new_y= 'NEXT', border=True)
    pdf.set_font('times', '', 12)

    
    textDateTime = f"Work Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    pdf.cell(80,6, textDateTime, ln=1, border=False)
    textLine = "For tags & units please refer to last page of this report."
    pdf.cell(80,6, textLine, ln=1, border=False)


    #leggi Units da Units
    with open('files/tags.csv') as file_units:
        readerUnits = csv.reader(file_units, delimiter = ";")
        #dfUnits = [rowUn for rowUn in readerUnits]
    
        # Ottieni le intestazioni (prima riga)
        headerUnits = next(readerUnits)
        
        # Modifica la prima colonna se il nome è 'Unnamed: 0.1'
        headerUnits[0] = re.sub(r'[^A-Za-z0-9]+', '_', headerUnits[0])
    
        # Leggi i dati rimanenti
        dfUnits = [rowUn for rowUn in readerUnits]
        
    # Aggiungi la nuova intestazione ai dati
    dfUnits.insert(0, headerUnits)
    
    #leggi dati di input da DatiGenerali.csv
    sessionDir = "sessions"
    fileDatiGen = os.path.join(sessionDir, f"DatiGenerali_{st.session_state.session_id}.csv")
    with open(fileDatiGen, newline='') as file_input:
        reader = csv.reader(file_input)
        data = [row[1:] for row in reader]

    fileDatiGeo = os.path.join(sessionDir, f"DatiGeo_{st.session_state.session_id}.csv")
    with open(fileDatiGeo, newline='') as geo_input:
        readerGeo = csv.reader(geo_input)
        #dataGeo = [row[1:] for row in readerGeo]
        dataGeo = [row for row in readerGeo]

    fileDatiConcrete = os.path.join(sessionDir, f"concrete_{st.session_state.session_id}.csv")
    with open(fileDatiConcrete, newline='') as concrete_input:
        readerConcrete = csv.reader(concrete_input, delimiter=",")
        #dataConcrete = [row[1:] for row in readerConcrete]
        # Leggo la prima riga: intestazioni di colonna 
        headers = next(readerConcrete)          # es: ["class", "ec1", "ecu", "fck", ...]
        # Sostituzione automatica "sigma" → "σ"
        headers = [h.replace("sigma", "σ") for h in headers]

        dataConcrete = []
        # Aggiungo la riga di testa 
        dataConcrete.append(headers)

        for row in readerConcrete:
            
            valori = []
            
            for idx, x in enumerate(row):
                
                header_col = headers[idx].strip().lower() # nome della colonna
                #print(f"Processing column: {header_col} with value: {x}")  # Debug print

                try:
                    # prova a convertire in float → se riesce, formatta a 2 decimali
                    num = float(x)
                    # ec1 ed ecu → 3 decimali 
                    if header_col in ("ec1", "ecu"):
                         valori.append(f"{num:.4f}") 
                    else: 
                        valori.append(f"{num:.2f}")

                except ValueError:
                    # se non è un numero, lascialo così com'è
                    valori.append(x)
            
            dataConcrete.append(valori)


    fileDatiSteel = os.path.join(sessionDir, f"steel_{st.session_state.session_id}.csv")
    
    with open(fileDatiSteel, newline='') as steel_input:
        readerSteel = csv.reader(steel_input, delimiter=",")
        #dataSteel = [row[1:] for row in readerSteel]
        headersSteel = next(readerSteel)          # es: ["class", "fyk", "esd", ...]    
        dataSteel = []
        dataSteel.append(headersSteel)

        for row in readerSteel:
            valori = []
            for idx, x in enumerate(row):  # salti la prima colonna come nel tuo codice
                header_col = headersSteel[idx].strip().lower()  # se la prima colonna contiene il nome del parametro
                
                try:
                    # prova a convertire in float → se riesce, formatta a 2 decimali
                    num = float(x)
                    if header_col.lower() in ("esd",):
                         #print(f"Formatting {header_col.lower()} with 4 decimals: {num:.4f}")  # Debug print
                         valori.append(f"{num:.4f}") 
                    else: 
                        valori.append(f"{num:.2f}")

                except ValueError:
                    # se non è un numero, lascialo così com'è
                    valori.append(x)
            dataSteel.append(valori)


    # Leggi i dati di input da 'Reinf_layers.csv'
    fileReinfLayers = os.path.join(sessionDir, f"Reinf_layers_{st.session_state.session_id}.csv")
    with open(fileReinfLayers) as layers_input:
        readerLayers = csv.reader(layers_input)
        
        # Ottieni le intestazioni (prima riga)
        header = next(readerLayers)
        
        # Modifica la prima colonna se il nome è 'Unnamed: 0.1'
        if header[0] == 'Unnamed: 0.1':
            header[0] = 'id'  # Rinomina la prima colonna
        
        
        # Leggi i dati rimanenti
        dataLayers = [rowlay for rowlay in readerLayers]

    # Aggiungi la nuova intestazione ai dati
    dataLayers.insert(0, header)


    # Leggi i dati di carico da 'DatiLoad.csv'
    fileDatiLoad = os.path.join(sessionDir, f"DatiLoad_{st.session_state.session_id}.csv")
    if os.path.exists(fileDatiLoad):
        with open(fileDatiLoad) as loads_input:
            readerLoads = csv.reader(loads_input)
            
            # Ottieni le intestazioni (prima riga)
            headerLoads = next(readerLoads)
            
            # Modifica la prima colonna se il nome è 'Unnamed: 0.1'
            if headerLoads[0] == 'Unnamed: 0.1':
                headerLoads[0] = 'id'  # Rinomina la prima colonna
            
            
            # Leggi i dati rimanenti
            dataLoads = [rowload for rowload in readerLoads]

        # Aggiungi la nuova intestazione ai dati
        dataLoads.insert(0, headerLoads)
    else:
        st.warning("No Load Data defined. Please go to the 'Loads' page to define them.")
        dataLoads = ["No Load Data"]  # Inizializza dataLoads come lista vuota se il file non esiste



    #leggi risultati da loadsMR.csv
    #with open('files/Output.csv') as output:
    fileloadsMR = os.path.join(sessionDir, f"loadsMR_{st.session_state.session_id}.csv")
    
    if os.path.exists(fileloadsMR):
        df_loadsMR = pd.read_csv(fileloadsMR)
        
        
        df_loadsMR_filtrato = df_loadsMR.drop(columns=['Nd_calc', 'Md_calc','Nd_maxC','Nd_maxT','Nsc','Nst'])
        # Arrotonda tutte le colonne numeriche a 2 decimali 
        df_loadsMR_filtrato = df_loadsMR_filtrato.round(2)
    
        
        # Sostituzione automatica "sigma" → "σ"

        headers = df_loadsMR_filtrato.columns.tolist()

        dataloadsMR = [headers] + df_loadsMR_filtrato.astype(str).values.tolist()
    else:
        st.warning("No Load Resistance data defined. Please go to the 'Moment Capacity' page to calculate them.")
        dataloadsMR = ["No stress information available"]  # Inizializza dataloadsMR come lista vuota se il file non esiste


    # Leggi SafetyFactors da 'SafetyFactors.csv'
    fileSafetyFactors = os.path.join(sessionDir, f"SafetyFactors_{st.session_state.session_id}.csv")

    if os.path.exists(fileSafetyFactors):
        with open(fileSafetyFactors) as sf_input:
            readerSF = csv.reader(sf_input)
           
            # Ottieni le intestazioni (prima riga)
            headerSF = next(readerSF)
            
            # Modifica la prima colonna se il nome è 'Unnamed: 0.1'
            if headerSF[0] == 'Unnamed: 0.1':
                headerSF[0] = 'id'  # Rinomina la prima colonna
            
            # Leggi i dati rimanenti
            dataSF = [rowSF for rowSF in readerSF]

        # Aggiungi la nuova intestazione ai dati
        dataSF.insert(0, headerSF)
    #pdf.cell(100,10, first_line, border=False , new_y= 'NEXT', align='L')
    #pdf.set_x(10)
    #pdf.cell(100,10, second_line, border=False, new_y= 'NEXT', align='L')
    #pdf.ln()


    #pdf.create_table(table_data = data,title='I\'m the first title', cell_width='even')

    pdf.set_text_color(30, 60, 200)     # blu per titoli
    pdf.set_font('times', 'B', 12)
    # add text
    # w= width 
    # h= height definisce l'altezza della cella (es: se h=10, la cella sarà alta 10mm)
    txt = "1.  I N P U T "    # testo da inserire
    # ln (0 False; 1 True - move cursor down to next line)
    # border (0 False; 1 True - add border around cell)
    # pdf.set_xy(10, 30) imposta la posizione del testo (x=10mm, y=30mm)
    pdf.ln()
    pdf.cell(80, 10, txt, ln=1, border=False)

    pdf.set_text_color(0, 0, 0)     # nero
    pdf.set_font('times', '', 12)
    #pdf.add_page()  # Aggiungi una nuova pagina prima di inserire la tabella dei dati generali
    #pdf.create_table(table_data = data,title='I\'m the first title', cell_width='even')
    pdf.create_table(table_data = data, title='General Data', cell_width='uneven')
    pdf.ln()
    #pdf.ln()

    #pdf.create_table(Concrete data)
    pdf.create_table(table_data = dataConcrete, title='Concrete Data', cell_width='uneven')
    pdf.ln()
    #pdf.ln()

    #pdf.create_table(Steel data)
    pdf.create_table(table_data = dataSteel, title='Steel Data', cell_width='uneven')
    pdf.ln()
    #pdf.ln()

    #pdf.create_table(Geometry data)
    pdf.create_table(table_data = dataGeo, title='Gemetry Data', cell_width='uneven')
    pdf.ln()
    #pdf.ln()

    #pdf.create_table(Reinforcement Layers data)
    pdf.create_table(table_data = dataLayers, title='Reinforcement Layers Data', cell_width='uneven')
    pdf.ln()
    #pdf.ln()
    # Aggiungi l'immagine img1 al PDF
    fileImgGeo = os.path.join(sessionDir, f"imgGeo_{st.session_state.session_id}.png")
    pdf.image(fileImgGeo, x=40, y=None, w=100, h=0)


    pdf.create_table(table_data = dataLoads, title='Combined Loads Data', cell_width='uneven')
    pdf.ln()

    pdf.add_page()
    pdf.set_xy(10, 30)
    #pdf.create_table(Output)
    pdf.set_text_color(30, 60, 200)     # blu per titoli
    pdf.set_font('times', 'B', 12)
    # add text
    # w= width 
    # h= height definisce l'altezza della cella (es: se h=10, la cella sarà alta 10mm)
    txt = "2.  O U T P U T "    # testo da inserire
    # ln (0 False; 1 True - move cursor down to next line)
    # border (0 False; 1 True - add border around cell)
    # pdf.set_xy(10, 30) imposta la posizione del testo (x=10mm, y=30mm)
    pdf.cell(80, 10, txt, ln=1, border=False)

    pdf.set_text_color(0, 0, 0)     # nero
    pdf.set_font('times', '', 12)

    pdf.create_table(table_data = dataloadsMR, title='Section Moment Capacity', cell_width='uneven')
    pdf.ln()
    #pdf.ln()
    # Aggiungi l'immagine img1 al PDF
    fileImg1 = os.path.join(sessionDir, f"img1_{st.session_state.session_id}.png")
    if os.path.exists(fileImg1):
        pdf.image(fileImg1, x=20, y=None, w=140, h=0)
    else:
        pdf.cell(80,10, "No image available imgage", ln=1, border=False)


    pdf.add_page()
    pdf.set_xy(10, 30)
    pdf.set_font('times', 'BU', 12)
    pdf.cell(80,10, 'Interaction Domains', ln=1, border=False)
    #pdf.ln()
    # Aggiungi l'immagine img1 al PDF
    fileImgMx = os.path.join(sessionDir, f"imgMx_{st.session_state.session_id}.png")
    if os.path.exists(fileImgMx):
        pdf.image(fileImgMx, x=30, y=None, w=140, h=0)
    else:
        pdf.cell(80,10, "No image available for imgMx", ln=1, border=False)
        
    fileImgMy = os.path.join(sessionDir, f"imgMy_{st.session_state.session_id}.png")
    if os.path.exists(fileImgMy):
        pdf.image(fileImgMy, x=30, y=None, w=140, h=0)
    else:
        pdf.cell(80,10, "No image available for imgMy", ln=1, border=False)

    pdf.add_page()
    pdf.set_xy(10, 30)
    pdf.cell(80,10, '... Interaction Domains', ln=1, border=False)
    #pdf.ln()    
    # Aggiungi l'immagine img1 al PDF
    fileImgMxMy = os.path.join(sessionDir, f"imgMxMy_{st.session_state.session_id}.png")
    if os.path.exists(fileImgMxMy):
        pdf.image(fileImgMxMy, x=30, y=None, w=140, h=0)
    else:
        pdf.cell(80,10, "No image available for imgMxMy", ln=1, border=False)

    if os.path.exists(fileSafetyFactors):
        pdf.create_table(table_data = dataSF, title='Section Check', cell_width='uneven')
    else:
        pdf.cell(80,10, "No safety factors available", ln=1, border=False)

    pdf.add_page()
    pdf.set_xy(10, 30)
    #pdf.create_table(Output)
    pdf.set_text_color(30, 60, 200)     # blu per titoli
    pdf.set_font('times', 'B', 12)
    # add text
    # w= width 
    # h= height definisce l'altezza della cella (es: se h=10, la cella sarà alta 10mm)
    txt = "3.  S Y M B O L S"    # testo da inserire
    # ln (0 False; 1 True - move cursor down to next line)
    # border (0 False; 1 True - add border around cell)
    # pdf.set_xy(10, 30) imposta la posizione del testo (x=10mm, y=30mm)
    pdf.cell(80, 10, txt, ln=1, border=False)

    pdf.set_text_color(0, 0, 0)     # nero
    pdf.set_font('times', '', 12)
    
    pdf.create_table(table_data = dfUnits, title='Tags and units', cell_width='uneven')
    pdf.ln()

    pdf.set_font('times', '', 10)

    # Percorso file
    #file_path = os.path.join(sessionDir, f"groupRatingInfo_{st.session_state.session_id}.json")
    #file_path = os.path.join("files", "groupRatingInfo.json")

    # Carica la lista
    #with open(file_path, "r") as f:
    #    ratingPipingGroup = json.load(f)

    # Scorri tutti i gruppi e stampa le variabili su righe consecutive
    '''
    for group_name, variabili in ratingPipingGroup.items():
        dia_min = variabili.get("dia_min")
        dia_max = variabili.get("dia_max")
        dia_rating = variabili.get("dia_rating")

        MAWP_min = variabili.get("MAWP_min")
        
        phrase = f"- RATING for material group {group_name} (NPS range from `{dia_min}` to `{dia_max}`) is determined by: NPS = {dia_rating}, MAWP = {MAWP_min:.2f} bar"
        #st.write(f"🔹 **{group_name}** ({D_min} - {D_max}) Rating = {MAWP}")
        
        pdf.cell(80,10, phrase, ln=1, border=False)



    
    pdf.set_font('times', 'BU', 12)
    pdf.cell(80,10, 'Symbols Legend', ln=1, border=False)

    pdf.set_font('times', '', 10)
    
    # Definizione delle larghezze per le colonne
    column1_width = 50
    column2_width = 100

    # Aggiunta delle righe con tabulazioni
    pdf.cell(column1_width, 5, 'DN:', border=0)
    pdf.cell(column2_width, 5, 'Nominal Diameter [inch]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'OD:', border=0)
    pdf.cell(column2_width, 5, 'Outside Diameter [mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'CA:', border=0)
    pdf.cell(column2_width, 5, 'Corrosion Allowance [mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'codeTol:', border=0)
    pdf.cell(column2_width, 5, 'Manufacturing Tolerance code', ln=1, border=0)   

    pdf.cell(column1_width, 5, 'TOL:', border=0)
    pdf.cell(column2_width, 5, 'Manufacturing Tolerance [perc. or mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'E:', border=0)
    pdf.cell(column2_width, 5, '', ln=1, border=0)

    pdf.cell(column1_width, 5, 'W:', border=0)
    pdf.cell(column2_width, 5, '', ln=1, border=0)

    pdf.cell(column1_width, 5, 'Y:', border=0)
    pdf.cell(column2_width, 5, '', ln=1, border=0)    

    pdf.cell(column1_width, 5, 'Temp:', border=0)
    pdf.cell(column2_width, 5, 'Temperature [°C]', ln=1, border=0)   

    pdf.cell(column1_width, 5, 'Press:', border=0)
    pdf.cell(column2_width, 5, 'Pressure [bar]', ln=1, border=0)  

    pdf.cell(column1_width, 5, 'thkC:', border=0)
    pdf.cell(column2_width, 5, 'Minimum calculated thickness [mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'thkCReq:', border=0)
    pdf.cell(column2_width, 5, 'Minmum required thickness [mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'thkCom:', border=0)
    pdf.cell(column2_width, 5, 'Commercially available thickness [mm]', ln=1, border=0)

    pdf.cell(column1_width, 5, 'Allow:', border=0)
    pdf.cell(column2_width, 5, 'Mat. Allowable Stress [MPa]', ln=1, border=0)    

    pdf.cell(column1_width, 5, 'MAWP:', border=0)
    pdf.cell(column2_width, 5, 'Maximum Allowable Working Pressure [bar]', ln=1, border=0)



    #pdf.add_page()
    #pdf.image('files/grafico.png', x=30, y=50, w=300) 

    '''
    pipReportPdf = os.path.join(sessionDir, f"report1_{st.session_state.session_id}.pdf")

    pdf.output(pipReportPdf)

    

