import streamlit as st
from reportPDF import UpdateReportPdf
import os
import base64

st.title('📑 Report')

#    === Controllo accesso ===
if 'prot' not in st.session_state:
    st.session_state.prot = False
    
if st.session_state.prot == False:
    st.info('unauthorized access')
    st.stop()

# funzione per verificare esistena files
def check_missing(files_dict, required_keys):
    missing_required = []
    missing_optional = []

    for key, path in files_dict.items():
        if not os.path.exists(path):
            if key in required_keys:
                missing_required.append((key, path))
            else:
                missing_optional.append((key, path))

    return missing_required, missing_optional


# --- Definizione dei file richiesti ---
sessionDir = "sessions"

file_paths = {
    "DatiGenerali": os.path.join(sessionDir, f"DatiGenerali_{st.session_state.session_id}.csv"),
    "DatiGeo": os.path.join(sessionDir, f"DatiGeo_{st.session_state.session_id}.csv"),
    "Concrete": os.path.join(sessionDir, f"concrete_{st.session_state.session_id}.csv"),
    "Steel": os.path.join(sessionDir, f"steel_{st.session_state.session_id}.csv"),
    "ReinfLayers": os.path.join(sessionDir, f"Reinf_layers_{st.session_state.session_id}.csv"),
    "DatiLoad": os.path.join(sessionDir, f"DatiLoad_{st.session_state.session_id}.csv"),
    "LoadsMR": os.path.join(sessionDir, f"loadsMR_{st.session_state.session_id}.csv"),
    "ImgGeo": os.path.join(sessionDir, f"imgGeo_{st.session_state.session_id}.png"),
    "Img1": os.path.join(sessionDir, f"img1_{st.session_state.session_id}.png"),
    "ImgMx": os.path.join(sessionDir, f"imgMx_{st.session_state.session_id}.png"),
    "ImgMy": os.path.join(sessionDir, f"imgMy_{st.session_state.session_id}.png"),
    "ImgMxMy": os.path.join(sessionDir, f"imgMxMy_{st.session_state.session_id}.png"),
}

# --- Elenco dei file fondamentali --- 
required_files = [ "DatiGenerali", "DatiGeo", "Concrete", "Steel", "ReinfLayers" ]

missing_required, missing_optional = check_missing(file_paths, required_files)

# --- Se mancano file fondamentali: STOP ---
if missing_required:
    st.error("❌ Some required information are missing. No possible to proceed.")
    for name, path in missing_required:
        st.write(f"- **{name}** → `{path}`")
    st.stop()  # 🔥 Interrompe l'esecuzione di Streamlit

# --- Se mancano solo file opzionali: avvisa ma continua ---
if missing_optional:
    st.warning("⚠️ Some optional information are missing):")
    for name, path in missing_optional:
        st.write(f"- **{name}** → `{path}`")

st.success("✅ All required information are present. The report can be generated.")



if st.button('Generate or update Report'):
    UpdateReportPdf()

pdfReport = os.path.join(sessionDir, f"report1_{st.session_state.session_id}.pdf")

if os.path.exists(pdfReport):

    if "show_pdf" not in st.session_state:
        st.session_state.show_pdf = False
    
    def toggle_pdf():
        st.session_state.show_pdf = not st.session_state.show_pdf
    
    st.markdown("<hr style='border: 0.5px solid red;'>", unsafe_allow_html=True)
    label = "👁️ Preview Report" if not st.session_state.show_pdf else "❌ Close Preview"
    #st.button(label, on_click=toggle_pdf, key="toggle_pdf_btn")

    # Visualizza PDF se attivo
    st.session_state.show_pdf = True
    
    if st.session_state.show_pdf:
        #with open(pdfReport, "rb") as f:
        #    base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        #    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
        #    st.markdown(pdf_display, unsafe_allow_html=True)
        with open(pdfReport, "rb") as f:
            st.download_button(
                label="📄 Open Report in a new window",
                data=f,
                file_name="ReportPflex1.pdf",
                mime="application/pdf",
                help='***Save Report in your local drive***'
            )
            
    # Separatore linea rossa
    st.markdown("""<hr style="border: 0.5px solid red;">""", unsafe_allow_html= True)
        
    #with open(pdfReport, "rb") as pdf_file:
    #    st.sidebar.download_button(
    #        label="💾 Download Report",
    #        data=pdf_file,
    #        file_name="ReportPflex1.pdf",
    #        mime="application/pdf",
    #        help='***Save Report in your local drive***'
    #    )

else:
    st.warning("No Report created. Please click on button 'Generate or update Report' to generate it.")
    st.stop()


        
    
        
       



