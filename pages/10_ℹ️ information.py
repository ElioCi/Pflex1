import streamlit as st


st.title('ℹ️ \u2003info ')
st.write('---')
st.markdown('### Application for ***Reinforced Concrete Section*** verification'  )
st.write('')
st.write('**Scope**')

st.write('This software helps the designer to check a Reinforced Concrete Section subject to external loads. It also draws the interaction domain of the section and shows the position of the loads combinations on it.')
#st.write('It is also possible to define three component groups and evaluate the rating of the piping system.')
st.write('Calculations are made according to EC2 and Italian Code NTC-2018')         
st.write('' )

st.write('**Program Flow**')

st.markdown(
    '<span style="color:red; font-weight:900">Main</span>: select new or stored project and press <b>start</b> ... ➡️ '
    '<span style="color:red; font-weight:900">General Data</span>: input data and tick the checkbox to confirm ... ➡️ '
    '<span style="color:red; font-weight:900">Materials</span>: define concrete and steel properties... ➡️ '
    '<span style="color:red; font-weight:900">Section Geometry</span>: define the geometry of the section and the steel reinforcement or change the existing one... ➡️ '
    '<span style="color:red; font-weight:900">External Loads</span>: Define external loads applied to the section or change the existing ones... ➡️ '
    '<span style="color:red; font-weight:900">Moment Capacity</span>: Evaluate the moment capacity of the section... ➡️ '
    '<span style="color:red; font-weight:900">Interaction Domain</span>: Draws the interaction domain of the section and shows the position of the loads combinations on it... 🏁 End of calculation',
    unsafe_allow_html=True
)



#st.write('**Program Flow**')

#st.write('**Main**: select new or stored project and press ***start*** ... ➡️ **General Data**: input data and tick the checkbox to confirm ... ➡️ **Materials**: define concrete and steel properties...  ➡️ **Section Geometry**: define the geometry of the section and the steel reinforcement or change the existing one... ➡️ **External Loads**: Define external loads applied to the section or change the existing ones... ➡️ **Moment Capacity**: Evaluate the moment capacity of the section... ➡️ **Interaction Domain**: Draws the interaction domain of the section and shows the position of the loads combinations on it...   🏁 End of calculation')
st.write('')

st.write('**Other menu voices**')
st.markdown(
    '<span style="color:red; font-weight:900">Save Project</span>: Let to save the current project on your local drive ... <br>'
    '<span style="color:red; font-weight:900">Units of measurement</span>: Shows the table of the used units ...',
     unsafe_allow_html=True
)


st.markdown('')

st.markdown('')
st.markdown('')
st.info("-- ©️ App developed by ing. Pasquale Aurelio Cirillo - Release 1.0 2026 --")

#st.markdown('<p style="color:red;">ℹ️ Questo è un messaggio di informazione in rosso</p>', unsafe_allow_html=True)
