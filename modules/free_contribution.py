from pathlib import Path

import streamlit as st

from PIL import Image  # pip install pillow


# --- PATH SETTING ---
THIS_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd
ASSETS_DIR = THIS_DIR / "assets"
STYLES_DIR = THIS_DIR / "styles"
CSS_FILE = STYLES_DIR / "main.css"

# --- GENERAL SETTINGS ---
#STRIPE_CHECKOUT = "https://buy.stripe.com/6oEdRj2Jp6I29qw3cd"
PAYPAL_CHECKOUT = 'https://www.paypal.com/donate/?business=NEWMGUVF3WCG6&no_recurring=0&item_name=Contributo+per+continuare+a+sviluppare+e+pubblicare+applicazioni.&currency_code=EUR'
CONTACT_EMAIL = "YOUREMAIL@EMAIL.COM"
PRODUCT_NAME = "Give Your Free Contribution"
PRODUCT_TAGLINE = "Support our idea of Free Team, passion driven!"
PRODUCT_DESCRIPTION = """
To continue developing and maintaining this and other valuable resources, we need your support. Your generous donation will directly contribute to:

- Development Costs: Ensuring the app remains cutting-edge and user-friendly.
- Maintenance and Updates: Regular updates to keep the app secure and efficient.
- Community Support: Providing tutorials, documentation, and customer support to help users get the most out of our application.

Every donation, no matter the size, makes a significant impact.
Thank you for considering our request. Together, we can continue to develop tools that support and inspire engineers, and not only, everywhere.

**This is your new superpower**
"""

# --- PAGE CONIFG ---
st.set_page_config(
    page_title= PRODUCT_NAME,
    page_icon= ":star:",
    layout= "centered",
    initial_sidebar_state= "collapsed"
)

# --- MAIN SECTION ---
st.header(PRODUCT_NAME)
st.subheader(PRODUCT_TAGLINE)
left_col, right_col = st.columns((2,1))   # left column i twice right column 2:1
with left_col:
    st.text("")
    st.write(PRODUCT_DESCRIPTION)
    #url = 'https://stackoverflow.com'
    url = PAYPAL_CHECKOUT
    
    #st.markdown(f'''<a href={PAYPAL_CHECKOUT}><button style="background-color:GreenYellow;">Get the add-in</button></a>''',unsafe_allow_html= True)
    #st.page_link(PAYPAL_CHECKOUT, label="PayPal Checkout", icon="🌎")
    st.markdown(f'''
        <a href= "https://www.paypal.com/donate/?business=NEWMGUVF3WCG6&no_recurring=0&item_name=Contributo+per+continuare+a+sviluppare+e+pubblicare+applicazioni.&currency_code=EUR">
        <button style="background-color:GreenYellow;">Donate your contribution</button></a>''',unsafe_allow_html=True)


with right_col:
    product_image = Image.open("assets/mani.jpg")
    st.image(product_image, width= 450)



