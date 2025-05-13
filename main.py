import streamlit as st
import pickle
import base64
from PIL import Image
import time

# Set konfigurasi halaman
st.set_page_config(
    page_title="Spam Email Detector 📧",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Load model dan vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open("emailspam_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()


# Fungsi untuk menambahkan background image dan styling custom CSS
def add_custom_bg():
    st.markdown(
        """
        <style>
        /* Background image fixed dan blur */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1533319417897-ef9d6d93d9d2?auto=format&fit=crop&w=1740&q=80");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            filter: brightness(0.85);
        }
        /* Style judul */
        h1 {
            color: #1f4e79;
            font-weight: 800;
            margin-bottom: 0.1rem;
        }
        /* Style paragraf */
        p.subtitle {
            font-size: 1.2rem;
            color: grey;
            margin-top: 0;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        /* Style tombol */
        div.stButton > button:first-child {
            background-color: #1f4e79;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #145374;
            color: #e0e0e0;
        }
        /* Style area teks */
        textarea {
            border-radius: 12px !important;
            border: 1.5px solid #1f4e79 !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            resize: vertical !important;
            min-height: 150px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_bg()

# Kontainer utama
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Judul dan deskripsi
st.markdown("<h1>📧 Spam Email Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Paste your email text below and find out instantly if it's <b>SPAM</b> or <b>NOT SPAM</b>.</p>", unsafe_allow_html=True)

# Input teks email
email_input = st.text_area("✍️ Paste your email text here:", height=180)

# Tombol prediksi di samping input menggunakan kolom
col1, col2, col3 = st.columns([1,2,1])
with col1:
    predict_button = st.button("🔍 Check Spam")

# Area hasil prediksi
result_placeholder = st.empty()

if predict_button:
    if not email_input.strip():
        st.warning("⚠️ Please enter some email text first!")
    else:
        # Animasi loading
        with st.spinner('Analyzing your email...'):
            time.sleep(1)  # Simulasi delay loading

            # Preprocessing sederhana
            cleaned_email = email_input.lower()
            vect_email = vectorizer.transform([cleaned_email])

            # Prediksi dan probabilitas
            prediction = model.predict(vect_email)[0]
            proba = model.predict_proba(vect_email)[0][prediction]

            # Tampilkan hasil dengan warna dan ikon sesuai prediksi
            if prediction == 1:
                result_placeholder.error(f"⚠️ This email is likely SPAM!(Confidence: {proba*100:.1f}%)")
            else:
                result_placeholder.success(f"✅ This email is NOT SPAM.(Confidence: {proba*100:.1f}%)")

# Footer kecil
st.markdown("<p style='text-align:center; margin-top:3rem; color:#666;'>Created By: Egi Pratama 🦾🤖🤘</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
