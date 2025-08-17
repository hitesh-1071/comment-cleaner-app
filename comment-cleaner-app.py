import streamlit as st
import pandas as pd
import re
import html
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from io import BytesIO

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Stemmer & Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# --- Cleaning Functions ---
def remove_emojis(text):
    return emoji.replace_emoji(text, "")

def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)

def expand_contractions(text):
    return contractions.fix(text)

def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

def remove_non_alphabetic(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)

def clean_text(
    text,
    language="english",
    remove_emoji=True,
    remove_html=True,
    expand_contract=True,
    normalize_repeat=True,
    remove_nonalpha=True,
    lowercase=True,
    stemming=False,
    lemmatization=False
):
    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)

    if remove_emoji:
        text = remove_emojis(text)
    if remove_html:
        text = remove_html_tags(text)
    if expand_contract:
        text = expand_contractions(text)
    if normalize_repeat:
        text = normalize_repeated_chars(text)
    if remove_nonalpha:
        text = remove_non_alphabetic(text)
    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    try:
        STOPWORDS = set(stopwords.words(language))
    except:
        STOPWORDS = set(stopwords.words("english"))

    tokens = [word for word in tokens if word not in STOPWORDS]

    # Apply stemming or lemmatization
    if stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    elif lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# --- Streamlit App ---
st.title("🧹 Customer Comment Cleaning App")
st.write("Choose whether you want to upload a file or manually enter comments.")

# --- First Selection ---
mode = st.radio("Select Input Method:", ("📂 Upload CSV/Excel", "✍️ Manual Input"))

# Sidebar Options
st.sidebar.header("⚙️ Cleaning Options")

# Language Support
language = st.sidebar.selectbox("Choose Language for Stopwords", ["english", "hindi", "marathi"])

# Extra Cleaning Options
remove_emoji = st.sidebar.checkbox("Remove Emojis", value=True)
remove_html = st.sidebar.checkbox("Remove HTML Tags", value=True)
expand_contract = st.sidebar.checkbox("Expand Contractions", value=True)
normalize_repeat = st.sidebar.checkbox("Normalize Repeated Characters", value=True)
remove_nonalpha = st.sidebar.checkbox("Remove Non-Alphabetic", value=True)
lowercase = st.sidebar.checkbox("Convert to Lowercase", value=True)

# Stemming / Lemmatization Options
apply_stemming = st.sidebar.checkbox("Apply Stemming (PorterStemmer)", value=False)
apply_lemmatization = st.sidebar.checkbox("Apply Lemmatization (WordNetLemmatizer)", value=False)

# --- Input Data ---
df = None

if mode == "📂 Upload CSV/Excel":
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        if "Comment" not in df.columns:
            st.error("❌ The file must contain a 'Comment' column.")
            df = None
        else:
            st.success("✅ File uploaded successfully!")

elif mode == "✍️ Manual Input":
    manual_text = st.text_area("Enter comments here (one comment per line):")
    if manual_text.strip() != "":
        comments = manual_text.strip().split("\n")
        df = pd.DataFrame({"Comment": comments})
        st.success("✅ Manual input loaded successfully!")

# --- Cleaning Process ---
if df is not None:
    with st.spinner("Cleaning comments... ⏳"):
        df["Original_WordCount"] = df["Comment"].apply(lambda x: len(str(x).split()))
        df["Cleaned_Comment"] = df["Comment"].apply(
            lambda x: clean_text(
                x,
                language=language,
                remove_emoji=remove_emoji,
                remove_html=remove_html,
                expand_contract=expand_contract,
                normalize_repeat=normalize_repeat,
                remove_nonalpha=remove_nonalpha,
                lowercase=lowercase,
                stemming=apply_stemming,
                lemmatization=apply_lemmatization
            )
        )
        df["Cleaned_WordCount"] = df["Cleaned_Comment"].apply(lambda x: len(str(x).split()))
        df["Cleaning_Flag"] = df["Cleaned_Comment"].apply(lambda x: 1 if x.strip() != "" else 0)

    st.subheader("📊 Preview of Cleaned Data")
    st.dataframe(df.head(10))

    # --- Download cleaned file ---
    file_type = st.radio("📥 Choose file format to download:", ("CSV", "Excel"))

    if file_type == "CSV":
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", data=csv, file_name="Cleaned_Comments.csv", mime="text/csv")
    else:
        output = BytesIO()
        df.to_excel(output, index=False, engine="openpyxl")
        st.download_button("⬇️ Download Cleaned Excel", data=output.getvalue(),
                           file_name="Cleaned_Comments.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

