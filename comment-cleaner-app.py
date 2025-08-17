import streamlit as st
import pandas as pd
import re
import html
import emoji
import contractions
import nltk
import zipfile
import io
import base64
import gzip
import time
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from io import BytesIO

# --- Download NLTK resources ---
nltk.download("punkt")
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

    if stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    elif lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# --- Utils ---
def rand_suffix():
    # कुछ ब्राउज़र्स cache/AV heuristic से बचने को filename random कर देते हैं
    return f"{int(time.time())}_{random.randint(1000,9999)}"

def to_base64_download_link(data_bytes: bytes, filename: str, mime: str, label: str):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- Streamlit App ---
st.title("🧹 Customer Comment Cleaning App")
st.write("Choose whether you want to upload a file or manually enter comments.")

mode = st.radio("Select Input Method:", ("📂 Upload CSV/Excel", "✍️ Manual Input"))

# Sidebar Options
st.sidebar.header("⚙️ Cleaning Options")

language = st.sidebar.selectbox("Choose Language for Stopwords", ["english", "hindi", "marathi"])
remove_emoji = st.sidebar.checkbox("Remove Emojis", value=True)
remove_html = st.sidebar.checkbox("Remove HTML Tags", value=True)
expand_contract = st.sidebar.checkbox("Expand Contractions", value=True)
normalize_repeat = st.sidebar.checkbox("Normalize Repeated Characters", value=True)
remove_nonalpha = st.sidebar.checkbox("Remove Non-Alphabetic", value=True)
lowercase = st.sidebar.checkbox("Convert to Lowercase", value=True)

apply_stemming = st.sidebar.checkbox("Apply Stemming (PorterStemmer)", value=False)
apply_lemmatization = st.sidebar.checkbox("Apply Lemmatization (WordNetLemmatizer)", value=False)

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

    # --- Download Section ---
    st.markdown("### 📥 Download Options")
    mode_dl = st.radio("Mode:", ("Normal (CSV/Excel/ZIP)", "Safe (TXT/JSON/Base64/GZIP)"))

    if mode_dl == "Normal (CSV/Excel/ZIP)":
        file_type = st.radio("Choose file format:", ("CSV", "Excel", "ZIP"))

        if file_type == "CSV":
            csv_bytes = df.to_csv(index=False).encode("utf-8")  # bytes
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=csv_bytes,
                file_name=f"Cleaned_Comments_{rand_suffix()}.csv",
                mime="text/csv"
            )

        elif file_type == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            output.seek(0)  # very important
            st.download_button(
                label="⬇️ Download Cleaned Excel",
                data=output,
                file_name=f"Cleaned_Comments_{rand_suffix()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:  # ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                # CSV
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                zipf.writestr("Cleaned_Comments.csv", csv_bytes)
                # Excel
                excel_output = io.BytesIO()
                with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="CleanedData")
                excel_output.seek(0)
                zipf.writestr("Cleaned_Comments.xlsx", excel_output.read())
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download Cleaned Data (ZIP)",
                data=zip_buffer,
                file_name=f"Cleaned_Comments_{rand_suffix()}.zip",
                mime="application/zip"
            )

    else:
        # SAFE OPTIONS (rarely flagged)
        st.markdown("**Safe options (recommended if browser shows virus warning):**")

        # 1) TXT (tab-delimited) — very safe
        txt_str = df.to_csv(index=False, sep="\t")
        st.download_button(
            label="⬇️ Safe Download (TXT)",
            data=txt_str,
            file_name=f"Cleaned_Comments_{rand_suffix()}.txt",
            mime="text/plain"
        )

        # 2) JSON — safe
        json_str = df.to_json(orient="records", force_ascii=False)
        st.download_button(
            label="⬇️ Safe Download (JSON)",
            data=json_str.encode("utf-8"),
            file_name=f"Cleaned_Comments_{rand_suffix()}.json",
            mime="application/json"
        )

        # 3) CSV (GZIP) — safe & small
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        gz_buf = io.BytesIO()
        with gzip.GzipFile(filename="Cleaned_Comments.csv", fileobj=gz_buf, mode="wb") as gz:
            gz.write(csv_bytes)
        gz_buf.seek(0)
        st.download_button(
            label="⬇️ Download CSV (GZIP .gz)",
            data=gz_buf,
            file_name=f"Cleaned_Comments_{rand_suffix()}.csv.gz",
            mime="application/gzip"
        )

        # 4) Base64 fallback link (copy-paste friendly)
        st.markdown("Or use Base64 fallback (click then save):")
        to_base64_download_link(
            data_bytes=csv_bytes,
            filename=f"Cleaned_Comments_{rand_suffix()}.csv",
            mime="text/csv",
            label="📎 Download CSV via Base64"
        )

        st.info("Safe mode files rarely get blocked. TXT/JSON open easily in Excel (Data → From Text/JSON).")
