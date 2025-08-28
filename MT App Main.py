import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import pandas as pd
import numpy as np
import time
from nltk.translate.bleu_score import sentence_bleu
from googletrans import Translator
import os
import tensorflow as tf
import json
import PyPDF2
from docx import Document
import asyncio
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

translator = Translator()

LANGUAGES = {
    "Auto Detect": "auto",
    "Arabic": "ar",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Irish": "ga",
    "Japanese": "ja",
    "Spanish": "es",
    "Russian": "ru",
    "Swedish": "sv",
    "Ukrainian": "uk",

}

helsinki_models = {}

def calculate_bleu_score(reference: str, candidate: str) -> float:
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

# File loading functions
def load_text_from_file(filepath, file_role="source"):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(filepath) if ext == ".csv" else pd.read_csv(filepath, sep='\t')
        if file_role not in df.columns:
            print(f"Warning: '{file_role}' column is missing in {filepath}. Using the last column.")
            df[file_role] = df.iloc[:, -1].astype(str)
        else:
            df[file_role] = df[file_role].astype(str)
        return df
    elif ext == ".pdf":
        text = ""
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            messagebox.showerror("File Read Error", f"Error reading PDF file: {e}")
            return None
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return pd.DataFrame({file_role: lines})
    elif ext == ".docx":
        text = ""
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            messagebox.showerror("File Read Error", f"Error reading DOCX file: {e}")
            return None
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return pd.DataFrame({file_role: lines})
    elif ext == ".parquet":
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            messagebox.showerror("File Read Error", f"Error reading Parquet file: {e}")
            return None
        if "source" not in df.columns:
            print(f"Warning: 'source' column missing in {filepath}. Using the last column as source.")
            df["source"] = df.iloc[:, -1].astype(str)
        else:
            df["source"] = df["source"].astype(str)
        if "target" not in df.columns:
            print(f"Warning: 'target' column missing in {filepath}. Using the last column as target.")
            df["target"] = df.iloc[:, -1].astype(str)
        else:
            df["target"] = df["target"].astype(str)
        return df
    else:
        messagebox.showerror("Unsupported File", f"File type {ext} is not supported.")
        return None

previous_target_file = None

def load_data():
    global previous_target_file
    file_choice = messagebox.askquestion("Data Source", "Do you want to load training data from Parquet files?")
    if file_choice == "yes":
        filepaths = filedialog.askopenfilenames(
            title="Select Parquet Files for Training",
            filetypes=[("Parquet Files", "*.parquet")]
        )
        if not filepaths:
            return None, None
        data_frames = []
        for file in filepaths:
            df = load_text_from_file(file, "source")
            if df is not None:
                data_frames.append(df)
        if not data_frames:
            return None, None
        combined_df = pd.concat(data_frames, ignore_index=True)
        if "source" not in combined_df.columns or "target" not in combined_df.columns:
            messagebox.showerror("Data Error", "Parquet files must contain both 'source' and 'target' columns.")
            return None, None
        print(f"Combined {len(filepaths)} parquet files with {len(combined_df)} samples.")
        source_data = combined_df[['source']]
        target_data = combined_df[['target']]
        return source_data, target_data
    else:
        source_file = filedialog.askopenfilename(
            title="Select Source Language File",
            filetypes=[("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("PDF Files", "*.pdf"),
                       ("Word Documents", "*.docx")]
        )
        if not source_file:
            return None, None
        source_data = load_text_from_file(source_file, "source")
        if source_data is None:
            return None, None
        print("Loaded source file:", source_file)

        if previous_target_file:
            use_previous = messagebox.askyesno("Use Previous Target",
                                               "Do you want to use the previously selected target language file?")
            target_file = previous_target_file if use_previous else filedialog.askopenfilename(
                title="Select Target Language File",
                filetypes=[("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("PDF Files", "*.pdf"),
                           ("Word Documents", "*.docx")]
            )
            if target_file:
                previous_target_file = target_file
        else:
            target_file = filedialog.askopenfilename(
                title="Select Target Language File",
                filetypes=[("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("PDF Files", "*.pdf"),
                           ("Word Documents", "*.docx")]
            )
            if target_file:
                previous_target_file = target_file

        if not target_file:
            return None, None
        target_data = load_text_from_file(target_file, "target")
        if target_data is None:
            return None, None
        print("Loaded target file:", target_file)

        if 'source' not in source_data.columns:
            print("Warning: 'source' column is missing in source file. Using last column as source.")
            source_data['source'] = source_data.iloc[:, -1].astype(str)
        if 'target' not in target_data.columns:
            print("Warning: 'target' column is missing in target file. Using last column as target.")
            target_data['target'] = target_data.iloc[:, -1].astype(str)

        source_data['source'] = source_data['source'].astype(str)
        target_data['target'] = target_data['target'].astype(str)
        print(f"Number of source sentences: {len(source_data)}")
        print(f"Number of target sentences: {len(target_data)}")

        DEBUG = True
        if DEBUG:
            n = min(len(source_data), len(target_data), 1000)
            print("DEBUG MODE: Using first", n, "samples for training/evaluation.")
            source_data = source_data.head(n)
            target_data = target_data.head(n)

        return source_data, target_data

# Evaluation function for the Evaluate Sample button
def run_evaluation(input_text, reference_text):
    max_length = 50
    # Tokenize input and the reference
    input_enc = tokenizer(input_text, return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
    reference_enc = tokenizer(reference_text, return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)["input_ids"]

    # Compute cross-entropy loss using the reference as labels (this does not affect model weights)
    outputs = model(**input_enc, labels=reference_enc)
    loss = outputs.loss

    # Generate a translation using the Helsinki model
    preds = model.generate(input_enc["input_ids"], max_length=max_length)
    pred_text = tokenizer.decode(preds[0], skip_special_tokens=True)

    # Calculate BLEU score between reference and the predicted text
    bleu = calculate_bleu_score(reference_text, pred_text)

    # Extract encoder representations for cosine similarity
    try:
        encoder_out = model.model.encoder(input_enc["input_ids"], attention_mask=input_enc["attention_mask"],
                                          output_hidden_states=True)
        input_embedding = tf.reduce_mean(encoder_out.last_hidden_state, axis=1).numpy()[0]
        ref_encoder_out = model.model.encoder(reference_enc, output_hidden_states=True)
        reference_embedding = tf.reduce_mean(ref_encoder_out.last_hidden_state, axis=1).numpy()[0]
        cos_sim = cosine_similarity(input_embedding, reference_embedding)
    except Exception as e:
        print("Error extracting encoder embeddings:", e)
        cos_sim = -1.0

    return loss.numpy(), bleu, cos_sim, pred_text

def evaluate_sample():
    input_text = input_entry.get("1.0", tk.END).strip()
    reference_text = ref_entry.get("1.0", tk.END).strip()
    if not input_text or not reference_text:
        result_label.config(text="Please enter both input and reference texts for evaluation.")
        return
    loss, bleu, cos_sim, pred_text = run_evaluation(input_text, reference_text)
    eval_text = (f"Cross-Entropy Loss: {loss:.4f}\n"
                 f"BLEU Score: {bleu:.4f}\n"
                 f"Cosine Similarity: {cos_sim:.4f}\n"
                 f"Predicted Translation:\n{pred_text}")
    result_label.config(text=eval_text)

def on_evaluate_button_click():
    threading.Thread(target=evaluate_sample).start()

# Helsinki translation function
def helsinki_translate_text(input_text, src_code, tgt_code):
    global helsinki_models
    model_key = (src_code, tgt_code)
    if model_key not in helsinki_models:
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        try:
            tokenizer_local = AutoTokenizer.from_pretrained(model_name)
            model_local = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            helsinki_models[model_key] = (model_local, tokenizer_local)
            print(f"Loaded Helsinki model for {src_code} -> {tgt_code}")
        except Exception as e:
            raise Exception(f"Error loading Helsinki model for {src_code} to {tgt_code}: {e}")
    else:
        model_local, tokenizer_local = helsinki_models[model_key]
    max_length = 50
    inputs = tokenizer_local(input_text, return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
    translated_ids = model_local.generate(**inputs)
    translation = tokenizer_local.decode(translated_ids[0], skip_special_tokens=True)
    return translation

def translate_text():
    input_text = input_entry.get("1.0", tk.END).strip()
    if not input_text:
        result_label.config(text="Please enter text to translate.")
        return

    selected_source = selected_language.get()
    selected_target = target_language.get()
    src_code = LANGUAGES.get(selected_source, "auto")
    tgt_code = LANGUAGES.get(selected_target, "en")

    # Google Translate part
    try:
        local_translator = Translator()
        google_trans = asyncio.run(
            local_translator.translate(input_text, src=src_code, dest=tgt_code)
        )
    except Exception as e:
        result_label.config(text=f"Error with Google Translate: {e}")
        return

    google_translation_text = google_trans.text

    # Helsinki translation model part
    try:
        helsinki_translation_text = helsinki_translate_text(input_text, src_code, tgt_code)
    except Exception as e:
        helsinki_translation_text = f"Error in Helsinki translation: {e}"

    # Form initial result text with both translations
    result_text = (f"Google Translate Output:\n{google_translation_text}\n\n"
                   f"Helsinki Model Output:\n{helsinki_translation_text}")

    # Check for provided reference translation to calculate BLEU score
    reference_text = ref_entry.get("1.0", tk.END).strip()
    if reference_text:
        bleu_score = calculate_bleu_score(reference_text, helsinki_translation_text)
        result_text += f"\n\nBLEU Score vs Reference: {bleu_score:.4f}"

    result_label.config(text=result_text)

def on_translate_button_click():
    threading.Thread(target=translate_text).start()

# GUI Setup
window = tk.Tk()
window.title("Machine Translation Evaluation")
window.geometry("600x750")

language_label = tk.Label(window, text="Select Source Language (for translation):")
language_label.pack(pady=5)

sorted_languages = sorted(LANGUAGES.keys(), key=lambda s: s.lower())
selected_language = ttk.Combobox(window, values=sorted_languages)
selected_language.set("Auto Detect")
selected_language.pack(pady=5)

# Target Language Options
target_language_label = tk.Label(window, text="Select Target Language (for translation):")
target_language_label.pack(pady=5)

target_language = ttk.Combobox(window, values=sorted_languages)
target_language.set("English")
target_language.pack(pady=5)

input_label = tk.Label(window, text="Enter Text for Translation (Input):")
input_label.pack(pady=5)

input_entry = tk.Text(window, height=8, width=60)
input_entry.pack(pady=5)

# Reference translation
ref_label = tk.Label(window, text="Enter Reference Translation:")
ref_label.pack(pady=5)

ref_entry = tk.Text(window, height=8, width=60)
ref_entry.pack(pady=5)

translate_button = tk.Button(window, text="Translate", command=on_translate_button_click)
translate_button.pack(pady=5)

evaluate_button = tk.Button(window, text="Evaluate Sample", command=on_evaluate_button_click)
evaluate_button.pack(pady=5)

result_label = tk.Label(window, text="Results will appear here", justify="left", wraplength=550)
result_label.pack(pady=5)

progress_bar = ttk.Progressbar(window, length=300)
progress_bar.pack(pady=5)

training_info = tk.StringVar()
training_info_label = tk.Label(window, textvariable=training_info)
training_info_label.pack(pady=5)

# Data loading for fine tuning
def start_training():
    source_data, target_data = load_data()
    if source_data is None or target_data is None:
        return
    messagebox.showinfo("Info", "Fine-tuning is disabled in this evaluation setup.")

train_button = tk.Button(window, text="Train Model",
                         command=lambda: threading.Thread(target=start_training).start())
train_button.pack(pady=5)

window.mainloop()
