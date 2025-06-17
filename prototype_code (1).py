"""
prototype code

"""

# Install dependencies
pip install numpy==1.25.2 openai-whisper==20231117 spacy==3.7.2
python -m spacy download en_core_web_sm

import whisper
import spacy
import os
from google.colab import files

# Load whisper and spaCy
model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])  # Optimize for speed

# --- SYMPTOM LIST (NO SYNONYMS) ---
SYMPTOMS = [
    'fever', 'headache', 'nausea', 'vomiting', 'cough', 'cold', 'chest pain', 'fatigue', 'low energy',
    'shortness of breath', 'muscle pain', 'joint pain', 'diarrhea', 'abdominal pain',
    'sore throat', 'loss of appetite', 'dizziness', 'sneezing', 'rash', 'runny nose',
    'pain behind the eyes', 'blurry', 'vision', 'blurry vision', 'blurred vision', 'sensitivity to light',
    'sensitivity to sound', 'tired all the time', 'swelling', 'itchy skin', 'dark urine',
    'loss of balance', 'night sweats', 'pain behind my eyes', 'pain in my eyes', 'muscle aches',
    'thirsty', 'chest pressure', 'head starts pounding'
]

# --- DISEASE LIST AND ICD-10 CODES ---
ICD10_CODES = {
    "influenza": "J11.1",  # Influenza
    "pneumonia": "J12.9",  # Viral pneumonia
    "bronchitis": "J40",   # Bronchitis
    "migraine": "G43.9",   # Migraine
    "diabetes": "E11.9",   # Type 2 diabetes
    "hypertension": "I10", # Essential hypertension
    "covid-19": "U07.1",   # COVID-19
    "asthma": "J45.909",   # Asthma
    "gastritis": "K29.70"  # Gastritis
}

# --- SYMPTOM DETECTION ---
def detect_symptoms(text):
    text_lower = text.lower()
    detected = [symptom for symptom in SYMPTOMS if symptom in text_lower]
    return list(set(detected))  # Remove duplicates

# --- DISEASE DETECTION ---
def detect_diseases(text):
    text_lower = text.lower()
    diseases = []
    for disease in ICD10_CODES.keys():
        if disease in text_lower:
            diseases.append((disease, ICD10_CODES[disease]))
    return diseases

# --- ACCURACY CHECK ---
def get_accuracy(transcribed_text, detected):
    true_positives = [symptom for symptom in detected if symptom in transcribed_text.lower()]
    false_negatives = [symptom for symptom in SYMPTOMS if symptom in transcribed_text.lower() and symptom not in true_positives]
    accuracy = round(len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-6) * 100, 2)
    return true_positives, false_negatives, accuracy

# --- TRANSCRIBE AUDIO ---
def process_audio(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        print("🔊 Transcribing audio...")
        result = model.transcribe(file_path)
        transcribed_text = result['text'].strip()
        if not transcribed_text:
            raise ValueError("Transcription failed: No text detected.")
        print(f"\n📜 Transcribed Text:\n  {transcribed_text}\n")

        detected_symptoms = detect_symptoms(transcribed_text)
        detected_diseases = detect_diseases(transcribed_text)
        print(f"🤒 Detected Symptoms: {detected_symptoms}")
        print(f"🩺 Detected Diseases with ICD-10 Codes: {detected_diseases}")

        true_pos, false_neg, accuracy = get_accuracy(transcribed_text, detected_symptoms)
        print(f"\n✅ True Positives: {true_pos}")
        print(f"❌ False Negatives: {false_neg}")
        print(f"🎯 Accuracy: {accuracy}%")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# --- UPLOAD FILE ---
print("Choose the audio file (mp3 or wav):")
uploaded = files.upload()
audio_path = next(iter(uploaded))
process_audio(audio_path)
