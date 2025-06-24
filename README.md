### MediVoice: Speech-to-Text Disease Detector
MediVoice is a B.Tech minor project aimed at automating medical documentation and disease detection using AI. It leverages OpenAI's Whisper for speech-to-text transcription and spaCy's NLP models to map symptoms to ICD-10 codes, improving healthcare efficiency and patient outcomes.

## Setup Instructions

# Clone the Repository:
git clone https://github.com/grk1102/MediVoice.git
cd MediVoice_GitHub_File


# Install Dependencies:Create a virtual environment and install required packages:
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt

# Required packages (in requirements.txt):
openai==0.28.0
spacy==3.7.2
torch==2.0.1


# Download spaCy Model:
python -m spacy download en_core_web_sm


# Run the Code:

For Python script:python src/prototype_code.py


For Jupyter notebooks (if added):jupyter notebook src/




## API Keys:

Obtain an OpenAI API key and set it in a .env file (not committed):OPENAI_API_KEY=your_key_here


Use a library like python-dotenv to load the key.



## Usage

Speech-to-Text: Input audio files (e.g., .wav) are transcribed using Whisper.
Disease Detection: Transcribed text is processed by spaCy to extract symptoms and map to ICD-10 codes.
Sample Data: Place audio files in data/ (excluded by .gitignore due to size).
