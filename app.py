import streamlit as st
import zipfile
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForQuestionAnswering
import torch
import requests
import shutil
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai
import openai
import pandas as pd
import plotly.express as px
from gtts import gTTS
import tempfile
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Set your API key here for the language translation this is from the google cloud api 
API_KEY = "AIzaSyBLUELtvdlQr3T5g5CU8UhN5JSBnDIXyQA"



# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

#Languages------------------------------------------------------------------

LANGUAGES = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi',
    'Arabic': 'ar',
    'Chinese': 'zh',
    'Dutch': 'nl',
    'Korean': 'ko',
    'Russian': 'ru',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Japanese': 'ja',
    'Italian': 'it',
    'German': 'de',
    'French': 'fr',
    'Greek': 'el',
    'Thai': 'th'
}
#<---------------------------------------------------------Sentiment Analysis----------------------------->


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        try:
            # Handle empty text
            if not text or len(text.strip()) == 0:
                return None

            # Split text into smaller chunks (to handle long texts)
            max_length = 1000
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            if not chunks:
                return None

            sentiments = []
            for chunk in chunks:
                try:
                    result = self.analyzer(chunk)
                    if result and len(result) > 0:
                        sentiments.append(result[0])
                except Exception as e:
                    st.warning(f"Chunk analysis failed: {str(e)}")
                    continue

            if not sentiments:
                return None

            # Count sentiments
            positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
            negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
            neutral_count = len(sentiments) - positive_count - negative_count

            total = len(sentiments)
            if total == 0:
                return None

            # Calculate percentages
            sentiment_scores = {
                'positive': (positive_count / total) * 100,
                'negative': (negative_count / total) * 100,
                'neutral': (neutral_count / total) * 100
            }

            # Determine overall sentiment
            max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            
            # Map sentiment to color and label
            sentiment_mapping = {
                'positive': {'color': 'green', 'label': 'Positive'},
                'negative': {'color': 'red', 'label': 'Negative'},
                'neutral': {'color': 'blue', 'label': 'Neutral'}
            }

            return {
                'overall_sentiment': sentiment_mapping[max_sentiment[0]]['label'],
                'color': sentiment_mapping[max_sentiment[0]]['color'],
                'confidence': max_sentiment[1] / 100,
                'breakdown': sentiment_scores
            }

        except Exception as e:
            st.error(f"Sentiment analysis failed: {str(e)}")
            return None



#<--------------------------------------------------Named Entity Model----------------------------------->

class MedicalNER:
    def __init__(self):
        self.nlp = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

    def get_named_entities(self, text):
        entities = self.nlp(text)
        return [(entity['word'], entity['entity_group']) for entity in entities]







#<---------------------------------------------------------Translator----------------------------->
class Translator:
    def __init__(self):
        self.translations_cache = {}

    def translate_text(self, text, target_language_code):
        """Translate text using Google Translation API v2 with caching to minimize API calls."""
        cache_key = (text, target_language_code)
        
        # Check cache first
        if cache_key in self.translations_cache:
            return self.translations_cache[cache_key]
        
        # API request if no cached result
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {'q': text, 'target': target_language_code, 'key': API_KEY}
        
        try:
            response = requests.get(url, params=params)
            response_data = response.json()
            translation = response_data['data']['translations'][0]['translatedText']
            
            # Cache the result to minimize API calls
            self.translations_cache[cache_key] = translation
            return translation
        
        except Exception as e:
            st.error(f"Translation failed: {response_data.get('error', {}).get('message', 'Unknown error')}")
            return text

#<---------------------------------------------------------PubMed model----------------------------->
# # Move this function outside the class
class PubMedBERTSummarizer:
    def __init__(self):
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def preprocess_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    @st.cache_data
    def get_sentence_embeddings(_self, text):
        inputs = _self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(_self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)

    def get_pubmedbert_summary(self, text):
        try:
            processed_text = self.preprocess_text(text)
            doc_embedding = self.get_sentence_embeddings(processed_text)
            sentences = sent_tokenize(processed_text)
            
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                sent_embedding = self.get_sentence_embeddings(sentence)
                similarity = torch.nn.functional.cosine_similarity(doc_embedding, sent_embedding).item()
                sentence_scores.append((i, sentence, similarity))
            
            sorted_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
            selected_sentences = sorted_sentences[:5]  # Select top 5 sentences
            summary_sentences = sorted(selected_sentences, key=lambda x: x[0])
            
            summary = ' '.join(sent for _, sent, _ in summary_sentences)
            return summary
            
        except Exception as e:
            st.error(f"Summarization error: {str(e)}")
            return text



#<-----------------------------------------------------Extracting the Text Data -------------------------------->



def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
    return text

def extract_files(uploaded_file):
    extract_to = "extracted_text_files"
    os.makedirs(extract_to, exist_ok=True)
    text_files = []

    # Handle zip files
    if uploaded_file.name.endswith(".zip"):
        print(f"Extracting zip file: {uploaded_file.name}")
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        text_files = [os.path.join(root, f) for root, _, files in os.walk(extract_to) for f in files if f.endswith('.txt')]
        print(f"Text files extracted from zip: {text_files}")

    # Handle single text files
    elif uploaded_file.name.endswith(".txt"):
        print(f"Processing text file: {uploaded_file.name}")
        file_path = os.path.join(extract_to, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text_files.append(file_path)

    # Handle PDF files
    elif uploaded_file.name.endswith(".pdf"):
        print(f"Processing PDF file: {uploaded_file.name}")
        pdf_path = os.path.join(extract_to, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_text = extract_text_from_pdf(pdf_path)

        # Save extracted text to a .txt file
        text_file_path = os.path.join(extract_to, uploaded_file.name.replace(".pdf", ".txt"))
        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(pdf_text)
        text_files.append(text_file_path)
        print(f"Extracted text saved to: {text_file_path}")

    # Ensure files are found
    if not text_files:
        raise FileNotFoundError("No text files found in the uploaded file.")

    return text_files

#<-----------------------------------------------------Text to audio -------------------------------->


# Text-to-speech function using SpeechT5
def text_to_speech_torch(text, speaker_id=0, sample_rate=16000):
    """Convert text to speech using SpeechT5 model and return audio file path."""
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings[speaker_id], vocoder)

    # Resample the speech to match the desired sample rate
    resample = Resample(orig_freq=24000, new_freq=sample_rate)
    resampled_speech = resample(speech.squeeze(0))

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        write(tmp_file.name, sample_rate, resampled_speech.numpy())
        return tmp_file.name





# def text_to_speech(text):
#     """Convert text to speech and return an audio file path."""
#     if not text.strip():  # Check if text is empty
#         return None
#     tts = gTTS(text=text, lang='en')
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
#         tts.save(tmp_file.name)
#         return tmp_file.name



#<---------------------------------------------------------Chatbot model----------------------------->


# Configure OpenAI API to use Ollama's local server
openai.api_base = 'http://localhost:11434/v1'
openai.api_key = 'ollama'  # Placeholder key, not used by Ollama

# Medical Chatbot class using Ollama for Q&A
class MedicalChatbot:
    def __init__(self):
        self.conversation_history = []

    def get_answer(self, question, context):
        messages = [
            {"role": "system", "content": "You are a medical expert chatbot. Answer based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ] + self.conversation_history[-3:]  # Keep only the last 3 interactions for context

        # Call the Ollama model using OpenAI's compatible API structure with "llama3.2"
        response = openai.ChatCompletion.create(
            model="llama3.2",  # Use "llama3.2" as the model name for Ollama
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )

        answer = response['choices'][0]['message']['content']
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        self.conversation_history = []


#<---------------------------------------------------Initialization----------------------------->

def initialize_session_state():
    for key, val in [
        ('summarizer', PubMedBERTSummarizer()),
        ('chatbot', MedicalChatbot()),
        ('translator', Translator()),
        ('ner', MedicalNER()),
        ('sentiment_analyzer', SentimentAnalyzer()),
        ('current_summary', None),
        ('translated_summary', None),
        ('selected_language', 'English'),
        ('chat_history', [])  # Store chat history in session state
    ]:
        st.session_state.setdefault(key, val)


# Custom CSS for active and inactive buttons
st.markdown("""
    <style>
    .stButton > button {
        background-color: white;
        color: black;
    }
    .stButton > button.active {
        background-color: red !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Function to update the selected section in session state
def set_active_button(button_name):
    st.session_state.active_button = button_name

# Initialize the active button state if not set
if 'active_button' not in st.session_state:
    st.session_state.active_button = "None"



















#<-------------------------------------------






def main():
    st.title("Medical Text Analysis System")
    st.write("Upload medical texts or enter raw text for summarization, translation, NER, and interactive Q&A")
    initialize_session_state()

    # Language selection in the sidebar
    target_language = st.sidebar.selectbox("Select Target Language", LANGUAGES.keys(), 
                                           index=list(LANGUAGES.keys()).index(st.session_state.selected_language))
    st.session_state.selected_language = target_language

    # Sidebar buttons for different functionalities
    st.sidebar.title("Analysis Options")
    original_text_button = st.sidebar.button("Original Text")
    summary_button = st.sidebar.button("Summary & Translation")
    ner_button = st.sidebar.button("Named Entity Recognition")
    qa_button = st.sidebar.button("Interactive Q&A")
    sentiment_button = st.sidebar.button("Sentiment Analysis")

    # Option to either upload a file or input raw text
    st.write("### Input Options:")
    uploaded_files = st.file_uploader("Upload medical text file(s)", type=["txt", "zip", "pdf"], accept_multiple_files=True)
    raw_text = st.text_area("Or, paste raw text here:")

    # Processing text if files or raw text are provided
    if uploaded_files or raw_text:
        try:
            text_data = ""

            # Process uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    text_files = extract_files(uploaded_file)
                    for text_file in text_files:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_data += f.read() + "\n"

            # Use raw text if provided
            if raw_text:
                text_data += raw_text

            st.write("### Processing Text")

            # Display the selected option based on button clicks
            if original_text_button:
                st.subheader("Original Text")
                st.text(text_data)
                if st.button("🔊 Listen to Original Text"):
                    audio_file = text_to_speech(text_data, "original_text.mp3")
                    audio_bytes = open(audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")

            elif summary_button:
                st.subheader("PubMedBERT Summary")
                summary = st.session_state.summarizer.get_pubmedbert_summary(text_data)
                st.session_state.current_summary = summary
                st.write(summary)
                
                if st.button("🔊 Listen to Summary"):
                    audio_file = text_to_speech(summary, "summary_text.mp3")
                    audio_bytes = open(audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")

                if target_language != "English":
                    translated_text = st.session_state.translator.translate_text(summary, LANGUAGES[target_language])
                    st.session_state.translated_summary = translated_text
                    st.write(f"Translation ({target_language}):\n{translated_text}")
                    if st.button("🔊 Listen to Translation"):
                        audio_file = text_to_speech(translated_text, "translated_text.mp3")
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

            elif ner_button:
                st.subheader("Named Entity Recognition (NER)")
                entities = st.session_state.ner.get_named_entities(text_data)
                entity_text = "\n".join([f"{entity.replace('#', '')} - {entity_type}" for entity, entity_type in entities])
                st.write(entity_text)
                
                if st.button("🔊 Listen to NER"):
                    audio_file = text_to_speech(entity_text, "ner_text.mp3")
                    audio_bytes = open(audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")

            elif qa_button:
                st.subheader("Interactive Q&A with Medical Expert Bot")
                if st.session_state.chat_history:
                    chat_text = "\n".join([f"🎃 You: {chat['question']}\n💡 Bot: {chat['answer']}" for chat in st.session_state.chat_history])
                    st.write(chat_text)
                    
                    if st.button("🔊 Listen to Q&A"):
                        audio_file = text_to_speech(chat_text, "qa_text.mp3")
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

                question = st.text_input("Enter your question:")
                answer_language = st.radio("Select answer language:", ["English", target_language], horizontal=True)

                if question:
                    answer = st.session_state.chatbot.get_answer(question, st.session_state.current_summary)
                    if answer_language != "English":
                        answer = st.session_state.translator.translate_text(answer, LANGUAGES[answer_language])
                    st.write("🎃 You:", question)
                    st.write("💡 Bot:", answer)
                    st.session_state.chat_history.append({"question": question, "answer": answer})

            elif sentiment_button:
                st.subheader("Sentiment Analysis")
                sentiment_result = st.session_state.sentiment_analyzer.analyze_sentiment(text_data)

                if sentiment_result:
                    sentiment_text = f"Overall Sentiment: {sentiment_result['overall_sentiment']}\n" \
                                     f"Confidence: {sentiment_result['confidence']*100:.1f}%\n" \
                                     f"Positive: {sentiment_result['breakdown']['positive']:.1f}%\n" \
                                     f"Negative: {sentiment_result['breakdown']['negative']:.1f}%\n" \
                                     f"Neutral: {sentiment_result['breakdown']['neutral']:.1f}%"
                    st.write(sentiment_text)
                    
                    if st.button("🔊 Listen to Sentiment Analysis"):
                        audio_file = text_to_speech(sentiment_text, "sentiment_text.mp3")
                        audio_bytes = open(audio_file, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")

                    # Visualization (kept as in the original code)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<p style='color: green'>Positive: {sentiment_result['breakdown']['positive']:.1f}%</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='color: red'>Negative: {sentiment_result['breakdown']['negative']:.1f}%</p>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<p style='color: blue'>Neutral: {sentiment_result['breakdown']['neutral']:.1f}%</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists("extracted_text_files"):
                shutil.rmtree("extracted_text_files")


if __name__ == "__main__":
    main()
