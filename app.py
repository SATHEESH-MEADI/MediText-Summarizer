import streamlit as st
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Set the path for the combined data file (update this path as needed)
data_file_path = "/Users/satheesh/Desktop/Data.txt"

# Load and preprocess the text data
with open(data_file_path, "r") as file:
    text_data = file.read()

# Split the text into passages (e.g., 200-word chunks for retrieval)
passages = [text_data[i:i + 200] for i in range(0, len(text_data), 200)]

# Load Sentence-BERT model for embedding-based retrieval
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for each passage
passage_embeddings = embedder.encode(passages, convert_to_tensor=True)

# Load Pegasus model and tokenizer for answer generation
pegasus_model_name = "google/pegasus-xsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Define a function to preprocess the question
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    return text

# Function to retrieve relevant passages
def retrieve_relevant_passages(question, passage_embeddings, passages, top_k=3, min_similarity=0.7):
    """
    Retrieve top-k passages based on cosine similarity with a stricter threshold.
    """
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, passage_embeddings)[0]
    filtered_passages = [(idx, cos_scores[idx].item()) for idx in range(len(cos_scores)) if cos_scores[idx] > min_similarity]
    filtered_passages.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve top passages
    relevant_passages = [passages[idx] for idx, _ in filtered_passages[:top_k]]
    return " ".join(relevant_passages)  # Combine only the most relevant passages

def generate_answer(question, context, max_length=200, min_length=100):
    """
    Generate an answer based on the question and context with length constraints.
    """
    # Add guidance to the prompt
    input_text = f"Answer the following question based on the context:\n\nQuestion: {question}\n\nContext: {context}"
    inputs = pegasus_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = pegasus_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=1.2,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    answer = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer

# Streamlit app UI
st.title("Clinical Data Question-Answering Chatbot")
st.write("Ask questions about the clinical data and receive generated answers based on relevant text passages.")

# Question input from user
question = st.text_input("Ask a question about the data:")

if question:
    # Preprocess the question
    preprocessed_question = preprocess_text(question)

    # Retrieve relevant passages for the question
    context = retrieve_relevant_passages(preprocessed_question, passage_embeddings, passages)

    # Generate an answer using the Pegasus model
    answer = generate_answer(preprocessed_question, context)

    # Display the answer
    st.write("### Answer:")
    st.write(answer)
