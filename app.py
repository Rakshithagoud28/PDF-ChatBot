import openai
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to create and store embeddings
def create_and_store_embeddings(chunks, model, index):
    embeddings = model.encode(chunks)
    index.add(embeddings)
    return embeddings

# Function to search for relevant chunks
def search_faiss_index(question, model, index, chunks, top_k=5):
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to generate answers using OpenAI API
def generate_answer(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit app
st.title("PDF-Based Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.write("### Extracted Text:")
    st.text_area("", value=text, height=300)

    # Split text into chunks
    with st.spinner("Processing text into chunks..."):
        chunks = split_text_into_chunks(text)

    # Load the sentence transformer model
    with st.spinner("Loading model and storing embeddings..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        create_and_store_embeddings(chunks, model, index)

    st.success("PDF processed successfully! You can now ask questions.")

    # Input for user question
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Finding the answer..."):
            relevant_chunks = search_faiss_index(question, model, index, chunks)
            answer = generate_answer(question, relevant_chunks)

        st.write("### Answer:")
        st.write(answer)

