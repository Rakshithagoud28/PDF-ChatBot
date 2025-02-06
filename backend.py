import faiss
import requests
import os
import pdfplumber
from sentence_transformers import SentenceTransformer

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to create and store embeddings using FAISS
def create_and_store_embeddings(chunks, model):
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    embeddings = model.encode(chunks)
    index.add(embeddings)
    return index, embeddings

# Function to search for relevant chunks
def search_faiss_index(question, model, index, chunks, top_k=5):
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function to generate an answer using Groq API
def generate_answer(question, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    response = requests.post(
        "https://api.groq.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
        json={"model": "llama3-8b", "messages": [{"role": "user", "content": prompt}]}
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "Error generating response."

# Main logic to process PDF and respond
def process_pdf_and_respond(pdf_path, question):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create and store embeddings
    index, _ = create_and_store_embeddings(chunks, model)

    # Search for relevant chunks
    relevant_chunks = search_faiss_index(question, model, index, chunks)

    # Generate an answer
    answer = generate_answer(question, relevant_chunks)

    return answer