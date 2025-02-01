import faiss
from sentence_transformers import SentenceTransformer
import openai
import pdfplumber
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can use a smaller version like "distilgpt2" for faster performance
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad token to be the same as eos token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    :param file_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    """
    Splits text into smaller chunks of a specified size.
    :param text: Input text to split.
    :param chunk_size: Maximum number of words in each chunk.
    :return: List of text chunks.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to create and store embeddings using FAISS
def create_and_store_embeddings(chunks, model):
    """
    Generates embeddings for text chunks and stores them in a FAISS index.
    :param chunks: List of text chunks.
    :param model: Pretrained SentenceTransformer model.
    :return: FAISS index containing the embeddings.
    """
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    embeddings = model.encode(chunks)
    index.add(embeddings)
    return index, embeddings

# Function to search for relevant chunks
def search_faiss_index(question, model, index, chunks, top_k=5):
    """
    Searches the FAISS index for chunks relevant to the input question.
    :param question: User's question.
    :param model: Pretrained SentenceTransformer model.
    :param index: FAISS index containing embeddings.
    :param chunks: Original text chunks.
    :param top_k: Number of relevant chunks to retrieve.
    :return: List of relevant text chunks.
    """
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function to generate an answer using GPT-2
# Function to generate an answer using GPT-2
def generate_answer(question, relevant_chunks):
    """
    Generates an answer based on the relevant text chunks and question using GPT-2.
    :param question: User's question.
    :param relevant_chunks: Relevant text chunks retrieved from FAISS.
    :return: Generated answer as a string.
    """
    # Join the relevant chunks to create context
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"

    # Tokenize the input prompt and ensure it's within the model's token limit
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    # Generate the output using GPT-2
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Explicitly passing the attention mask
        pad_token_id=tokenizer.eos_token_id,   # Ensuring pad_token_id is set
        max_new_tokens=150,                    # Control the number of tokens generated (output length)
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )

    # Decode and return the generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


# Main logic to tie everything together
def process_pdf_and_respond(pdf_path, question, openai_api_key):
    """
    Complete pipeline for processing a PDF and responding to a question.
    :param pdf_path: Path to the PDF file.
    :param question: User's question.
    :param openai_api_key: OpenAI API key for generating answers.
    :return: Generated answer.
    """
    # Set the OpenAI API key (not needed for local GPT-2 model)
    openai.api_key = openai_api_key

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
