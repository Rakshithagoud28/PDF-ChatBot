from backend import process_pdf_and_respond

# File path to the PDF
pdf_path = r"C:\Users\Rakshitha Goud\Downloads\raks resume 3.pdf"

# Question you want to ask
#question = "What is the main topic discussed in this document?","What are the skills of the candidate"
question = [
    "What is the main topic discussed in this document?",
    "What are the skills of the candidate?"
]

# Your OpenAI API key
# openai_api_key = "__"  # Replace with your actual API key

# Call the function
try:
    answer = process_pdf_and_respond(pdf_path, question, openai_api_key)
    print("Answer:", answer)
except Exception as e:
    print("Error:", e)
    
from backend import generate_answer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Your question and relevant chunks (example)
question = "What is the main topic of the document?"
relevant_chunks = ["Chunk 1 text here", "Chunk 2 text here"]  # Add relevant chunks here

# Call the generate_answer function
answer = generate_answer(question, relevant_chunks)
print(answer)

