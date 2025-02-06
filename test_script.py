import os
from backend import process_pdf_and_respond

# File path to the PDF (ensure it's accessible)
pdf_path = r"C:\Users\Rakshitha Goud\Downloads\raks_resume_3.pdf"

# List of questions to ask
questions = [
    "What is the main topic discussed in this document?",
    "What are the skills of the candidate?",
    "What is the name of the candidate?"
]

# Ensure Groq API key is set
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Call function for each question
for question in questions:
    try:
        answer = process_pdf_and_respond(pdf_path, question)
        print(f"Q: {question}\nA: {answer}\n")
    except Exception as e:
        print(f"Error processing question '{question}': {e}")