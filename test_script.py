import os
from backend import process_pdf_and_respond

# ✅ FIXED PDF PATH
pdf_path = r"C:\Users\Rakshitha Goud\Downloads\raks resume 3.pdf"

questions = [
    "What is the main topic discussed in this document?",
    "What are the skills of the candidate?",
    "What is the name of the candidate?"
]

<<<<<<< HEAD
=======
# Ensure Groq API key is set
>>>>>>> 3007fd6429043a7c675316354cc939ddc5dd30cb
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

<<<<<<< HEAD
=======
# Call function for each question
>>>>>>> 3007fd6429043a7c675316354cc939ddc5dd30cb
for question in questions:
    try:
        answer = process_pdf_and_respond(pdf_path, question)
        print(f"Q: {question}\nA: {answer}\n")
    except Exception as e:
        print(f"Error processing question '{question}': {e}")