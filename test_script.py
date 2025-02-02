from backend import process_pdf_and_respond

# File path to the PDF (ensure it's accessible)
pdf_path = r"C:\Users\Rakshitha Goud\Downloads\raks_resume_3.pdf"

# List of questions to ask
questions = [
    "What is the main topic discussed in this document?",
    "What are the skills of the candidate?",
    "what is the name of the candidate?"
]

# OpenAI API key (remove if using Groq)

openai.api_key = os.getenv("OPENAI_API_KEY")
# openai_api_key = "yosk-proj-EfAIgBj_JLN8kVcGlNLHhDhOXY-79QahP023qy825MuAnYydZ3NhPTfBXdG-1cA4WossmaefrcT3BlbkFJc2ihriRPMnz04c3HwYLqKeJeZbxJ0rI4Zupzo54S94LVinEYPbzBn5pbW5lK9UTCuKZudcwRcA"

# Call function for each question
for question in questions:
    try:
        answer = process_pdf_and_respond(pdf_path, question, openai_api_key)
        print(f"Q: {question}\nA: {answer}\n")
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
