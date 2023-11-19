from transformers import pipeline
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def answer_question_bert(pdf_path, question):
    pdf_text = extract_text_from_pdf(pdf_path)

    # Use pre-trained BERT model for question answering
    qa_pipeline = pipeline('question-answering', model='deepset/bert-base-cased-squad2', tokenizer='deepset/bert-base-cased-squad2')

    # Find the answer using BERT
    result = qa_pipeline({'question': question, 'context': pdf_text})

    return result['answer']

# Example usage
pdf_path = r"C:\Users\Parth Dodia\Downloads\BCVS SOP.pdf"
user_question = input("Enter your question: ")
answer = answer_question_bert(pdf_path, user_question)
print(f"Answer: {answer}")
