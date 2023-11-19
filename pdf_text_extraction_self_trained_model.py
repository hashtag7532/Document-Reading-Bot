from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import PyPDF2

class CustomDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {'context': self.contexts[idx], 'question': self.questions[idx], 'answer': self.answers[idx]}

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Example training data
contexts_train = [...]  # List of document texts for training
questions_train = [...]  # List of questions for training
answers_train = [...]  # List of corresponding answers for training

# Tokenize the training dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_dataset = CustomDataset(contexts_train, questions_train, answers_train, tokenizer)

# Initialize BERT model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-base-cased')

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=5e-5)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for epoch in range(3):  # Adjust the number of epochs
    for batch in dataloader:
        outputs = model(**batch, labels=batch['answer'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_qa')
tokenizer.save_pretrained('fine_tuned_bert_qa')

# Use the fine-tuned model for question answering
def answer_question_bert_fine_tuned(pdf_path, question):
    pdf_text = extract_text_from_pdf(pdf_path)

    # Load the fine-tuned BERT model for question answering
    fine_tuned_model = BertForQuestionAnswering.from_pretrained('fine_tuned_bert_qa')
    fine_tuned_tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_qa')

    # Tokenize the input
    tokenized_input = fine_tuned_tokenizer(question, pdf_text, return_tensors='pt')

    # Get the answer using the fine-tuned model
    start_logits, end_logits = fine_tuned_model(**tokenized_input).start_logits, fine_tuned_model(**tokenized_input).end_logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    answer = fine_tuned_tokenizer.decode(tokenized_input['input_ids'][0][answer_start:answer_end])

    return answer

# Example usage
pdf_path = 'example.pdf'
user_question = input("Enter your question: ")
answer = answer_question_bert_fine_tuned(pdf_path, user_question)
print(f"Answer: {answer}")
