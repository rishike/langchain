from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model_name = 'sentence-transformers/all-MiniLM-L6-v2'

embedding = HuggingFaceEmbeddings(model_name=model_name)

text = 'New delhi is the capital of india'
documents = [
    'New Delhi is the capital of India',
    'Kolkata is the capital of West Bengal',
    'Paris is the capital of France'
]

# query_vector = embedding.embed_query(text)

doc_vector = embedding.embed_documents(documents)

print(doc_vector)