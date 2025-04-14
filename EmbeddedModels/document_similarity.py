from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents = [
    "The sun rises in the east.",
    "She enjoys reading books in her free time.",
    "Technology is changing the way we live and work.",
    "They went for a walk in the park after dinner.",
    "Learning new skills can boost your confidence."
]

query = "from which side sun rises"
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]
index , score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is: ", score)