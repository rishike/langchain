from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model='gpt-4', temperature=0.1, max_completion_tokens=100) # max_completion_tokens use for response content restricted to given input value of max_completion_tokens

result = model.invoke("What is the capital of canada?")

print(result)