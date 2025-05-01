from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
prompt = PromptTemplate(
    template='Write a summary for the following topic - \n {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

# print(docs[0].page_content)

chain = prompt | model | parser
result = chain.invoke({'topic': docs[0].page_content})
print(result)