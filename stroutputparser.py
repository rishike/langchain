from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task="text-generation"
# )

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs={
        "temperature":0.5,
        "max_new_tokens":100,
        "do_sample":True
    }
)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# one way
# prompt1 = template1.invoke({'topic': 'black hole'})
# result = model.invoke(prompt1)

# # print(result.content)

# prompt2 = template2.invoke({'text': result.content})
# result1 = model.invoke(prompt2)

# print(result1.content)


# str output parser way
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)