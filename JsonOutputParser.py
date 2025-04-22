from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="llama-3-1-8b-instruct-urx",
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

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name , age , city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


template2 = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

#### first way

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))



#### Chain way

chain = template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)