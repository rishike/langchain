from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )


# model = ChatHuggingFace(llm=llm)
model = ChatOpenAI(model='gpt-4', temperature=0.1, max_completion_tokens=100)


class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place': 'indian'})

# print(prompt)

# result = model.invoke(prompt)

# final_result =  parser.parse(result.content)

# print(final_result)


# chain way

chain = template | model | parser
result = chain.invoke({'place': 'canadian'})
print(result)