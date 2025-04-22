from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

## local 
# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-3-1b-it",
#     task="text-generation"
# )

# hugging face 
# llm = HuggingFacePipeline.from_model_id(
#     model_id="tiiuae/falcon-7b-instruct",
#     task="text-generation",
#     pipeline_kwargs={
#         "temperature": 0.5,
#         "max_new_tokens": 500,
#         "do_sample": True,
#         "return_full_text": False,       
#     },
# )

# openai
# llm = ChatOpenAI(llm=llm)

model = ChatOpenAI(model='gpt-4', temperature=0.1, max_completion_tokens=100)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template=(
        "Provide 3 facts about {topic} \n {format_instruction}"
    ),
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

####### manual way

# prompt = template.invoke({'topic': 'volcano'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

######### chain way 

chain = template | model | parser

final_result = chain.invoke({'topic': 'volcano'})


print(final_result)