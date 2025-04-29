from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


passthrough = RunnablePassthrough()

prompt1 = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='explain a joke: {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'bear'})
print(result)