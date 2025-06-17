from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env')

model = OpenAI(model="gpt-4o-mini")

system_message = 'Você é um avaliador de produtos experiente.'

prompt_features = ChatPromptTemplate.from_messages(
    [
        ('system', system_message),
        ('human', 'Liste todas as features do produto {product}.')
    ]
)

pros_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_message),
        (
            'human',
            """
            Dadas essas features: {features}, liste 5 pontos positivos das features. Envie sua resposta
            como uma lista ordenada de um a 5, precedida pelo título: 'Pontos positivos:'. Siga rigorosamente estas instruções.     
            """
        )
    ]
)

cons_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_message),
        (
            'human',
            """
            Dadas essas features: {features}, liste 5 pontos negativos das features. Envie sua resposta
            como uma lista ordenada de um a 5, precedida pelo título: 'Pontos negativos:'. Siga rigorosamente estas instruções.       
            """
        )
    ]
)

pros_chain = (
    RunnableLambda(lambda x: {'features': x})  # Recebe o argumento
    | pros_prompt
    | model
)
cons_chain = (
    RunnableLambda(lambda x: {'features': x})
    | cons_prompt
    | model
)


def combine_pros_cons(pros_cons_dict):
    return f"{pros_cons_dict['pros']}{pros_cons_dict['cons']}"


chain = (
    prompt_features
    | model
    | RunnableParallel(pros=pros_chain, cons=cons_chain)
    | RunnableLambda(combine_pros_cons)
)

response = chain.invoke({'product': 'Xiaomi Note 8'})

if 'Human:' in response:
    response = response.replace('Human:', '')
elif 'human:' in response:
    response = response.replace('human:', '')

print(response)  # Output quebrado. Culapa da IA.
