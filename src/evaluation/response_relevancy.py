from openai import AsyncOpenAI
from ragas.llms import llm_factory
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy

async def getting_resp_rel_metric(judje_model, embeddings_model_name, user_input, response, retrieved_context):
    client = AsyncOpenAI(
        base_url="http://192.168.88.21:91/v1", 
        api_key="ollama"
    )

    llm = llm_factory(judje_model, client=client)
    
    # Инициализируем эмбеддинги через LangChain Community
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_context
    )

    scorer = ResponseRelevancy(llm=llm, embeddings=embeddings)
    result = await scorer.single_turn_ascore(sample)
    
    return result