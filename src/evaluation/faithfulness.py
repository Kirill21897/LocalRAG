import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness
from ragas import SingleTurnSample # Импортируем контейнер для данных

async def getting_faithfulness_metric(judje_model, user_input, response, retrieved_context):
    # Setup LLM
    client = AsyncOpenAI(
        base_url="http://192.168.88.21:91/v1", 
        api_key="ollama" 
    )

    llm = llm_factory(judje_model, client=client)

    # Create metric
    scorer = Faithfulness(llm=llm)

    # Оборачиваем данные в SingleTurnSample (требование новых версий Ragas)
    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_context
    )

    # Используем асинхронный метод
    result = await scorer.single_turn_ascore(sample)

    return result.value