import requests

class Generator:
    def __init__(self, model_name="qwen3:14b", base_url="http://192.168.88.21:91"):
        self.url = f"{base_url}/api/generate"
        self.model_name = model_name

    def generate(self, question: str, context: list) -> str:
        # Объединяем найденные чанки текста в единый контекст
        context_text = "\n\n".join(context)
        
        # Формируем промпт для модели
        full_prompt = (
            f"Используй следующий контекст, чтобы ответить на вопрос.\n"
            f"Контекст: {context_text}\n"
            f"Вопрос: {question}\n"
            f"Ответ:"
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            # Извлекаем текст из ключа 'response' (как в твоем примере)
            return response.json().get("response", "Ошибка: Пустой ответ от сервера")
        except Exception as e:
            return f"Ошибка при обращении к серверу LLM: {e}"