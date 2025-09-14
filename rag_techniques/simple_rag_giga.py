from gigachat import GigaChat, context

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt, simple_rag_prompt_ru


settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class SimpleRag(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 temperature: float = 0.0,
                 collection_name: str = 'collection',
                 language: str = 'en'
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.llm_model = GigaChat(
            credentials=settings.token, 
            verify_ssl_certs=False,
        )

        self.temperature = temperature
        self.language = language

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.search(query=query, limit=10)
        chunks = [chunk['chunk_text'] for chunk in chunks]

        if self.language == 'en':
            prompt = simple_rag_prompt(query, chunks)
        else:
            prompt = simple_rag_prompt_ru(query, chunks)

        payload = {
            "messages": prompt,
            "temperature": self.temperature 
        } 

        answer = self.llm_model.chat(payload=payload).choices[0].message.content

        return ModelAnswer(
            query=query,
            context=chunks,
            answer=answer
        )


if __name__ == '__main__':
    model = SimpleRag(collection_name='natural_questions')
    print(model('who got the first nobel prize in physics'))



