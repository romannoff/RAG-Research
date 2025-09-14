from gigachat import GigaChat, context

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class DartboardRag(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 temperature: float = 0.7,
                 collection_name: str = 'collection',
                 diversity_weight: float = 1.0,
                 relevance_weight: float = 2.0,
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.answer_temperature = temperature

        self.llm_model = GigaChat(credentials=settings.token, verify_ssl_certs=False)

        self.diversity_weight = diversity_weight
        self.relevance_weight = relevance_weight
        self.sigma = 0.1

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.dartboard_search(
            query=query,
            limit=10,
            diversity_weight=self.diversity_weight,
            relevance_weight=self.relevance_weight,
            sigma=self.sigma
            )

        prompt = simple_rag_prompt(query, chunks)

        payload = {
            "messages": prompt,
            "temperature": self.answer_temperature 
        } 

        answer = self.llm_model.chat(payload=payload).choices[0].message.content

        return ModelAnswer(
            query=query,
            context=chunks,
            answer=answer
        )

    


if __name__ == '__main__':
    model = DartboardRag(collection_name='natural_questions')
    model('who got the first nobel prize in physics')



