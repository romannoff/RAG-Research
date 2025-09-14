from gigachat import GigaChat, context

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class ContextEnrichmentRag(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 temperature: float = 0.0,
                 collection_name: str = 'collection',
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.answer_temperature = temperature

        self.llm_model = GigaChat(credentials=settings.token, verify_ssl_certs=False)

        collection_info = self.database.client.get_collection(collection_name)
        self.max_id = collection_info.points_count

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.search(query=query, limit=3)
        ids = [chunk['id'] for chunk in chunks]

        encrichment_ids = set()

        for id_ in ids:
            encrichment_ids.add(max(id_ - 1, 0))
            encrichment_ids.add(id_)
            encrichment_ids.add(min(id_ + 1, self.max_id))

        encrichment_ids = list(encrichment_ids)

        chunks = self.database.search_point_by_ids(encrichment_ids)
        chunks = [chunk['chunk_text'] for chunk in chunks]

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
    model = ContextEnrichmentRag(collection_name='natural_questions')
    model('who got the first nobel prize in physics')



