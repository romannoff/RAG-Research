from gigachat import GigaChat, context

from database.vbase import QdrantBase
from database.bm25_search import BM25Search
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class FusionRag(Model):
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

        self.bm25 = BM25Search(collection_name)

    def __call__(self, query: str) -> ModelAnswer:
        vbase_chunks = self.database.search(query=query, limit=5)
        vbase_chunks = [chunk['chunk_text'] for chunk in vbase_chunks]

        bm25_chunks = self.bm25.search(query)[0]

        chunks = list(set(vbase_chunks + bm25_chunks))

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
    model = FusionRag(collection_name='natural_questions')
    model('who got the first nobel prize in physics')



