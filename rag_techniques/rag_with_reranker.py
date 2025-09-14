from gigachat import GigaChat, context

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt, reranker_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class RerankerRAG(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 answer_temperature: float = 0.7,
                 reranker_temperature: float = 0.0,
                 collection_name: str = 'collection',
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.answer_temperature = answer_temperature
        self.reranker_temperature = reranker_temperature

        self.llm_model = GigaChat(credentials=settings.token, verify_ssl_certs=False)

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.search(query=query, limit=7)
        chunks = [chunk['chunk_text'] for chunk in chunks]

        chunks = self.reranker(query, chunks)

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

    def reranker(self, query: str, chunks: list[str]):
        new_chunks = []

        for chunk in chunks:
            payload = reranker_prompt(query, chunk, self.reranker_temperature)
            completion = self.llm_model.chat(payload)
            label = completion.choices[0].message.function_call.arguments['label']
            
            if int(label):
                new_chunks.append(chunk)

        if not len(new_chunks):
            return ['Tell the user that there is no answer to this question.', ]
        
        return new_chunks
    

if __name__ == '__main__':
    model = RerankerRAG(collection_name='natural_questions')
    model('where did they film high school musical two')