from gigachat import GigaChat, context

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt, hyde_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class HyDeRAG(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 answer_temperature: float = 0.7,
                 decomposition_temperature: float = 0.0,
                 collection_name: str = 'collection',
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.answer_temperature = answer_temperature
        self.decomposition_temperature = decomposition_temperature

        self.llm_model = GigaChat(credentials=settings.token, verify_ssl_certs=False)

    def __call__(self, query: str) -> ModelAnswer:
        document = self.hyde(query)
        
        chunks = self.database.search(query=document, limit=10)
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

    def hyde(self, query):
        payload = hyde_prompt(query, temperature=self.decomposition_temperature)
        completion = self.llm_model.chat(payload)
        new_query = completion.choices[0].message.function_call.arguments
        return new_query['hypothetical_document']

if __name__ == '__main__':
    model = HyDeRAG(collection_name='natural_questions')
    print(model('who got the first nobel prize in physics'))
