from langchain_openai import ChatOpenAI

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt, simple_rag_prompt_ru

settings = Config.from_yaml("config.yaml")


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

        self.answer_temperature = temperature

        self.llm_model = ChatOpenAI(
            base_url=settings.base_url,
            model=settings.model,
            openai_api_key=settings.password,
            temperature=self.answer_temperature,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
        )
        self.language = language

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.search(query=query, limit=10)
        chunks = [chunk['chunk_text'] for chunk in chunks]

        if self.language == 'en':
            prompt = simple_rag_prompt(query, chunks)
        else:
            prompt = simple_rag_prompt_ru(query, chunks)

        answer = self.llm_model.invoke(prompt).content

        return ModelAnswer(
            query=query,
            context=chunks,
            answer=answer
        )


if __name__ == '__main__':
    model = SimpleRag(collection_name='natural_questions')
    model('who got the first nobel prize in physics')



