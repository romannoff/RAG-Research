import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    FactualCorrectness
)

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.config import Config


settings = Config.from_yaml("config.yaml")


class Evaluator:
    def __init__(self, research_name: str = 'evaluation_results'):
        self.llm = ChatOpenAI(
            base_url=settings.base_url,
            model=settings.model,
            openai_api_key=settings.password,
            temperature=0.0,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
        )

        self.embeddings = OpenAIEmbeddings(
            base_url=settings.base_url,
            openai_api_key=settings.password,
        )

        self.metrics = [
            Faithfulness(),                          # проверяет «faithfulness» (фактическую обоснованность) 
            LLMContextPrecisionWithReference(),      # точность контекста (сравнивается с ground truth)
            LLMContextRecall(),                      # полнота контекста 
            FactualCorrectness()                     # проверка фактов в ответе (аналог answer_correctness) 
        ]
        self.research_name = research_name


    def eval(self, dataset):
        self.dataset = Dataset.from_dict(dataset)

        results = evaluate(
            dataset=self.dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=False,
        )
        # return results

        print("=== РЕЗУЛЬТАТЫ ОЦЕНКИ RAG СИСТЕМЫ ===")
        print(results)

        # Сохранение детального отчёта
        df_results = results.to_pandas()
        df_results.to_csv(f"research/data/ragas_{self.research_name}.csv", index=False)

        return results
