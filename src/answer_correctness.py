from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
import json
from tqdm.notebook import tqdm
import numpy as np

from src.config import Config


settings = Config.from_yaml("config.yaml")

class AnswerCorrectness():
    def __init__(self):
        self.so_llm_model = OpenAI(
            base_url=settings.base_url,
            api_key=settings.password,
            timeout=30.0
        )

    def get_correctness(self, questions: list[str], answers: list[str], true_answers: list[str]):
        labels = []

        for question, answer, true_answer in tqdm(zip(questions, answers, true_answers), total=len(questions)):
            try:
                labels.append(self.eval(question, answer, true_answer)[0])
            except:
                break


        return np.mean(labels), labels

    def eval(self, question: str, answer: str, true_answer: str):
        prompt, extra_body = self.prompt(question=question, answer=answer, true_answer=true_answer)

        completion = self.so_llm_model.chat.completions.create(
                model=settings.model,
                messages=prompt,
                extra_body=extra_body,
                temperature=0.0,
                timeout=30.0
        )
        res = json.loads(completion.choices[0].message.content)
        return res['label'], res['reflections']


    @staticmethod
    def prompt(question: str, answer: str, true_answer: str):
        system = "You are an impartial judge. Your task is to compare the two texts and determine how they are consistent in meaning\n"
        instructions = """
Comparison algorithm:
1. If the correct answer is "No answer", and the system's response contains the full answer (although correct), then label=0. This is important because the system should be able to say "No answer" if there really is no answer in the database.
2. If there is a correct answer, but the system's answer is PARTIALLY correct, then label=1.
3. If the system's answer is partially correct, but contains more information than the correct answer, then label=1. Because the system usually gives longer answers.
"""

        user_input = f"""
Answer (the system's response):
{answer}

Correct answer (reference answer):
{true_answer}
"""

        output = """
Output the result in the format:
- reflections: 1-2 sentences explaining the reason for the verdict.
- label: "1" (if the meaning matches) or "0" (if it doesn't match).
"""
        examples = """
Example 1:
Answer (the system's response):
Global trade during the Ming Dynasty of China was significant and diverse. The Ming Empire engaged in extensive trade with both Europe and Japan, and the economy was one of the largest in the world during that period. Key aspects include:
1. **Foreign Trade**
2. **Currency and Trade**
3. **Economic Policies**
These factors contributed to a vibrant and expansive global trade network during the Ming Dynasty.

Correct answer (reference answer): 
No answer

reflections: the correct answer is no answer, but the current answer is not, so it is incorrect.
label: 0

Example 2:
Answer (the system's response):
The majority owner of Reading Football Club is Dai Yongge and Dai Xiuli. Sir John Madejski is the chairman of the club.
Ccorrect answer (reference answer): 
Dai Yongge

reflections: the answer correctly mentions that Dai Yongge is the founder of the football club
label: 1
"""

        messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instructions + user_input + output}
        ]    

        json_schema = EstimatorAnswer.model_json_schema()
        extra_body = {"guided_json": json_schema}
        return  messages, extra_body


class Labels(int, Enum):
    bad = 0
    good = 1


class EstimatorAnswer(BaseModel):
    reflections: str
    label: Labels