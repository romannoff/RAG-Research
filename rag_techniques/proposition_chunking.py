from pydantic import BaseModel
from openai import OpenAI
import json

from src.config import Config


settings = Config.from_yaml("config.yaml")

system = "Break down the following text into simple, self-contained propositions."
instructions = """Ensure that each proposition meets the following criteria:

1. Express a Single Fact: Each proposition should state one specific fact or claim.
2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.
"""

user_text = """
Return the following JSON: 
{"propositions": ["proposition_1", "proposition_2", ...]}

Text:
"""

class PropositionAnswer(BaseModel):
    propositions: list[str]

class PropositionsGenerator:
    def __init__(self, temperature : float = 0.7):
        self.so_llm_model = OpenAI(
            base_url=settings.base_url,
            api_key=settings.password,
        )
        self.temperature = temperature

    def proposition_prompt(self, text: str):

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instructions + user_text + text + '\nAnswer:\n'}
            ]    

            json_schema = PropositionAnswer.model_json_schema()
            extra_body = {"guided_json": json_schema}
            return  messages, extra_body
    
    def __call__(self, text: str):
        prompt, extra_body = self.proposition_prompt(text)

        completion = self.so_llm_model.chat.completions.create(
            model=settings.model,
            messages=prompt,
            extra_body=extra_body,
            temperature=self.temperature,
            max_tokens=4000,
        )
        res = json.loads(completion.choices[0].message.content)

        return res['propositions']

