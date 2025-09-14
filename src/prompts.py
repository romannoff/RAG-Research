from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from enum import Enum
from gigachat.models import Chat, Function, Messages, MessagesRole, FunctionParameters
from gigachat.models.chat_function_call import ChatFunctionCall


def simple_rag_prompt(user_query: str, chunks: list[str]):
    system = 'You are a RAG system that provides accurate responses based on retrieved documents.'

    istructions = """
Core Rules
- Ground in sources: Base answers on retrieved documents, not assumptions
- Cite clearly: Indicate which information comes from which sources
- Handle conflicts: When sources disagree, present both perspectives
- Stay focused: Answer the user's actual question directly
- If the context does not contain an answer to the user's question, then return "No answer"
- Answer BRIEFLY and to the point.
"""

    goal = "Goal: Be a reliable bridge between retrieved information and user needs with maximum accuracy and transparency."

    content = istructions + "\nQuestion:\n" + user_query + "\n" + goal + "Context:\n" + "\n".join(chunks) + "\nAnswer:\n"
    # messages = [
            # SystemMessage(content=system),
            # HumanMessage(content=content)
        # ]
    messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
    ]   
    
    return messages


def simple_rag_prompt_ru(user_query: str, chunks: list[str]):
    system = 'Ты - RAG система, которая предоставляет точные ответы на основе полученных документов.'

    istructions = """
Основные правила
- Опираться на источники: Основывать ответы на полученных документах, а не на предположениях
- Четко цитировать: Указывать, какая информация из каких источников получена
- Разрешать конфликты: Когда источники расходятся во мнениях, представлять обе точки зрения
- Оставаться сосредоточенным: отвечать непосредственно на актуальный вопрос пользователя
- Если контекст не содержит ответа на вопрос пользователя, то верните "Ответа нет"
- Отвечайте кратко и по существу.

"""

    goal = "Цель: Стать надежным связующим звеном между полученной информацией и потребностями пользователей с максимальной точностью и прозрачностью."

    content = istructions + "\nQuestion:\n" + user_query + "\n" + goal + "Context:\n" + "\n".join(chunks) + "\nAnswer:\n"
    # messages = [
            # SystemMessage(content=system),
            # HumanMessage(content=content)
        # ]

    messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
    ]  
    
    return messages


rewriter_function=Function(
    name="rewriter_function",
    description="rewrites the user's question so that it becomes more informative in order to retrieve the most relevant information.",
    parameters=FunctionParameters(
        type="object",
        properties={"reflection": {"type": "string", "description": "your thoughts on how best to rewrite the user's question"}, "rewritten_query": {"type": "string", "description": "the rewritten question"}},
        required=["reflection", "rewritten_query"],
        )
)


def rewriter_prompt(user_query: str, temperature):
    system = "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.\n"
    instruction = "Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n"

    schema = """Return JSON:
{"reflections": "your thoughts", "rewritten_query": "new query"}
"""  

    user_text = f"""Original query: {user_query}

Rewritten query:"""  
    
    payload = Chat(
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction + schema + user_text + '\nAnswer:\n'}
            ],
        function_call=ChatFunctionCall(name="rewriter_function"),
        functions=[
            rewriter_function
        ],
        temperature=temperature
    )
    return payload


decompositor_function=Function(
    name="decompositor_function",
    description="Breaks down a complex question into 2-3 subqueries that will help you find a better answer to the original question.",
    parameters=FunctionParameters(
        type="object",
        properties={"reflection": {"type": "string", "description": "Your thoughts on splitting the query into subqueries."}, "decomposition_queries": {"type": "array","items": {"type": "string"}, "description": "a list of 2-3 subqueries"}},
        required=["reflection", "decomposition_queries"],
        )
)


def decompositor_prompt(user_query: str, temperature):
    system = "You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.\n"
    instruction = """Given the original query, decompose it into 2-3 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
Original query:
"""

    example = """Example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?
"""

    schema = """Return JSON:
{"reflections": "your thoughts", "decomposition_queries": ["new query_1", ...]}
"""   

    payload = Chat(
        messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction + user_query + example + schema + '\nAnswer:\n'}
        ],
        function_call=ChatFunctionCall(name="decompositor_function"),
        functions=[
            decompositor_function
        ],
        temperature=temperature
    )
    return payload


hyde_function=Function(
    name="hyde_function",
    description="generates a hypothetical document that will directly answer this question.",
    parameters=FunctionParameters(
        type="object",
        properties={"reflection": {"type": "string", "description": "your thoughts on generating a hypothetical document."}, "hypothetical_document": {"type": "string", "description": "hypothetical document that will directly answer this question."}},
        required=["reflection", "hypothetical_document"],
        )
)


def hyde_prompt(user_query: str, temperature):
    system = "You are an AI assistant who should generate a document that hypothetically could contain a response to a user's query."
    instruction = f"\nGiven the question '{user_query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth. the document size has be exactly 400 characters."

    schema = """\nReturn JSON:
{"reflections": "your thoughts", "hypothetical_document": "document"}
"""      

    payload = Chat(
        messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction + schema + '\nAnswer:\n'}
        ],
        function_call=ChatFunctionCall(name="hyde_function"),
        functions=[
            hyde_function
        ],
        temperature=temperature
    )
    return payload


reranker_function=Function(
    name="reranker_function",
    description="Evaluates whether the document is relevant to the user's request",
    parameters=FunctionParameters(
        type="object",
        properties={"reflection": {"type": "string", "description": "Reasoning about the document's compliance with the request"}, "label": {"type": "string", "enum": ["0", "1"], "description": "1 - the document matches the request, 0 - not"}},
        required=["reflection", "label"],
        )
)


def reranker_prompt(user_query: str, chunk: str, temperature):
    system = "You are a binary classifier that compares a user query and a document.\n"
    instruction = "Your task is to evaluate the document and return either 1 or 0. Return the value 1 if the document contains even indirect information related to the user's request. Return 0 if the document does not contain ANY information at the user's request.\n"

    instruction2 = "Even if the document does not fully answer the question, but only reveals its sub-topic, then assign it the label 1\n"

    user_input = f"""
Query: {user_query}
Document: 
{chunk}
"""

    schema = """\nReturn JSON:
{"reflections": "your short thoughts", "label": 0 or 1}
"""  
# "reflections": "your short thoughts", 

    payload = Chat(
        messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input + instruction + instruction2 + schema + '\nAnswer:\n'}
        ],
        function_call=ChatFunctionCall(name="reranker_function"),
        functions=[
            reranker_function
        ],
        temperature=temperature
    )
    return payload