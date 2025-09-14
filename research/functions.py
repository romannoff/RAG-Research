from bs4 import BeautifulSoup
import re

def extract_text_from_tokens(tokens):
 
    text_tokens = []
    for token in tokens:
        if token and not token.startswith('<'):
            text_tokens.append(token)
    
    text = ' '.join(text_tokens)

    text = re.sub(r' +', ' ', text)
    text = text.strip()

    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
     
    return text

def extract_text_from_nq_document(document):
    """
    Извлекает читаемый текст из документа Natural Questions
    """
    html_content = document['document']['html']
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text()
    
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text

def get_short_answer(example):
    """
    Извлекает short_answer из примера Natural Questions датасета.
    
    Args:
        example: пример из датасета с полями 'annotations' и 'document'
    
    Returns:
        str: текст short_answer или пустая строка, если ответа нет
    """
    annotations = example['annotations']
    
    short_answers = annotations['short_answers']
    
    for answer in short_answers:
        if answer['text']:
            return answer['text'][0]
    return ''

def get_natural_questions_sample(document, get_context=True):
    question = document['question']['text']
    answer = get_short_answer(document)
    if get_context:
        context = extract_text_from_nq_document(document)
    else:
        context = None
    
    return question, context, answer


def get_ms_marco_sample(document, get_context=True):
    question = document['query']
    answer = document['answers']
    if get_context:
        context = document['passages']['passage_text']
    else:
        context = None

    if not answer:
        answer = 'No answer'
    else:
        answer = answer[0]
    
    return question, context, answer