from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


class BM25Search():
    def __init__(
            self,
            collection_name: str = 'collection',
            host: str = "localhost", 
            port: int = 6333, 
            timeout: int = 1000,
            ):
        client = QdrantClient(host=host, port=port, timeout=timeout)

        self.all_docs = []  # будем хранить (id, text)
        offset = None

        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            self.all_docs.extend([
                (point.payload['id'], point.payload['chunk_text'])
                for point in points
            ])
            
            if offset is None:
                break
        
        tokenized_texts = [word_tokenize(doc[1].lower()) for doc in self.all_docs]
    
        self.model = BM25Okapi(tokenized_texts)
    
    def search(self, query: str, top_k: int = 5):
        tokenized_query = word_tokenize(query.lower())
    
        scores = self.model.get_scores(tokenized_query)
        # связываем каждый документ с его скором
        scored_docs = sorted(
            zip(self.all_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # возвращаем список словарей с id и текстом
        return [doc[1] for doc, _ in scored_docs[:top_k]], [doc[0] for doc, _ in scored_docs[:top_k]]
        

if __name__ == "__main__":
    bm = BM25Search(collection_name='natural_questions')
    results = bm.search('nobel prize', 3)
    for r in results:
        print(r)
