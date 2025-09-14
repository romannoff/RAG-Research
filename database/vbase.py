import uuid
from typing import List, Dict, Any, Optional
from scipy.special import logsumexp
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np


class QdrantBase:
    def __init__(
            self, 
            vector_dimension: int, 
            model_name: str, 
            collection_name: str = 'collection',
            host: str = "localhost", 
            port: int = 6333, 
            timeout: int = 1000,
            custom_search = None,
            device='cpu'
            ):
        """
        Инициализация клиента Qdrant и настройка коллекций
        
        Args:
            vector_dimension: размерность векторов модели
            model_name: название векторизатора
            host: хост Qdrant сервера
            port: порт Qdrant сервера 
            custom_search: кастомная функция для поиска чанков 
        """
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.encoder = SentenceTransformer(model_name, device=device)
        
        self.custom_search = custom_search

        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        
        # Инициализируем коллекции
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Создание коллекций в Qdrant"""

        try:
            # Проверяем, существует ли коллекция
            self.client.get_collection(self.collection_name)
            # logger.info(f"Коллекция {collection_name} уже существует")
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE
                )
            )
            # logger.info(f"Создана коллекция {collection_name}")
    
    def _create_embeddings(self, text: list[str]) -> np.ndarray:
        """
        Создание эмбеддингов для текстов
        
        Args:
            texts: список текстов для векторизации
            
        Returns:
            массив векторов
        """
        return self.encoder.encode(text)
    

    def search(self, query: str, limit: int):
        """
        Поиск по запросу
        
        Args:
            query: Запрос
            limit: Количество чанков
            
        Returns:
            Список чанков со score
        """
        query_embedding = self._create_embeddings([query])[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            with_vectors=False,
        )
        
        # Преобразуем результаты
        results = []
        for point in search_result.points:
            result = {
                **point.payload,
                'score': point.score 
            }
            results.append(result)
            
        return results


    def find_segments(self, query: str):
        query_embedding = self._create_embeddings([query])[0]

        collection_info = self.client.get_collection(self.collection_name)
        total_points = collection_info.points_count

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=total_points,
            with_vectors=True,
            # score_threshold=0.2,
        )
        
        # Преобразуем результаты
        results = []
        for point in search_result.points:
            result = {
                **point.payload,
                'score': point.score 
            }
            results.append(result)

        results = sorted(results, key=lambda x: x['id'])
        return results
        

    def add_point(self, chunks: list, payload: dict | None = None):
        """
        Добавление чанков в векторную базу данных

        Args:
            chunks: список чанков
            payload: мета-данные о чанке
        """
    
        embeddings = self._create_embeddings(chunks)
            
        points = []

        # Создаем точки для каждого чанка
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            if payload is not None:
                payload['chunk_text'] = chunk
            else:
                payload = {'chunk_text': chunk}
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
    
        # Загружаем точки в коллекцию
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_point_by_ids(self, point_ids: list[int]):
        """
        Получить точку по ID
        
        Args:
            point_ids: ID точек
            
        Returns:
            Словарь с данными точки или None, если точка не найдена
        """
        result = []

        for point_id in point_ids:

            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="id",
                        match=MatchValue(value=point_id)
                    )
                ]
            )
            
            # Используем limit=1, так как значение уникально
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=True,
                with_vectors=False  # Векторы обычно не нужны для простого получения
            )
            
            points, _ = response
            if points:
                result.append(points[0].payload)
        return result
    
    def dartboard_search(
            self, 
            query: str, 
            limit=10,
            diversity_weight: float = 1.0,
            relevance_weight: float = 1.0,
            sigma: float = 0.1,
            ):
        query_embedding = self._create_embeddings([query])[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=limit * 3,
            with_vectors=True,
        )
        
        # Преобразуем результаты
        texts = []
        scores = []
        vectors = []
        for point in search_result.points:
            texts.append(point.payload['chunk_text'])
            scores.append(point.score)
            vectors.append(point.vector)

        vectors = np.array(vectors)
        scores = np.array(scores)

        document_distances = 1 - np.dot(vectors, vectors.T)

        result = self.dartboard_processing(
            query_distances=scores,
            document_distances=document_distances,
            documents=texts,
            limit=limit,
            sigma=sigma,
            diversity_weight=diversity_weight,
            relevance_weight=relevance_weight,
        )
        return result

    
    @staticmethod
    def lognorm(dist:np.ndarray, sigma:float):
        """
        Calculate the log-normal probability for a given distance and sigma.
        """
        if sigma < 1e-9: 
            return -np.inf * dist
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)
    
    def dartboard_processing(
            self,
            query_distances: np.ndarray,
            document_distances: np.ndarray,
            documents: list[str],
            limit: int,
            sigma: float,
            diversity_weight: float,
            relevance_weight: float,
            ):
        # Avoid division by zero in probability calculations
        sigma = max(sigma, 1e-5)
        
        # Convert distances to probability distributions
        query_probabilities = query_distances
        document_probabilities = document_distances
        
        # Initialize with most relevant document
        most_relevant_idx = np.argmax(query_probabilities)
        selected_indices = np.array([most_relevant_idx])
        selection_scores = [1.0] # dummy score for the first document
        # Get initial distances from the first selected document
        max_distances = document_probabilities[most_relevant_idx]
        
        # Select remaining documents
        while len(selected_indices) < limit:
            # Update maximum distances considering new document
            updated_distances = np.maximum(max_distances, document_probabilities)
            
            # Calculate combined diversity and relevance scores
            combined_scores = (
                updated_distances * diversity_weight +
                query_probabilities * relevance_weight
            )
            
            # Normalize scores and mask already selected documents
            normalized_scores = logsumexp(combined_scores, axis=1)
            normalized_scores[selected_indices] = -np.inf
            
            # Select best remaining document
            best_idx = np.argmax(normalized_scores)
            best_score = np.max(normalized_scores)
            
            # Update tracking variables
            max_distances = updated_distances[best_idx]
            selected_indices = np.append(selected_indices, best_idx)
            selection_scores.append(best_score)
        
        # Return selected documents and their scores
        selected_documents = [documents[i] for i in selected_indices]
        return selected_documents