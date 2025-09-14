from gigachat import GigaChat, context
import bisect

from database.vbase import QdrantBase
from src.model import Model, ModelAnswer
from src.config import Config
from src.prompts import simple_rag_prompt

settings = Config.from_yaml("config.yaml")
context.session_id_cvar.set(settings.session_id)


class SegmentExtractionRAG(Model):
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3', 
                 vector_dimension: int = 1024,
                 answer_temperature: float = 0.7,
                 collection_name: str = 'collection',
                ):

        self.database = QdrantBase(
            model_name=model_name,
            vector_dimension=vector_dimension,
            collection_name=collection_name,
        )

        self.answer_temperature = answer_temperature

        self.llm_model = GigaChat(credentials=settings.token, verify_ssl_certs=False)

    def __call__(self, query: str) -> ModelAnswer:
        chunks = self.database.find_segments(query=query)
        
        scores = [chunk['score'] for chunk in chunks]
        segments = self.get_best_segments(scores)
        
        new_chunks = []

        for segment in segments:
            new_chunks += chunks[segment[0]:segment[1]]
        
        chunks = [chunk['chunk_text'] for chunk in new_chunks]

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

    
    def get_best_segments(self,
                        relevance_values: list,
                        max_length: int = 5,
                        overall_max_length: int = 10,
                        minimum_value: float = 0.5,
                        alpha_max: float = 0.5,
                        alpha_topk: float = 0.25,
                        alpha_mean: float = 0.1,
                        length_bonus: float = 0.15,
                        topk: int = 2,
                        min_segment_length: int = 2,
                        gap_merge: int = 1):
        """

        SCORE = alpha_max*max + alpha_topk*mean_topk + alpha_mean*mean + length_bonus*(length/max_length)

        - фильтруем кандидаты по minimum_value
        - решаем DP для неперекрывающихся сегментов
        - затем: расширяем выбранные сегменты туда, где есть свободная вместимость и соседние чанки неотрицательны
        - затем: объединяем сегменты, между которыми gap <= gap_merge

        Параметры length_bonus и min_segment_length помогут получить длиннее группы.
        """
        n = len(relevance_values)
        if n == 0:
            return []

        # 1) генерируем кандидатов
        candidates = []
        for start in range(n):
            for end in range(start + 1, min(n, start + max_length) + 1):
                length = end - start
                window = relevance_values[start:end]
                s_sum = sum(window)
                s_mean = s_sum / length
                s_max = max(window)
                k = min(topk, length)
                mean_topk = sum(sorted(window, reverse=True)[:k]) / k

                # бонус за длину (нормируем на max_length)
                bonus = (length / max_length) if length > 0 else 0.0

                score = (alpha_max * s_max +
                        alpha_topk * mean_topk +
                        alpha_mean * s_mean +
                        length_bonus * bonus)

                candidates.append({
                    "start": start,
                    "end": end,
                    "length": length,
                    "score": score,
                    "s_max": s_max,
                    "s_mean": s_mean,
                    "mean_topk": mean_topk
                })

        # 2) фильтр кандидатов по порогу и минимальной длине
        candidates = [c for c in candidates if c["score"] >= minimum_value and c["length"] >= min_segment_length]
        if not candidates:
            return []

        # 3) сортируем по end и готовим p[i]
        candidates.sort(key=lambda c: c["end"])
        m = len(candidates)
        ends = [candidates[i]["end"] for i in range(m)]
        starts = [candidates[i]["start"] for i in range(m)]
        p = [-1] * m
        for i in range(m):
            j = bisect.bisect_right(ends, starts[i]) - 1
            p[i] = j

        # 4) DP (weighted interval scheduling с capacity)
        C = overall_max_length
        dp = [[0.0] * (C + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            cand = candidates[i - 1]
            w = cand["length"]
            v = cand["score"]
            for cap in range(C + 1):
                no_take = dp[i - 1][cap]
                take = no_take
                if w <= cap:
                    prev_idx = p[i - 1]
                    if prev_idx >= 0:
                        take = dp[prev_idx + 1][cap - w] + v
                    else:
                        take = v
                dp[i][cap] = max(no_take, take)

        # восстановление
        chosen = []
        cap = C
        i = m
        while i > 0:
            if dp[i][cap] == dp[i - 1][cap]:
                i -= 1
            else:
                cand = candidates[i - 1]
                chosen.append([cand["start"], cand["end"]])
                cap -= cand["length"]
                i = p[i - 1] + 1
        chosen.reverse()

        # 5) Попытка расширить каждый выбранный сегмент (если осталась capacity)
        #    расширяем влево/вправо пока в документе есть места и суммарная длина не превышает overall_max_length
        used_len = sum(e - s for s, e in chosen)
        free_capacity = overall_max_length - used_len
        if free_capacity > 0:
            for seg in chosen:
                while free_capacity > 0 and seg[0] > 0 and relevance_values[seg[0] - 1] >= 0:
                    seg[0] -= 1
                    free_capacity -= 1
                while free_capacity > 0 and seg[1] < n and relevance_values[seg[1]] >= 0:
                    seg[1] += 1
                    free_capacity -= 1

        if not chosen:
            return []
        merged = []
        cur_s, cur_e = chosen[0]
        for s, e in chosen[1:]:
            if s - cur_e <= gap_merge:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        return merged



if __name__ == '__main__':
    model = SegmentExtractionRAG(collection_name='natural_questions')
    res = model('where did they film high school musical two')
    print(res['answer'])
    print(res['context'])
    
    res = model('when is the next deadpool movie being released')
    print(res['answer'])
    print(res['context'])


