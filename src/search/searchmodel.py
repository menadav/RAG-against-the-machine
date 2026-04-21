from tqdm import tqdm
from src.parse.models import MinimalSearchResults, StudentSearchResults


class SearchService:
    def __init__(self, retrieval):
        self.retrieval = retrieval

    def process_batch(self, questions, k: int):
        all_results = []
        for q in tqdm(questions, desc="Searching Dataset ...", unit="query"):
            sources = self.retrieval.find_top_k(q.question, k=k)
            all_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))
        return all_results

    def format_output(self, results, k: int) -> StudentSearchResults:
        return StudentSearchResults(search_results=results, k=k)
