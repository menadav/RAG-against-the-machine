from tqdm import tqdm
from typing import List, Any
from src.parse.models import MinimalSearchResults, \
    StudentSearchResults, UnansweredQuestion


class SearchService:
    """
    Handles the orchestration of search
    queries across the retrieval system.
    """
    def __init__(self, retrieval: Any) -> None:
        """Initializes the service with a retrieval engine instance."""
        self.retrieval = retrieval

    def process_batch(self,
                      questions: List[UnansweredQuestion],
                      k: int
                      ) -> List[MinimalSearchResults]:
        """Processes a list of questions and retrieves
        the top-k sources for each.

        Args:
            questions (List[UnansweredQuestion]): List of questions to process.
            k (int): Number of results to retrieve per question.

        Returns:
            List[MinimalSearchResults]: List containing
            the mapping of questions
            to their retrieved sources.
        """
        all_results: List[MinimalSearchResults] = []
        for q in tqdm(questions, desc="Searching Dataset ...", unit="query"):
            sources = self.retrieval.find_top_k(q.question, k=k)
            all_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question_str=q.question,
                retrieved_sources=sources
            ))
        return all_results

    def format_output(
            self,
            results: List[MinimalSearchResults],
            k: int) -> StudentSearchResults:
        """Packages the search results into a final student result format.

        Args:
            results (List[MinimalSearchResults]): Processed search results.
            k (int): Number of neighbors retrieved.

        Returns:
            StudentSearchResults: The structured output object.
        """
        return StudentSearchResults(search_results=results, k=k)
