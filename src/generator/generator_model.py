import uuid
from typing import List, Optional, Union
import torch
from tqdm import tqdm
from transformers import pipeline
from src.retrieval.retrieval_model import Retrieval
from src.parse.models import (
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults
)
import os
import warnings
import logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class AnswerService:
    """Handles the RAG generation pipeline
    by constructing prompts and querying the LLM.
    """
    def __init__(self, llm_model: str, retrieval: Retrieval) -> None:
        """Initializes the LLM pipeline and retrieval system."""
        self.generator = pipeline(
            "text-generation",
            model=llm_model,
            device_map="auto",
            dtype=torch.float16
        )
        self.retrieval = retrieval

    def generate_data_answer(
            self,
            questions: StudentSearchResults
            ) -> StudentSearchResultsAndAnswer:
        """Generates answers for an entire batch of search results.

        Args:
            questions (StudentSearchResults):
            Batch of queries with their retrieved sources.

        Returns:
            StudentSearchResultsAndAnswer:
            The structured dataset containing generated answers.
        """
        results = []
        for res in tqdm(questions.search_results,
                        desc="Generating answer dataset"):
            res_data = self.generate_answer(
                res.question_str, 0, res.retrieved_sources, False)
            clean_answer: str = str(res_data)
            answer_obj = MinimalAnswer(
                question_id=res.question_id,
                question_str=res.question_str,
                retrieved_sources=res.retrieved_sources,
                answer=clean_answer
            )
            results.append(answer_obj)
        return StudentSearchResultsAndAnswer(
            search_results=results,
            k=questions.k
        )

    def generate_answer(
            self,
            query: str,
            k: int = 0,
            sources: Optional[List[MinimalSource]] = None,
            check: bool = True
            ) -> Union[str, StudentSearchResultsAndAnswer]:
        """Generates a concise answer for a single query
        based on provided context.

        Args:
            query (str): The user's question.
            k (int): Number of sources to retrieve if check is True.
            sources (Optional[List[MinimalSource]]): Pre-retrieved sources.
            check (bool): Whether to perform retrieval or use provided sources.

        Returns:
            Union[str, StudentSearchResultsAndAnswer]:
            The generated answer string or structured object.
        """
        actual_sources: List[MinimalSource] = sources if sources is not None \
            else []
        if check:
            query_str = " ".join(query) if isinstance(query, list) else query
            raw_list = self.retrieval.find_top_k(query_str, k)
            actual_sources = [
                MinimalSource(
                    file_path=item.file_path,
                    first_character_index=int(item.first_character_index),
                    last_character_index=int(item.last_character_index)
                ) for item in raw_list
            ]
        else:
            actual_sources = sources or []
        context = self._build_context(actual_sources)
        prompt = self._create_prompt(query, context)
        tokenizer = getattr(self.generator, "tokenizer", None)
        eos_id = getattr(tokenizer, "eos_token_id", 0) if tokenizer else 0
        result = self.generator(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
        raw_answer: str = result[0]['generated_text'].strip()
        clean_answer: str = (
            raw_answer.split('\n')[0].replace("Answer:", "").strip()
        )
        if len(clean_answer) < 3 or\
                not any(char.isalnum() for char in clean_answer):
            clean_answer = "I don't know"
        if not check:
            return clean_answer
        final_prompt = self.get_structured_response(
            query, clean_answer, actual_sources, k
            )
        return final_prompt

    def _build_context(self, sources: List[MinimalSource]) -> str:
        """Reads and concatenates content from the specified file sources.

        Returns:
            str: A formatted string containing the content
            of all retrieved sources.
        """
        context_parts = []
        for src in sources:
            try:
                with open(src.file_path, 'r', encoding='utf-8') as f:
                    f.seek(src.first_character_index)
                    length = (
                        src.last_character_index - src.first_character_index
                        )
                    content = f.read(length)
                    context_parts.append(
                        f"Source: {src.file_path}\nContent: {content.strip()}"
                        )
            except Exception:
                continue
        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Constructs the prompt template for the LLM using context and query.

        Returns:
            str: The final prompt string.
        """
        return (
            "Context information is below.\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query concisely in 1 or 2 sentences max.\n"
            "If the answer is not in the context,"
            " strictly respond with 'I don't know'.\n"
            f"Query: {query}\n"
            "Answer: "
        )

    def get_structured_response(
            self, query: str, answer: str, sources: List, k: int
            ) -> StudentSearchResultsAndAnswer:
        """Wraps the answer and sources into a formal data schema.

        Returns:
            StudentSearchResultsAndAnswer: The formatted result object.
        """
        return StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question_str=query,
                    retrieved_sources=sources,
                    answer=answer
                )
            ],
            k=k
        )
