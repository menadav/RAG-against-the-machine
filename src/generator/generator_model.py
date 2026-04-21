import uuid
from typing import List
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
    def __init__(self, llm_model: str, retrieval: Retrieval) -> None:
        self.generator = pipeline(
            "text-generation",
            model=llm_model,
            device_map="auto",
            dtype=torch.float16
        )
        self.retrieval = retrieval

    def generate_data_answer(self, questions: StudentSearchResults):
        results = []
        for res in tqdm(questions.search_results,
                        desc="Generating answer dataset"):
            clean_answer = self.generate_answer(
                res.question, 0, res.retrieved_sources, False)
            answer_obj = MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=clean_answer
            )
            results.append(answer_obj)
        return StudentSearchResultsAndAnswer(
            search_results=results,
            k=questions.k
        )

    def generate_answer(
            self, query: str, k: int = 0, sources=None, check: bool = True
            ) -> str:
        if check:
            sources = self.retrieval.find_top_k(query, k)
        context = self._build_context(sources)
        prompt = self._create_prompt(query, context)
        result = self.generator(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            eos_token_id=self.generator.tokenizer.eos_token_id,
        )
        raw_answer = result[0]['generated_text'].strip()
        clean_answer = raw_answer.split('\n')[0].replace("Answer:", "").strip()
        if len(clean_answer) < 3 or\
                not any(char.isalnum() for char in clean_answer):
            clean_answer = "I don't know"
        if not check:
            return clean_answer
        return self.get_structured_response(query, clean_answer, sources, k)

    def _build_context(self, sources: List[MinimalSource]) -> str:
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
        return StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=sources,
                    answer=answer
                )
            ],
            k=k
        )
