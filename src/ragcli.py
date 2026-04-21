import sys
from typing import List
from src.constant import EXTENSIONS, PATH_CP, CONFIG_PATH, \
    DEFAULT_K, DEFAULT_CHUNK
from src.parse.configmanager import ConfigManager
from src.ingestion.processor import DocumentProcessor
from src.ingestion.ingestservice import IngestionService
from src.retrieval.retrieval_model import Retrieval
from src.parse.models import MinimalSource
from src.search.searchmodel import SearchService
from src.generator.generator_model import AnswerService


class RAGCli:
    def __init__(self):
        self.llm = "Qwen/Qwen3-0.6B"
        self.config_manager = ConfigManager()
        self.processor = DocumentProcessor(max_chunk_size=DEFAULT_CHUNK)
        self.retrieval = Retrieval()
        self.search_service = SearchService(self.retrieval)
        self.ingestion_service = IngestionService(
            processor=self.processor,
            retrieval=self.retrieval,
            config_manager=self.config_manager
        )
        self.answer_service = None

    def index(self, max_chunk_size=DEFAULT_CHUNK) -> None:
        if self.config_manager.check_extensions(EXTENSIONS):
            sys.exit(1)
        if self.config_manager.checker(
                max_chunk_size, CONFIG_PATH, EXTENSIONS, PATH_CP):
            return
        else:
            self.ingestion_service.run_pipeline(
                max_chunk_size, EXTENSIONS, PATH_CP, CONFIG_PATH
            )

    def search(self, question: str, k: int = DEFAULT_K) -> List[MinimalSource]:
        # salida .json
        return self.retrieval.find_top_k(question, k)

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str, k: int = DEFAULT_K
            ) -> None:
        try:
            questions, name = self.config_manager.search_data(dataset_path)
            if not questions:
                return
            results = self.search_service.process_batch(questions, k)
            final_output = self.search_service.format_output(results, k)
            path_final = self.config_manager._save_json(
                save_directory, final_output, name)
            print(f"Saved student_search_results to {path_final}")
        except ValueError as e:
            print(e)

    def answer(self, query: str, k: int = DEFAULT_K):
        self.answer_service = AnswerService(self.llm, self.retrieval)
        return self.answer_service.generate_answer(query, k)

    def answer_dataset(
            self,
            student_search_results_path,
            save_directory
            ) -> None:
        try:
            questions, name = self.config_manager.load_search_results(
                student_search_results_path
            )
            self.answer_service = AnswerService(self.llm, self.retrieval)
            len_questions = len(questions.search_results)
            results = self.answer_service.generate_data_answer(questions)
            final_name = self.config_manager._save_json(
                save_directory, results, name
                )
            final_len = len(results.search_results)
            print(
                f"Loaded {len_questions} "
                f"questions from {student_search_results_path}"
                )
            print(f"Processed {final_len} of {len_questions} questions")
            print(f"Saved student_search_results_and_answer to {final_name}")
        except ValueError as e:
            print(e)

    def evaluate(self) -> None:
        pass
