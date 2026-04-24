import sys
from typing import Optional
from src.constant import EXTENSIONS, PATH_CP, CONFIG_PATH, \
    DEFAULT_K, DEFAULT_CHUNK, LLM
from src.parse.configmanager import ConfigManager
from src.ingestion.processor import DocumentProcessor
from src.ingestion.ingestservice import IngestionService
from src.retrieval.retrieval_model import Retrieval
from src.search.searchmodel import SearchService
from src.generator.generator_model import AnswerService
from src.evaluate.func_eval import calculate_recall_at_k


class RAGCli:
    def __init__(self) -> None:
        self.llm: str = LLM
        self.config_manager = ConfigManager()
        self.processor = DocumentProcessor(max_chunk_size=DEFAULT_CHUNK)
        self.retrieval = Retrieval()
        self.search_service = SearchService(self.retrieval)
        self.ingestion_service = IngestionService(
            processor=self.processor,
            retrieval=self.retrieval,
            config_manager=self.config_manager
        )
        self.answer_service: Optional[AnswerService] = None

    def index(self, max_chunk_size: int = DEFAULT_CHUNK) -> None:
        try:
            if self.config_manager.check_extensions(EXTENSIONS):
                sys.exit(1)
            if self.config_manager.checker(
                    max_chunk_size, CONFIG_PATH, EXTENSIONS, PATH_CP):
                return
            else:
                self.ingestion_service.run_pipeline(
                    max_chunk_size, EXTENSIONS, PATH_CP, CONFIG_PATH
                )
        except ValueError:
            print("[WARNING] Incorrect Flag", file=sys.stderr)
            sys.exit(1)

    def search(self, question: str, k: int = DEFAULT_K) -> None:
        try:
            self.retrieval.find_top_k(question, k)
        except ValueError:
            print("[WARNING] Incorrect Value", file=sys.stderr)
            sys.exit(1)

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str,
            k: int = DEFAULT_K
            ) -> None:
        try:
            questions, name = self.config_manager.search_data(dataset_path)
            if not questions:
                return
            results = self.search_service.process_batch(questions, k)
            final_output = self.search_service.format_output(results, k)
            path_final = self.config_manager._save_json(
                save_directory, final_output, name)
            print(
                f"Saved student_search_results to {path_final}",
                file=sys.stderr
                )
        except ValueError as e:
            print(e)
            sys.exit(1)
        except TypeError:
            print("[WARNING] Incorrect Flag", file=sys.stderr)
            sys.exit(1)

    def answer(
            self, query: str, k: int = DEFAULT_K
               ) -> None:
        try:
            self.answer_service = AnswerService(self.llm, self.retrieval)
            self.answer_service.generate_answer(query, k)
        except ValueError:
            print("[WARNING] Incorrect Flag", file=sys.stderr)
            sys.exit(1)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
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
            sys.exit(1)
        except TypeError:
            print("[WARNING] Incorrect Flag", file=sys.stderr)
            sys.exit(1)

    def evaluate(
            self, student_answer_path: str,
            dataset_path: str,
            k: int,
            max_context_length: int
            ) -> None:
        try:
            check = False
            data, _ = self.config_manager.load_search_results(
                student_answer_path)
            data_rag = self.config_manager.load_rag_data(dataset_path)
            check = True
            long = len(data_rag.rag_questions)
            k_levels = sorted(list(set([1, 3, 5, k])))
            fi = calculate_recall_at_k(
                data,
                data_rag,
                k_values=k_levels,
                max_context=max_context_length
            )
            print(f"Student data is valid: {check}")
            print(f"Total number of questions: {long}")
            print("Evaluation Results")
            print("========================================")
            for i, k_val in enumerate(k_levels):
                print(f"Recall@{k_val}: {fi[i]:.2f}")
        except ValueError as e:
            if e:
                print(
                    f"{e}\n"
                    f"\nStudent data is valid: {check}", file=sys.stderr
                    )
            else:
                print("[ERROR] Incorrect Value", file=sys.stderr)
            sys.exit(1)
