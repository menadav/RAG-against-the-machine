import sys
import json
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
    """Command Line Interface for the RAG system operations.

    Orchestrates the ingestion, search, answering,
    and evaluation pipelines.
    """
    def __init__(self) -> None:
        """Initializes the CLI services, configuration manager, and models."""
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
        """Ingests repository files and builds the search index.

        Args:
            max_chunk_size (int): Character limit per chunk for the processor.
        """
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
        except ValueError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def search(self, question: str, k: int = DEFAULT_K) -> None:
        """Performs a direct hybrid search query and displays results.

        Args:
            question (str): The search query.
            k (int): Number of top results to retrieve.
        """
        try:
            if self.config_manager.checker_k(k):
                sys.exit(1)
            results = self.retrieval.find_top_k(question, k)
            data_to_print = [r.model_dump() for r in results]
            print(json.dumps(data_to_print, indent=4, ensure_ascii=False))
        except ValueError:
            print("[WARNING] Incorrect Value", file=sys.stderr)
            sys.exit(1)

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str,
            k: int = DEFAULT_K
            ) -> None:
        """Processes a batch of questions from a file and saves search results.

        Args:
            dataset_path (str): Path to the JSON containing questions.
            save_directory (str): Directory where results will be saved.
            k (int): Number of sources per result.
        """
        try:
            if not self.config_manager.checker_k(k):
                sys.exit(1)
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
        """Generates an LLM answer for a single query using retrieved context.

        Args:
            query (str): The user's question.
            k (int): Number of context sources to retrieve.
        """
        try:
            if self.config_manager.checker_k(k):
                sys.exit(1)
            self.answer_service = AnswerService(self.llm, self.retrieval)
            result_object = self.answer_service.generate_answer(
                    query,
                    k,
                    None,
                    True
                    )
            print(result_object)
            if hasattr(result_object, "model_dump"):
                dict_data = result_object.model_dump()
            else:
                dict_data = result_object
            print(json.dumps(dict_data, indent=4, ensure_ascii=False))
        except ValueError:
            print("[WARNING] Incorrect Flag", file=sys.stderr)
            sys.exit(1)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        """Generates answers for a batch of existing search results.

        Args:
            student_search_results_path (str): Path to the input JSON results.
            save_directory (str): Destination for the final answers.
        """
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
            dataset_path: str
            ) -> None:
        """Evaluates model performance using Recall@K metrics.

        Args:
            student_answer_path (str): Path to the generated student results.
            dataset_path (str): Path to the ground truth RAG dataset.
            k (int): Metric depth for evaluation.
            max_context_length (int): Constraint for context evaluation.
        """
        try:
            check = False
            data, _ = self.config_manager.load_search_results(
                student_answer_path)
            data_rag = self.config_manager.load_rag_data(dataset_path)
            check = True
            long = len(data_rag.rag_questions)
            fi, k_levels = calculate_recall_at_k(
                data,
                data_rag,
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
