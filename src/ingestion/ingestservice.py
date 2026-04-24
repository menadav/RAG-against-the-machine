from pathlib import Path
from typing import List, Generator, Any, Tuple
from tqdm import tqdm
from src.parse.models import IngestConfig
from src.retrieval.retrieval_model import Retrieval
from src.ingestion.processor import DocumentProcessor
from src.parse.configmanager import ConfigManager


class IngestionService:
    def __init__(self,
                 processor: DocumentProcessor,
                 retrieval: Retrieval,
                 config_manager: ConfigManager
                 ) -> None:
        self.processor = processor
        self.retrieval = retrieval
        self.config_manager = config_manager

    def run_pipeline(self,
                     max_chunk_size: int,
                     extensions: List[str],
                     path_cp: str,
                     config_path: str
                     ) -> None:
        all_chunks_text = []
        all_sources = []
        for source, text in self._ingest_files(
                max_chunk_size, extensions, path_cp):
            all_chunks_text.append(text)
            all_sources.append(source.model_dump())
        if not all_chunks_text:
            print("[ERROR] No files found to ingest.")
            return
        self.retrieval.build_and_save(all_chunks_text, all_sources)
        self._save_ingestion_state(
            max_chunk_size, extensions, config_path, path_cp)
        print("Ingestion complete! Indices saved under data/processed/")

    def _ingest_files(self,
                      max_chunk_size: int,
                      extensions: List[str],
                      path_cp: str
                      ) -> Generator[Tuple[Any, str], None, None]:
        self.processor.max_chunk_size = max_chunk_size
        path_base = Path(path_cp)
        files = [f for f in path_base.rglob("*")
                 if f.is_file() and f.suffix in extensions]
        for fil in tqdm(files, desc="Loading Bm25 ...", unit="files"):
            content = self.processor.read_file(fil)
            if content:
                yield from self.processor.splited_file(fil.as_posix(), content)

    def _save_ingestion_state(
            self,
            max_chunk_size: int,
            extensions: List[str],
            config_path: str,
            path_cp: str
            ) -> None:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        _, check_time = self.config_manager.get_repo_status(
            path_cp, extensions)
        config = IngestConfig(
            max_chunk_size=max_chunk_size,
            extensions=extensions,
            path=str(path_cp),
            last_modif=check_time
        )
        config_file.write_text(config.model_dump_json(indent=4))
