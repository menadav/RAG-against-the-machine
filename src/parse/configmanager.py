import json
from typing import Tuple, Union, Any, List
from pydantic import ValidationError
from pathlib import Path
from src.parse.models import UnansweredQuestion, StudentSearchResults
from src.parse.models import StudentSearchResultsAndAnswer, RagDataset
from filetype_scanner.allowed_extensions import ALLOWED_EXTENSIONS


class ConfigManager:
    def checker(
            self,
            max_chunk_size: int,
            config_path: str,
            extensions: List[str],
            path_cp: str
            ) -> bool:
        if 0 < max_chunk_size <= 2000:
            pass
        else:
            raise ValueError("[ERROR] You would need max_chunk_size 1-2000")
        try:
            config = Path(config_path)
            if config.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if config.get("max_chunk_size") == max_chunk_size:
                        same_extensions = sorted(
                            config.get("extensions", [])
                            ) == sorted(extensions)
                        if same_extensions:
                            if config.get("path") == path_cp:
                                _, time = self.get_repo_status(
                                    path_cp, extensions
                                    )
                                last_time = config.get("last_modif")
                                if last_time is not None and \
                                        abs(
                                            float(last_time) - float(time)
                                            ) < 0.001:
                                    return True
                            else:
                                raise ValueError("[WARNING] Diferrent path.\n")
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            raise ValueError("[WARNING] File not Found create a new.\n")
        return False

    def get_repo_status(
            self,
            path_base: str,
            extension: list[str]
            ) -> Tuple[int, float]:
        files = [f for f in Path(
            path_base
            ).rglob("*") if f.is_file() and f.suffix in extension]
        if not files:
            return 0, 0
        last_modified = max(f.stat().st_mtime for f in files)
        return len(files), last_modified

    def check_extensions(self, extension: list[str]) -> bool:
        if not extension:
            raise ValueError("[WARNING] You need any extensio\n")
        for x in extension:
            if x not in ALLOWED_EXTENSIONS:
                print(
                    "[WARNING] The extension no is in the list.\n"
                    )
                return True
        return False

    def load_rag_data(self, path: str) -> RagDataset:
        path_base = Path(path)
        if not path_base.exists():
            raise ValueError(f"El archivo del dataset no existe: {path}")
        with open(path_base, 'r', encoding='utf-8') as f:
            data: Any = json.load(f)
        try:
            if isinstance(data, dict):
                return RagDataset(**data)
            if isinstance(data, list):
                return RagDataset(rag_questions=data)
            raise ValueError(f"Formato de JSON inesperado: {type(data)}")
        except Exception as e:
            raise ValueError(f"Error al validar el dataset con Pydantic: {e}")

    def load_search_results(
            self, path: str
            ) -> Tuple[StudentSearchResults, str]:
        path_base = Path(path)
        if not path_base.exists():
            raise ValueError("Path incorrect")
        with open(path_base, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        try:
            validate_obj = StudentSearchResults(**full_data)
            return (validate_obj, path_base.name)
        except ValidationError as e:
            raise ValueError(f"Error de validación: {e}")

    def search_data(
            self,
            data_set_path: str
            ) -> Tuple[list[UnansweredQuestion], str]:
        path_base = Path(data_set_path)
        if not path_base.exists():
            raise ValueError("[WARNING] Path incorrect")
        file_name = path_base.name
        try:
            with open(path_base, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            if isinstance(full_data, dict):
                data = next(iter(full_data.values()))
            else:
                data = full_data
            if not isinstance(data, list):
                return [], ""
            validate_questions: List[UnansweredQuestion] = []
            for item in data:
                try:
                    validate_obj = UnansweredQuestion(**item)
                    validate_questions.append(validate_obj)
                except ValidationError as e:
                    raise ValueError(e)
            return (validate_questions, file_name)
        except PermissionError:
            raise ValueError("[WARNING] Permision Error")
        except json.JSONDecodeError:
            raise ValueError("[WARNING0 Json incorrect]")

    def _save_json(
            self, path: str,
            data: Union[StudentSearchResults | StudentSearchResultsAndAnswer],
            name: str
            ) -> str:
        prefix = "data/raw/vllm-0.10.1/"
        check = ["data/raw/vllm-0.10.1/", "vllm/vllm/", "vllm/"]
        for result in data.search_results:
            for source in result.retrieved_sources:
                path_str = source.file_path
                if path_str.startswith(prefix):
                    continue
                clean = path_str
                for bad_prefix in check:
                    if clean.startswith(bad_prefix):
                        clean = clean[len(bad_prefix):]
                if clean.startswith(("docs/", "tests/", "examples/")):
                    source.file_path = f"{prefix}{clean}"
                else:
                    source.file_path = f"{prefix}vllm/{clean}"
        base_dir = Path(path)
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
        elif not base_dir.is_dir():
            base_dir.unlink()
            base_dir.mkdir(parents=True, exist_ok=True)
        output_file = base_dir / name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            data.model_dump_json(indent=4),
            encoding="utf-8"
        )
        return str(output_file)
