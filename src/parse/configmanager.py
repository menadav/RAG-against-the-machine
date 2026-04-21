import json
from typing import Tuple
from pydantic import ValidationError
from pathlib import Path
from src.parse.models import UnansweredQuestion, StudentSearchResults
from src.parse.models import StudentSearchResultsAndAnswer
from filetype_scanner.allowed_extensions import ALLOWED_EXTENSIONS


class ConfigManager:
    def checker(
            self, max_chunk_size, config_path, extensions, path_cp
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
                                file, time = self.get_repo_status(
                                    path_cp, extensions
                                    )
                                last_time = config.get("last_modif")
                                if float(last_time) == float(time):
                                    return True
                            else:
                                raise ValueError("[WARNING] Diferrent path.\n")
        except (FileNotFoundError, json.JSONDecodeError, PermissionError):
            raise ValueError("[WARNING] File not Found create a new.\n")
        return False

    def get_repo_status(
            self, path_base, extension
            ):
        files = [f for f in Path(
            path_base
            ).rglob("*") if f.is_file() and f.suffix in extension]
        if not files:
            return 0, 0
        last_modified = max(f.stat().st_mtime for f in files)
        return len(files), last_modified

    def check_extensions(self, extension: list[str]) -> bool:
        if extension == []:
            raise ValueError("[WARNING] You need any extensio\n")
        for x in extension:
            if x in ALLOWED_EXTENSIONS:
                pass
            else:
                raise ValueError(
                    "[WARNING] The extension no is in the list.\n"
                    )
        return False

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
            self, data_set_path,
            check: bool = True
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
                return []
            validate_questions = []
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
            data: StudentSearchResults | StudentSearchResultsAndAnswer,
            name: str
            ) -> str:
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
        print(output_file)
        return str(output_file)
