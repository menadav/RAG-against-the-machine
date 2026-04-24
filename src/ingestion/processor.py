from pathlib import Path
from typing import List, Tuple, Generator
from src.parse.models import MinimalSource


class DocumentProcessor:
    def __init__(self,
                 max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def read_file(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return ""

    def splited_file(
            self,
            file_path: str,
            content: str
            ) -> Generator[Tuple[MinimalSource, str], None, None]:
        path_obj = Path(file_path)
        extension = path_obj.suffix
        start = 0
        text_len = len(content)
        while start < text_len:
            end = min(start + self.max_chunk_size, text_len)
            if end < text_len:
                chunk_candidate = content[start:end]
                separator, priority = self.get_chunks(
                    extension, chunk_candidate)
                threshold = 0.8 if priority == "high" else 0.9
                if separator != -1 and separator > (
                        self.max_chunk_size * threshold):
                    end = start + separator
            chunk_text = content[start:end]
            if len(chunk_text.strip()) > 0:
                source = MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end
                )
                yield source, chunk_text
            start = end

    def get_chunks(
            self, extension: str,
            chunk_candidate: str
            ) -> Tuple[int, str]:
        ext = extension.lower()
        if ext == '.py':
            separator = self._find_last_separator(
                chunk_candidate, ["\ndef ", "\nclass "]
                )
            if separator != -1:
                return separator, "high"
            return self._find_last_separator(
                chunk_candidate, ["\n\n", "\n"]
                ), "low"
        elif ext == '.md':
            separator = self._find_last_separator(
                chunk_candidate, ["\n#", "\n##", "\n###"]
                )
            if separator != -1:
                return separator, "high"
            return self._find_last_separator(
                chunk_candidate, ["\n\n", "\n"]
                ), "low"
        else:
            separator = self._find_last_separator(chunk_candidate, ["\n\n"])
            if separator != -1:
                return separator, "high"
            return self._find_last_separator(
                chunk_candidate, ["\n", "; ", ". "]
                ), "low"

    def _find_last_separator(
            self,
            text: str,
            separators: List[str]
            ) -> int:
        for sep in separators:
            idx = text.rfind(sep)
            if idx != -1:
                return idx + len(sep)
        return -1
