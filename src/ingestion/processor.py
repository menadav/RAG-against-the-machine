from pathlib import Path
from src.parse.models import MinimalSource


class DocumentProcessor:
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size

    def read_file(self, file_path: Path):
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return None

    def splited_file(self, file_path, content):
        path_obj = Path(file_path)
        extension = path_obj.suffix
        folder = path_obj.parent.name
        relative_path = str(file_path).replace("\\", "/")
        context_prefix = (
            f"FILE: {folder} {folder}"
            f"SOURCE: {relative_path}"
            f"CONTEXT: {path_obj.parent.name}\n"
            "---\n"
            )
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
                searchable_text = context_prefix + chunk_text
                source = MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end
                )
                yield source, searchable_text
            start = end

    def get_chunks(self, extension, chunk_candidate):
        if extension == '.py':
            separator = self._find_last_separator(
                chunk_candidate, ["\ndef ", "\nclass "]
                )
            if separator != -1:
                return separator, "high"
            separator = self._find_last_separator(
                chunk_candidate, ["\n\n", "\n"]
                )
            return separator, "low"
        elif extension == '.md':
            separator = self._find_last_separator(
                chunk_candidate, ["\n#", "\n##", "\n###", "\n####"]
                )
            if separator != -1:
                return separator, "high"
            separator = self._find_last_separator(
                chunk_candidate, ["\n", "\n\n"]
                )
            return separator, "low"
        else:
            separator = self._find_last_separator(
                chunk_candidate, ["\n\n"]
                )
            if separator != -1:
                return separator, "high"
            separator = self._find_last_separator(
                chunk_candidate, ["\n", ". "]
                )
            return separator, "low"

    def _find_last_separator(self, text, separators):
        for sep in separators:
            idx = text.rfind(sep)
            if idx != -1:
                return idx + len(sep)
        return -1
