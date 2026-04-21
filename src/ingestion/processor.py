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
        extension = file_path.suffix
        start = 0
        text_len = len(content)
        while start < text_len:
            end = min(start + self.max_chunk_size, text_len)
            chunk_candidate = content[start:end]
            separator = self.get_chunks(extension, chunk_candidate)
            if separator != -1 and separator > 0:
                end = start + separator
            chunk_text = content[start:end]
            if end <= start:
                end = min(start + self.max_chunk_size, text_len)
            chunk_text = content[start:end]
            if len(chunk_text.strip()) > 0:
                source = MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end
                )
                yield source, chunk_text
            start = end

    def get_chunks(self, extension, chunk_candidate):
        """Aplica la lógica de segmentación según la extensión."""
        if extension == '.py':
            separator = self._find_last_separator(
                chunk_candidate, ["\ndef ", "\nclass ", "\n\n", "\n"]
                )
        elif extension == '.md':
            separator = self._find_last_separator(
                chunk_candidate, ["\n#", "\n\n", "\n"]
                )
        else:
            separator = self._find_last_separator(
                chunk_candidate, ["\n\n", "\n", ". "]
                )
        return separator

    def _find_last_separator(self, text, separators):
        """Busca el último separador disponible para no cortar abruptamente."""
        for sep in separators:
            idx = text.rfind(sep)
            if idx != -1:
                return idx + len(sep)
        return -1
