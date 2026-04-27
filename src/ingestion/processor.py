from pathlib import Path
from typing import List, Tuple, Generator
from src.parse.models import MinimalSource


class DocumentProcessor:
    """Handles file reading and context-aware text chunking.

    Attributes:
        max_chunk_size (int): The target maximum size for each text segment.
    """
    def __init__(self,
                 max_chunk_size: int) -> None:
        """Initializes the processor with a defined maximum chunk size."""
        self.max_chunk_size = max_chunk_size

    def read_file(self, file_path: Path) -> str:
        """Reads the content of a file using UTF-8 encoding.

        Returns:
            str: The file content, or an empty string if reading fails.
        """
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return ""

    def splited_file(
            self,
            file_path: str,
            content: str
            ) -> Generator[Tuple[MinimalSource, str], None, None]:
        """Splits content into logical chunks based on structure.

        Uses the get_chunks method to identify optimal breaking points,
        ensuring code/markdown structure is preserved.

        Yields:
            Tuple[MinimalSource, str]: The chunk metadata and the text content.
        """
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
        """Determines the best separator index and priority for chunking.

        Args:
            extension (str): The file extension to determine logic.
            chunk_candidate (str): The current text buffer to analyze.

        Returns:
            Tuple[int, str]: (The index of the separator, priority level).
        """
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
        """Finds the last occurrence of any given separator within the text.

        Returns:
            int: The index after the separator, or -1 if none are found.
        """
        for sep in separators:
            idx = text.rfind(sep)
            if idx != -1:
                return idx + len(sep)
        return -1
