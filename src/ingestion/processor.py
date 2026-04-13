import os
from pathlib import Path

class DocumentProcessor:
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size

    def is_supported(self, file_path):
        """Verifica si la extensión es válida."""
        return Path(file_path).suffix in self.supported_extensions

    def read_file(self, file_path):
        """Lee el contenido manejando posibles errores de codificación."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            print(f"[SKIP] Archivo no legible (binario?): {file_path}")
            return None
        except Exception as e:
            print(f"[ERROR] No se pudo leer {file_path}: {e}")
            return None

    def get_chunks(self, text, file_path):
        """Aplica la lógica de segmentación según la extensión."""
        extension = Path(file_path).suffix
        if extension == '.py':
            return self._chunk_python(text)
        elif extension == '.md':
            return self._chunk_markdown(text)
        elif extension  == '.json':
            return self._chunk_json(text)
        else:
            return self._chunk_generic(text)

    def _chunk_python(self, text):
        return self._chunk_generic(text)

    def _chunk_markdown(self, text):
        return self._chunk_generic(text)

    def _chunk_json(text):
        return self._chunk_generic(text) 

    def _chunk_generic(self, text):
        return chunks
