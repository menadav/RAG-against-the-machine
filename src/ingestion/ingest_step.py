from pathlib import Path
from src.ingestion.processor import DocumentProcessor
#from src.parse.models import

def ingest(max_chunk_size, extension, config_path):
    processor = DocumentProcessor(max_chunk_size, extension)
    path_base = Path(config_path)
    for file in path_base.rglob("*"):
        if file.is_file() and file.suffix in extension:
            content = processor.read_file(file)
            if content:
                yield from processor.splited_file(content)
    ingest_extension(max_chunk_size, extension, config_path)


def ingest_extension(max_chunk_size, ALLOWED_EXTENSION):
    # control de erroes perfeccionar
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config_data = { 
             "max_chunk_size": max_chunk_size,
             "extensions": ALLOWED_EXTENSION
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, ident=4)

