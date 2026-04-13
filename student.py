import fire
import os
import json
import sys
from src.parse.check_repit import checker, check_extensions
from src.ingestion.ingest_step  import ingest
from filetype_scanner.cp_extensions import EXTENSIONS, PATH_CP

class Pipeline:
    def index(self, max_chunk_size=2000) -> None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(BASE_DIR, "data", "processed", "config.json")
        if check_extensions(EXTENSIONS, PATH_CP):
            sys.exit(0)
        if checker(max_chunk_size, config_path, EXTENSIONS):
            return
        else:
            ingest(max_chunk_size, EXTENSIONS, PATH_CP)
            print(f"Ingestion complete! Indices saved under data/processed/")



    def answer(self, question):
        print(f"Buscando respuesta para: {question}")

if __name__ == "__main__":
    fire.Fire(Pipeline)
