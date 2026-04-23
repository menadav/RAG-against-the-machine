import os
import bm25s
import json
import chromadb
from typing import Any, List, Dict, cast
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from pathlib import Path
from src.parse.models import MinimalSource
from tqdm import tqdm
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("bm25s").setLevel(logging.ERROR)


class Retrieval:
    metadata: List[Dict[str, Any]] = []
    index_dir: Path
    metadata_path: Path
    retrieval: Any

    def __init__(
            self,
            index_dir: str = "data/processed/bm25_index",
            metadata_path: str = "data/processed/chunks/chunk.json",
            chroma_path: str = "data/processed/chroma_db"
            ) -> None:
        self.index_dir = Path(index_dir)
        self.metadata_path = Path(metadata_path)
        self.retrieval = None
        self.metadata: List[Dict[str, Any]] = []
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="vllm_docs",
            embedding_function=cast(EmbeddingFunction, self.emb_fn)
        )

    def build_and_save(
            self,
            all_chunks_text: List[str],
            all_sources: List[Dict[str, Any]]
            ) -> None:
        tokenized_corpus = bm25s.tokenize(
            all_chunks_text, stopwords="en", show_progress=False
            )
        self.retrieval = bm25s.BM25(corpus=all_chunks_text)
        self.retrieval.index(tokenized_corpus, show_progress=False)
        self.retrieval.corpus = None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval.save(self.index_dir)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        for i, meta in enumerate(all_sources):
            meta["content"] = all_chunks_text[i]
        with open(self.metadata_path, "w") as f:
            json.dump(all_sources, f, indent=4)
        batch_size = 100
        total_chunks = len(all_chunks_text)
        for i in tqdm(range(
                0, total_chunks, batch_size), desc="Loading chromadb ..."):
            end_idx = min(i + batch_size, total_chunks)
            batch_texts = all_chunks_text[i:end_idx]
            batch_metas = all_sources[i:end_idx]
            batch_ids = [str(j) for j in range(i, end_idx)]
            clean_metas = []
            for meta in batch_metas:
                clean_meta = {
                    k: (v if isinstance(v, (
                        str, int, float, bool
                        )) or v is None else str(v))
                    for k, v in meta.items()
                }
                clean_metas.append(clean_meta)
            self.collection.add(
                documents=batch_texts,
                metadatas=clean_meta,
                ids=batch_ids
            )

    def load(self) -> None:
        if not self.index_dir.exists():
            raise ValueError("[ERROR] No index Found")
        self.retrieval = bm25s.BM25.load(self.index_dir, load_corpus=True)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def find_top_k(
            self,
            query_tokens: list,
            k: int
            ) -> list[MinimalSource]:
        if self.retrieval is None:
            self.load()
        query_str = " ".join(query_tokens)
        query_clean = " ".join(
            char if char.isalnum() or char.isspace() or char in "_." else " "
            for char in query_str.lower()
        )
        query_list = query_clean.split()
        bm25_indices, scores = self.retrieval.retrieve([query_list], k=k)
        query_str = " ".join(query_tokens)
        chroma_results = self.collection.query(
            query_texts=[query_str], n_results=k)
        top_k_indices = bm25_indices.flatten()
        bm25_scores = scores.flatten()
        final_sources: list[MinimalSource] = []
        seen_paths: set[str] = set()
        for i, idx in enumerate(top_k_indices):
            if bm25_scores[i] > 1:
                meta = self.metadata[idx]
                self._add_to_final(final_sources, seen_paths, meta)
        raw_metadatas = chroma_results.get("metadatas")
        if isinstance(raw_metadatas, list) and len(raw_metadatas) > 0:
            metadatas_list = raw_metadatas[0]
            if isinstance(metadatas_list, list):
                for meta in metadatas_list:
                    if len(final_sources) >= k:
                        break
                    if isinstance(meta, dict):
                        self._add_to_final(final_sources, seen_paths, meta)
        return final_sources

    def _add_to_final(
            self,
            final_list: List[MinimalSource],
            seen_set: set[str],
            meta: dict[str, Any]
            ) -> None:
        f_path = str(meta.get("file_path", ""))
        f_idx = int(meta.get("first_character_index", 0))
        l_idx = int(meta.get("last_character_index", 0))
        unique_key = f"{f_path}_{f_idx}"
        if unique_key not in seen_set:
            clean_path = f_path.split("vllm-0.10.1/")[-1] \
                if "vllm-0.10.1/" in f_path \
                else os.path.basename(f_path)
            source = MinimalSource(
                file_path=clean_path,
                first_character_index=f_idx,
                last_character_index=l_idx
            )
            final_list.append(source)
            seen_set.add(unique_key)
