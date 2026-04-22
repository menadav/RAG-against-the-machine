import os
import bm25s
import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from src.parse.models import MinimalSource
from tqdm import tqdm
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("bm25s").setLevel(logging.ERROR)


class Retrieval:
    def __init__(
            self,
            index_dir="data/processed/bm25_index",
            metadata_path="data/processed/chunks/chunk.json",
            chroma_path="data/processed/chroma_db"
            ) -> None:
        self.index_dir = Path(index_dir)
        self.metadata_path = Path(metadata_path)
        self.retrieval = None
        self.metadata = []
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="vllm_docs",
            embedding_function=self.emb_fn
        )

    def build_and_save(self, all_chunks_text, all_sources):
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
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metas,
                ids=batch_ids
            )

    def load(self):
        if not self.index_dir.exists():
            raise ValueError("[ERROR] No index Found")
        self.retrieval = bm25s.BM25.load(self.index_dir, load_corpus=True)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def find_top_k(self, query_tokens: list, k: int):
        if self.retrieval is None:
            self.load()
        query_clean = "".join(
            char if char.isalnum() or char.isspace() or char in "_." else " "
            for char in query_tokens.lower()
        )
        query_list = query_clean.split()
        bm25_indices, scores = self.retrieval.retrieve([query_list], k=k)
        chroma_results = self.collection.query(
            query_texts=[query_tokens], n_results=k)
        top_k_indices = bm25_indices.flatten()
        bm25_scores = scores.flatten()
        final_sources = []
        seen_paths = set()
        for i, idx in enumerate(top_k_indices):
            if bm25_scores[i] > 1:
                meta = self.metadata[idx]
                self._add_to_final(final_sources, seen_paths, meta)
        for meta in chroma_results['metadatas'][0]:
            if len(final_sources) >= k:
                break
            self._add_to_final(final_sources, seen_paths, meta)
        return final_sources

    def _add_to_final(self, final_list, seen_set, meta):
        unique_key = f"{meta['file_path']}_{meta['first_character_index']}"
        if unique_key not in seen_set:
            raw_path = meta['file_path']
            if "vllm-0.10.1/" in raw_path:
                clean_path = raw_path.split("vllm-0.10.1/")[-1]
            else:
                clean_path = os.path.basename(raw_path)
            source = MinimalSource(
                file_path=clean_path,
                content=meta.get("content", ""),
                first_character_index=meta["first_character_index"],
                last_character_index=meta["last_character_index"]
            )
            final_list.append(source)
            seen_set.add(unique_key)
