import sys
import uuid
from typing import List, Union, Optional
try:
    from pydantic import BaseModel, Field
except ImportError:
    print("[ERROR] Install pydantic", file=sys.stderr)
    sys.exit(1)


class IngestConfig(BaseModel):
    max_chunk_size: int
    extensions: List[str]
    path: str
    last_modif: float


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    difficulty: Optional[str] = None
    is_valid: Optional[bool] = True


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: Optional[str] = Field(None, alias="question")
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
