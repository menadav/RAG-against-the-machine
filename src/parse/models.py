import sys
import uuid
from typing import List, Union, Optional
try:
    from pydantic import BaseModel, Field, AliasChoices
except ImportError:
    print("[ERROR] Install pydantic", file=sys.stderr)
    sys.exit(1)


class IngestConfig(BaseModel):
    """Configuration schema for the ingestion process.

    Attributes:
        max_chunk_size (int): Maximum character limit per text chunk.
        extensions (List[str]): File extensions included in the ingestion.
        path (str): The root directory path of the repository.
        last_modif (float): Timestamp of the last repository modification.
    """
    max_chunk_size: int
    extensions: List[str]
    path: str
    last_modif: float


class MinimalSource(BaseModel):
    """Represents a specific segment of a source file.

    Attributes:
        file_path (str): The path to the source file.
        first_character_index (int): Start position of the chunk in the file.
        last_character_index (int): End position of the chunk in the file.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Schema for a question that has not yet been answered by the system.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): The question text.
        difficulty (Optional[str]): Difficulty level of the question.
        is_valid (Optional[bool]): Status flag for question validity.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    difficulty: Optional[str] = None
    is_valid: Optional[bool] = True


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question that has been answered with context sources."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Container for the full dataset of RAG questions.

    Attributes:
        rag_questions (List[Union[AnsweredQuestion, UnansweredQuestion]]):
        Collection of answered and unanswered questions.
    """
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Schema for search results mapped to a specific question.

    Attributes:
        question_id (str): Identifier of the query.
        question_str (Optional[str]): The original question text.
        retrieved_sources (List[MinimalSource]):
        List of chunks retrieved for the query.
    """
    question_id: str
    question_str: Optional[str] = Field(
        None, validation_alias=AliasChoices(
            "question", "question_str"
            )
            )
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Schema for a search result that includes a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Collection of search results for student queries.

    Attributes:
        search_results (List[MinimalSearchResults]):
        List of individual search results.
        k (int): Number of sources retrieved per query.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Extended search results that include answers for all queries."""
    search_results: List[MinimalAnswer]  # type: ignore
