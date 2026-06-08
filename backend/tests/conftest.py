import sys
import os

# Backend modules use flat imports — add backend/ to path before any import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from search_tools import CourseSearchTool


@pytest.fixture
def sample_search_results():
    return SearchResults(
        documents=[
            "Lesson 1 content: RAG is retrieval augmented generation.",
            "Semantic search uses embeddings to find similar content.",
        ],
        metadata=[
            {"course_title": "RAG Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "RAG Course", "lesson_number": 1, "chunk_index": 1},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults(
        documents=[], metadata=[], distances=[],
        error="Search error: n_results exceeds collection count",
    )


@pytest.fixture
def mock_vector_store(sample_search_results):
    store = MagicMock(spec=VectorStore)
    store.search.return_value = sample_search_results
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return store


@pytest.fixture
def course_search_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def sample_course():
    return Course(
        title="RAG Course",
        course_link="https://example.com/rag-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction to RAG", lesson_link="https://example.com/lesson/1"),
            Lesson(lesson_number=2, title="Vector Databases", lesson_link="https://example.com/lesson/2"),
        ],
    )


@pytest.fixture
def sample_chunks():
    return [
        CourseChunk(
            content="Lesson 1 content: Introduction to RAG systems.",
            course_title="RAG Course",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="RAG combines retrieval with generation for better answers.",
            course_title="RAG Course",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Lesson 2 content: Vector databases store embeddings.",
            course_title="RAG Course",
            lesson_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def ephemeral_vector_store(tmp_path, sample_course, sample_chunks):
    """Real VectorStore backed by a per-test temp directory."""
    store = VectorStore(str(tmp_path / "chroma"), "all-MiniLM-L6-v2", max_results=5)
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)
    return store
