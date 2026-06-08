import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


@pytest.fixture
def mock_config(tmp_path):
    return SimpleNamespace(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="claude-test",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH=str(tmp_path / "chroma"),
    )


@pytest.fixture
def rag_with_mock_ai(mock_config):
    """RAGSystem with real (temp) ChromaDB and a mocked AIGenerator."""
    with patch("rag_system.AIGenerator") as MockAI:
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "Mocked answer."
        MockAI.return_value = mock_ai

        from rag_system import RAGSystem
        system = RAGSystem(mock_config)

    system._mock_ai = mock_ai
    return system


# ── Setup tests ───────────────────────────────────────────────────────────────

class TestRAGSystemSetup:

    def test_search_course_content_tool_registered(self, rag_with_mock_ai):
        assert "search_course_content" in rag_with_mock_ai.tool_manager.tools

    def test_get_course_outline_tool_registered(self, rag_with_mock_ai):
        assert "get_course_outline" in rag_with_mock_ai.tool_manager.tools

    def test_ai_generator_receives_all_tool_definitions(self, rag_with_mock_ai):
        rag_with_mock_ai.query("test", session_id="s1")
        call_kwargs = rag_with_mock_ai._mock_ai.generate_response.call_args[1]
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


# ── Query behaviour tests ─────────────────────────────────────────────────────

class TestRAGSystemQuery:

    def test_query_returns_answer_string_and_sources_list(self, rag_with_mock_ai):
        answer, sources = rag_with_mock_ai.query("What is RAG?", session_id="s1")
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_query_returns_ai_generator_response(self, rag_with_mock_ai):
        rag_with_mock_ai._mock_ai.generate_response.return_value = "specific answer"
        answer, _ = rag_with_mock_ai.query("anything", session_id="s1")
        assert answer == "specific answer"

    def test_sources_reset_between_queries(self, rag_with_mock_ai):
        """Sources from the first query must not leak into the second query."""
        rag_with_mock_ai.tool_manager.tools["search_course_content"].last_sources = [
            {"label": "RAG Course - Lesson 1", "url": "https://example.com/lesson/1"}
        ]
        _, sources_first = rag_with_mock_ai.query("first query", session_id="s1")
        _, sources_second = rag_with_mock_ai.query("second query", session_id="s1")
        assert sources_second == []

    def test_session_history_updated_after_query(self, rag_with_mock_ai):
        rag_with_mock_ai.query("what is machine learning?", session_id="test-session")
        history = rag_with_mock_ai.session_manager.get_conversation_history("test-session")
        assert history is not None
        assert "machine learning" in history

    def test_exception_from_ai_generator_propagates(self, rag_with_mock_ai):
        """When AIGenerator raises, the exception must bubble up (causes HTTP 500).
        This test confirms the propagation path that produces 'Query failed'."""
        rag_with_mock_ai._mock_ai.generate_response.side_effect = RuntimeError("Anthropic API error")
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            rag_with_mock_ai.query("What is RAG?", session_id="s1")

    def test_query_passes_tool_manager_to_ai_generator(self, rag_with_mock_ai):
        rag_with_mock_ai.query("test", session_id="s1")
        call_kwargs = rag_with_mock_ai._mock_ai.generate_response.call_args[1]
        assert call_kwargs["tool_manager"] is rag_with_mock_ai.tool_manager


# ── Integration: real ChromaDB + mocked AI ────────────────────────────────────

class TestRAGSystemIntegration:

    def test_end_to_end_content_query_returns_answer(
        self, rag_with_mock_ai, sample_course, sample_chunks
    ):
        rag_with_mock_ai.vector_store.add_course_metadata(sample_course)
        rag_with_mock_ai.vector_store.add_course_content(sample_chunks)

        rag_with_mock_ai._mock_ai.generate_response.return_value = "RAG is retrieval augmented generation."
        answer, sources = rag_with_mock_ai.query("What is RAG?", session_id="int-test")
        assert answer == "RAG is retrieval augmented generation."

    def test_search_tool_executes_real_chromadb_search_without_crash(
        self, rag_with_mock_ai, sample_course, sample_chunks
    ):
        """Directly invoke the search tool against real ChromaDB — the most
        targeted integration test for the n_results > count crash."""
        rag_with_mock_ai.vector_store.add_course_metadata(sample_course)
        rag_with_mock_ai.vector_store.add_course_content(sample_chunks)

        result = rag_with_mock_ai.tool_manager.execute_tool(
            "search_course_content", query="introduction to RAG"
        )
        assert isinstance(result, str), f"Expected str, got {type(result)}: {result!r}"

    def test_search_tool_on_empty_chromadb_does_not_raise(self, rag_with_mock_ai):
        """Query with no indexed documents must return a string, not raise."""
        result = rag_with_mock_ai.tool_manager.execute_tool(
            "search_course_content", query="anything"
        )
        assert isinstance(result, str)
