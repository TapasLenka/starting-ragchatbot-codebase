import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:

    def test_basic_search_returns_formatted_string(self, course_search_tool, sample_search_results):
        result = course_search_tool.execute("what is RAG?")
        assert isinstance(result, str)
        assert "RAG Course" in result
        assert "RAG is retrieval augmented generation" in result

    def test_course_name_filter_passed_to_store(self, course_search_tool, mock_vector_store):
        course_search_tool.execute("RAG basics", course_name="RAG Course")
        mock_vector_store.search.assert_called_once_with(
            query="RAG basics",
            course_name="RAG Course",
            lesson_number=None,
        )

    def test_lesson_number_filter_passed_to_store(self, course_search_tool, mock_vector_store):
        course_search_tool.execute("lesson intro", lesson_number=1)
        mock_vector_store.search.assert_called_once_with(
            query="lesson intro",
            course_name=None,
            lesson_number=1,
        )

    def test_vector_store_error_returned_as_string_not_raised(self, mock_vector_store, error_search_results):
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        # Must NOT raise — error propagates as a plain string
        result = tool.execute("anything")
        assert "Search error" in result
        assert isinstance(result, str)

    def test_empty_results_returns_no_content_message(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("obscure topic")
        assert "No relevant content found" in result

    def test_empty_results_with_course_filter_includes_course_name(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("obscure topic", course_name="RAG Course")
        assert "RAG Course" in result

    def test_last_sources_populated_after_search(self, course_search_tool, sample_search_results):
        course_search_tool.execute("what is RAG?")
        assert len(course_search_tool.last_sources) == len(sample_search_results.documents)
        assert all("label" in s for s in course_search_tool.last_sources)
        assert all("url" in s for s in course_search_tool.last_sources)

    def test_last_sources_url_comes_from_get_lesson_link(self, course_search_tool, mock_vector_store):
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        course_search_tool.execute("what is RAG?")
        assert any(s["url"] == "https://example.com/lesson/1" for s in course_search_tool.last_sources)

    def test_last_sources_label_includes_course_and_lesson(self, course_search_tool):
        course_search_tool.execute("what is RAG?")
        source = course_search_tool.last_sources[0]
        assert "RAG Course" in source["label"]
        assert "Lesson 1" in source["label"]

    def test_last_sources_empty_before_first_search(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        assert tool.last_sources == []


class TestToolDefinition:

    def test_tool_definition_has_required_top_level_fields(self, course_search_tool):
        defn = course_search_tool.get_tool_definition()
        assert defn["name"] == "search_course_content"
        assert "description" in defn
        assert "input_schema" in defn

    def test_tool_definition_query_is_required_parameter(self, course_search_tool):
        schema = course_search_tool.get_tool_definition()["input_schema"]
        assert "query" in schema["properties"]
        assert "query" in schema.get("required", [])


class TestToolManager:

    def test_execute_dispatches_to_registered_tool(self, mock_vector_store, sample_search_results):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        result = manager.execute_tool("search_course_content", query="test")
        mock_vector_store.search.assert_called_once()
        assert isinstance(result, str)

    def test_execute_unknown_tool_returns_error_string(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result.lower()

    def test_get_last_sources_returns_populated_sources_after_search(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources_clears_all_tool_sources(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        assert manager.get_last_sources() == []


class TestCourseSearchToolIntegration:
    """Integration tests: real ChromaDB with sentence-transformers."""

    def test_search_with_fewer_docs_than_n_results_does_not_raise(self, ephemeral_vector_store):
        """chromadb raises InvalidArgumentError when n_results > docs in collection.
        VectorStore.search() must catch this and return gracefully."""
        # 3 chunks in store, max_results=5 — triggers the n_results > count path
        tool = CourseSearchTool(ephemeral_vector_store)
        result = tool.execute("introduction to RAG")
        assert isinstance(result, str), f"Expected str, got {type(result)}: {result!r}"

    def test_filtered_search_by_course_name_returns_string(self, ephemeral_vector_store):
        tool = CourseSearchTool(ephemeral_vector_store)
        result = tool.execute("introduction", course_name="RAG Course")
        assert isinstance(result, str)

    def test_filtered_search_returns_content_or_no_results_message(self, ephemeral_vector_store):
        tool = CourseSearchTool(ephemeral_vector_store)
        result = tool.execute("introduction to RAG", course_name="RAG Course")
        # Either found something or a clean "no results" message — never an exception
        assert "RAG Course" in result or "No relevant content found" in result

    def test_search_on_empty_collection_does_not_raise(self, tmp_path):
        """Querying an empty ChromaDB collection must not propagate an exception."""
        from vector_store import VectorStore
        store = VectorStore(str(tmp_path / "empty_chroma"), "all-MiniLM-L6-v2", max_results=5)
        tool = CourseSearchTool(store)
        result = tool.execute("anything")
        assert isinstance(result, str)
