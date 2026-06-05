import pytest
from unittest.mock import MagicMock, patch


# ── embedder ──────────────────────────────────────────────────────────────────

def test_embed_raises_before_warmup():
    from app.rag import embedder
    original = embedder._model
    embedder._model = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.embed("hello")
    finally:
        embedder._model = original


def test_embed_returns_list_of_floats():
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    with patch("app.rag.embedder._model", mock_model):
        from app.rag.embedder import embed
        result = embed("test query")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ── indexer: text representation ──────────────────────────────────────────────

def test_object_to_text_excludes_metadata():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "created_at": "2026-01-01",
                "is_deleted": False,
                "created_by": 42,
            }

    text = _object_to_text("Customer", FakeObj())
    assert "Customer" in text
    assert "Alice" in text
    assert "alice@example.com" in text
    assert "created_at" not in text
    assert "is_deleted" not in text
    assert "created_by" not in text


def test_object_to_text_skips_booleans():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {"name": "Bob", "is_active": True, "verified": False}

    text = _object_to_text("Customer", FakeObj())
    assert "Bob" in text
    assert "is_active" not in text
    assert "verified" not in text


def test_object_to_text_skips_foreign_keys():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {"title": "Report A", "customer_id": 42, "machine_id": 7}

    text = _object_to_text("Report", FakeObj())
    assert "Report A" in text
    assert "customer_id" not in text
    assert "machine_id" not in text


def test_object_to_text_uses_natural_format():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {"name": "Alice", "segment": "enterprise"}

    text = _object_to_text("Customer", FakeObj())
    assert "name: Alice" in text
    assert "; " in text  # uses "; " separator


def test_object_to_text_per_type_override():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {"name": "Alice"}

        def rag_document(self):
            return "Custom document text"

    text = _object_to_text("Customer", FakeObj())
    assert text == "Custom document text"


# ── indexer: storage ──────────────────────────────────────────────────────────

def test_index_object_executes_upsert():
    class FakeObj:
        def model_dump(self):
            return {"name": "Bob", "segment": "starter"}

    with patch("app.rag.indexer._is_postgres", return_value=True), \
         patch("app.rag.indexer.embed", return_value=[0.1, 0.2, 0.3]), \
         patch("app.rag.indexer.get_engine"), \
         patch("app.rag.indexer.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        from app.rag.indexer import index_object
        index_object("Customer", 1, FakeObj())

    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


def test_index_object_swallows_errors():
    with patch("app.rag.indexer.embed", side_effect=Exception("model error")):
        from app.rag.indexer import index_object

        class FakeObj:
            def model_dump(self):
                return {"name": "Test"}

        index_object("Customer", 1, FakeObj())  # must not raise


def test_delete_from_index_removes_row():
    mock_row = MagicMock()
    with patch("app.rag.indexer._is_postgres", return_value=True), \
         patch("app.rag.indexer.get_engine"), \
         patch("app.rag.indexer.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = mock_row
        from app.rag.indexer import delete_from_index
        delete_from_index("Customer", 42)

    mock_session.delete.assert_called_once_with(mock_row)
    mock_session.commit.assert_called_once()


def test_delete_from_index_missing_row_is_safe():
    with patch("app.rag.indexer._is_postgres", return_value=True), \
         patch("app.rag.indexer.get_engine"), \
         patch("app.rag.indexer.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.first.return_value = None
        from app.rag.indexer import delete_from_index
        delete_from_index("Customer", 999)  # must not raise

    mock_session.delete.assert_not_called()


# ── retriever ─────────────────────────────────────────────────────────────────

def _make_row(object_type, object_id, document):
    row = MagicMock()
    row.object_type = object_type
    row.object_id = object_id
    row.document = document
    return row


def test_retrieve_merges_semantic_and_fts():
    sem_row = _make_row("Customer", 1, "Customer name: Alice")
    fts_row = _make_row("Customer", 2, "Customer name: Bob")

    with patch("app.rag.retriever.embed", return_value=[0.1, 0.2]), \
         patch("app.rag.retriever.get_engine"), \
         patch("app.rag.retriever.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        # First exec call = semantic, second = FTS
        mock_session.exec.side_effect = [
            MagicMock(**{"all.return_value": [sem_row]}),
            MagicMock(**{"all.return_value": [fts_row]}),
        ]
        from app.rag.retriever import retrieve
        results = retrieve("find alice")

    assert len(results) == 2
    assert all("score" in r for r in results)
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_empty_index_returns_empty():
    with patch("app.rag.retriever.embed", return_value=[0.1, 0.2]), \
         patch("app.rag.retriever.get_engine"), \
         patch("app.rag.retriever.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.exec.return_value.all.return_value = []
        from app.rag.retriever import retrieve
        results = retrieve("anything")

    assert results == []


def test_retrieve_empty_query_returns_empty():
    from app.rag.retriever import retrieve
    with patch("app.rag.retriever.embed") as mock_embed:
        results = retrieve("   ")
    mock_embed.assert_not_called()
    assert results == []


def test_retrieve_rrf_boosts_shared_results():
    # A row appearing in both semantic and FTS gets a higher RRF score
    shared = _make_row("Customer", 1, "Customer name: Alice")
    semantic_only = _make_row("Customer", 2, "Customer name: Bob")

    with patch("app.rag.retriever.embed", return_value=[0.1]), \
         patch("app.rag.retriever.get_engine"), \
         patch("app.rag.retriever.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.exec.side_effect = [
            MagicMock(**{"all.return_value": [shared, semantic_only]}),
            MagicMock(**{"all.return_value": [shared]}),   # shared in FTS too
        ]
        from app.rag.retriever import retrieve
        results = retrieve("alice")

    assert results[0]["metadata"]["object_id"] == 1  # shared ranks first


# ── semantic_search tool ──────────────────────────────────────────────────────

def test_semantic_search_tool_registered():
    from app.agents.tools import _tool_schemas
    assert "semantic_search" in _tool_schemas
    schema = _tool_schemas["semantic_search"]
    assert "query" in schema["parameters"]
    assert "type_name" in schema["parameters"]
    assert "top_k" in schema["parameters"]


def test_semantic_search_tool_calls_retrieve():
    with patch("app.rag.retriever.retrieve", return_value=[]) as mock_retrieve:
        from app.agents.tools import _tools
        result = _tools["semantic_search"](query="test", type_name="", top_k=3)
    mock_retrieve.assert_called_once_with("test", type_name=None, top_k=3)
    assert isinstance(result, list)
