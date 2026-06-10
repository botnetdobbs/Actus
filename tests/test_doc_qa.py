"""Tests for the Document Q&A module: parser, tools, upload endpoint, and trigger integration."""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import get_token, seed_user


# ── Parser unit tests ──────────────────────────────────────────────────────────

class TestChunkText:
    def setup_method(self):
        from app.agents.doc_qa.parser import _chunk_text
        self._chunk = _chunk_text

    def test_single_chunk_when_short(self):
        text = "Short text."
        assert self._chunk(text, chunk_chars=1200) == [text]

    def test_splits_at_word_boundary(self):
        words = ["word"] * 300  # 300 * 5 = 1500 chars (> 1200)
        text = " ".join(words)
        chunks = self._chunk(text, chunk_chars=1200, overlap_chars=0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_no_mid_word_splits(self):
        text = "a" * 500 + " " + "b" * 500 + " " + "c" * 500
        chunks = self._chunk(text, chunk_chars=600, overlap_chars=0)
        for chunk in chunks:
            # Each chunk should not have a split inside a run of same chars
            assert chunk.strip() == chunk

    def test_overlap(self):
        text = "alpha " * 250  # 1500 chars
        chunks = self._chunk(text, chunk_chars=600, overlap_chars=100)
        # With overlap, second chunk starts before where first ended
        assert len(chunks) >= 2

    def test_empty_returns_empty(self):
        assert self._chunk("", chunk_chars=1200) == []

    def test_whitespace_only_returns_empty(self):
        assert self._chunk("   \n\t  ", chunk_chars=1200) == []

    def test_pathological_no_whitespace(self):
        text = "a" * 2500
        chunks = self._chunk(text, chunk_chars=1200, overlap_chars=0)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 1200

    def test_respects_max_cap(self):
        from app.agents.doc_qa.parser import MAX_CHUNKS
        text = "word " * (MAX_CHUNKS * 10)
        chunks = self._chunk(text, chunk_chars=50, overlap_chars=0)
        assert len(chunks) <= MAX_CHUNKS


class TestExtractText:
    def test_file_not_found(self):
        from app.agents.doc_qa.parser import extract_text
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/path/file.pdf")

    def test_unsupported_extension(self, tmp_path):
        from app.agents.doc_qa.parser import extract_text
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(str(f))

    def test_pdf_success(self, tmp_path):
        from app.agents.doc_qa.parser import extract_text
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some extracted text from PDF."
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        with patch("pdfplumber.open", return_value=mock_pdf):
            text = extract_text(str(f))
        assert "Some extracted text from PDF." in text

    def test_docx_success(self, tmp_path):
        from app.agents.doc_qa.parser import extract_text
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake")
        para1 = MagicMock()
        para1.text = "First paragraph."
        para2 = MagicMock()
        para2.text = "Second paragraph."
        mock_doc = MagicMock()
        mock_doc.paragraphs = [para1, para2]
        with patch("docx.Document", return_value=mock_doc):
            text = extract_text(str(f))
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_empty_raises(self, tmp_path):
        from app.agents.doc_qa.parser import extract_text
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        with patch("pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(ValueError, match="No extractable text"):
                extract_text(str(f))

    def test_pdf_password_protected(self, tmp_path):
        from app.agents.doc_qa.parser import extract_text
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        with patch("pdfplumber.open", side_effect=Exception("encrypted")):
            with pytest.raises(ValueError, match="password-protected"):
                extract_text(str(f))


# ── Tool unit tests ────────────────────────────────────────────────────────────

class TestChunkAndIndexDocument:
    def _make_tools(self):
        from app.agents.doc_qa.tools import chunk_and_index_document
        return chunk_and_index_document

    def test_happy_path(self, tmp_path):
        tool_fn = self._make_tools()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        with patch("app.agents.doc_qa.tools._UPLOAD_DIR", tmp_path.resolve()), \
             patch("app.agents.doc_qa.tools.extract_text", return_value="word " * 100), \
             patch("app.agents.doc_qa.tools.index_text") as mock_index:
            result = tool_fn(str(f))
        assert "session_id" in result
        assert result["chunks_indexed"] >= 1
        assert result["char_count"] > 0
        assert mock_index.call_count >= 1

    def test_file_not_found_propagates(self):
        tool_fn = self._make_tools()
        # Non-existent path raises FileNotFoundError from resolve(strict=True)
        with pytest.raises(FileNotFoundError):
            tool_fn("/no/such/file.pdf")

    def test_scanned_pdf_propagates(self, tmp_path):
        tool_fn = self._make_tools()
        f = tmp_path / "scan.pdf"
        f.write_bytes(b"fake")
        with patch("app.agents.doc_qa.tools._UPLOAD_DIR", tmp_path.resolve()), \
             patch("app.agents.doc_qa.tools.extract_text", side_effect=ValueError("No extractable text")):
            with pytest.raises(ValueError, match="No extractable text"):
                tool_fn(str(f))

    def test_partial_failure_returns_warning(self, tmp_path):
        tool_fn = self._make_tools()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        call_count = [0]
        def flaky_index(*_):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated failure")
        with patch("app.agents.doc_qa.tools._UPLOAD_DIR", tmp_path.resolve()), \
             patch("app.agents.doc_qa.tools.extract_text", return_value="word " * 300), \
             patch("app.agents.doc_qa.tools.index_text", side_effect=flaky_index):
            result = tool_fn(str(f))
        assert result["chunks_indexed"] >= 1
        assert result["warning"] is not None

    def test_all_fail_raises(self, tmp_path):
        tool_fn = self._make_tools()
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake")
        with patch("app.agents.doc_qa.tools._UPLOAD_DIR", tmp_path.resolve()), \
             patch("app.agents.doc_qa.tools.extract_text", return_value="word " * 100), \
             patch("app.agents.doc_qa.tools.index_text", side_effect=RuntimeError("always fails")):
            with pytest.raises(RuntimeError, match="0/"):
                tool_fn(str(f))

    def test_path_traversal_blocked(self, tmp_path):
        from app.agents.doc_qa.tools import chunk_and_index_document
        # A path outside the upload dir should be rejected before any IO
        with pytest.raises((ValueError, FileNotFoundError)):
            chunk_and_index_document("/etc/passwd")

    def test_symlink_traversal_blocked(self, tmp_path):
        from app.agents.doc_qa.tools import chunk_and_index_document, _UPLOAD_DIR
        # Symlink inside upload dir pointing outside — resolve() should catch it
        evil_target = tmp_path / "secret.txt"
        evil_target.write_text("secret")
        upload_subdir = _UPLOAD_DIR
        upload_subdir.mkdir(parents=True, exist_ok=True)
        link = upload_subdir / "evil_link.pdf"
        try:
            link.symlink_to(evil_target)
            with pytest.raises(ValueError, match="upload directory"):
                chunk_and_index_document(str(link))
        finally:
            link.unlink(missing_ok=True)


class TestSearchDocument:
    def test_calls_retrieve_with_session_type(self):
        from app.agents.doc_qa.tools import search_document
        mock_results = [
            {"document": "chunk text", "metadata": {"object_id": 0}, "score": 0.9}
        ]
        with patch("app.agents.doc_qa.tools.retrieve", return_value=mock_results) as mock_r:
            result = search_document("abc-123", "what is the main finding?", top_k=3)
        mock_r.assert_called_once_with("what is the main finding?", type_name="doc:abc-123", top_k=3)
        assert result[0]["text"] == "chunk text"
        assert result[0]["chunk_index"] == 0
        assert result[0]["score"] == 0.9

    def test_formats_results_correctly(self):
        from app.agents.doc_qa.tools import search_document
        mock_results = [
            {"document": "passage A", "metadata": {"object_id": 2}, "score": 0.75},
            {"document": "passage B", "metadata": {"object_id": 5}, "score": 0.55},
        ]
        with patch("app.agents.doc_qa.tools.retrieve", return_value=mock_results):
            result = search_document("sid", "query")
        assert len(result) == 2
        assert result[1]["chunk_index"] == 5


class TestCleanupDocument:
    def test_calls_delete_by_type_with_prefix(self):
        from app.agents.doc_qa.tools import cleanup_document
        with patch("app.agents.doc_qa.tools.delete_by_type", return_value=7) as mock_del:
            result = cleanup_document("my-session-id")
        mock_del.assert_called_once_with("doc:my-session-id")
        assert result["deleted_chunks"] == 7

    def test_idempotent_returns_zero(self):
        from app.agents.doc_qa.tools import cleanup_document
        with patch("app.agents.doc_qa.tools.delete_by_type", return_value=0):
            result = cleanup_document("missing-session")
        assert result["deleted_chunks"] == 0


# ── Indexer extension tests ────────────────────────────────────────────────────

class TestIndexText:
    def test_noop_on_sqlite(self):
        from app.rag.indexer import index_text
        with patch("app.rag.indexer._is_postgres", return_value=False), \
             patch("app.rag.indexer.embed") as mock_embed:
            index_text("doc:test", 0, "some text")
        mock_embed.assert_not_called()

    def test_happy_path(self):
        from app.rag.indexer import index_text
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("app.rag.indexer._is_postgres", return_value=True), \
             patch("app.rag.indexer.embed", return_value=[0.1] * 384), \
             patch("app.rag.indexer.Session", return_value=mock_session), \
             patch("app.rag.indexer.pg_insert") as mock_insert:
            mock_stmt = MagicMock()
            mock_stmt.on_conflict_do_update.return_value = mock_stmt
            mock_insert.return_value = mock_stmt
            index_text("doc:sess", 1, "hello world")
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()


class TestDeleteByType:
    def test_noop_on_sqlite(self):
        from app.rag.indexer import delete_by_type
        with patch("app.rag.indexer._is_postgres", return_value=False):
            result = delete_by_type("doc:anything")
        assert result == 0

    def test_returns_count(self):
        from app.rag.indexer import delete_by_type
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.return_value.rowcount = 3
        with patch("app.rag.indexer._is_postgres", return_value=True), \
             patch("app.rag.indexer.Session", return_value=mock_session):
            result = delete_by_type("doc:my-session")
        assert result == 3
        mock_session.commit.assert_called_once()


# ── Upload endpoint tests ──────────────────────────────────────────────────────

class TestUploadEndpoint:
    def test_requires_auth(self, client):
        resp = client.post("/v1/doc-qa/upload", files={"file": ("doc.pdf", b"data", "application/pdf")})
        assert resp.status_code == 401

    def test_viewer_blocked(self, client, engine):
        seed_user(engine, "viewer1", "viewer")
        token = get_token(client, "viewer1")
        resp = client.post(
            "/v1/doc-qa/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("doc.pdf", b"data", "application/pdf")},
        )
        assert resp.status_code == 403

    def test_unsupported_extension(self, client, engine):
        seed_user(engine, "analyst1", "analyst")
        token = get_token(client, "analyst1")
        resp = client.post(
            "/v1/doc-qa/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 422

    def test_pdf_upload_success(self, client, engine, tmp_path):
        seed_user(engine, "analyst2", "analyst")
        token = get_token(client, "analyst2")
        with patch("app.agents.doc_qa.router.UPLOAD_DIR", tmp_path):
            resp = client.post(
                "/v1/doc-qa/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("report.pdf", b"pdfcontent", "application/pdf")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "file_path" in data
        assert data["filename"] == "report.pdf"
        assert data["size_bytes"] == len(b"pdfcontent")
        assert Path(data["file_path"]).exists()

    def test_docx_upload_success(self, client, engine, tmp_path):
        seed_user(engine, "analyst3", "analyst")
        token = get_token(client, "analyst3")
        with patch("app.agents.doc_qa.router.UPLOAD_DIR", tmp_path):
            resp = client.post(
                "/v1/doc-qa/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("brief.docx", b"docxdata", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        assert resp.status_code == 201
        assert resp.json()["filename"] == "brief.docx"

    def test_file_too_large(self, client, engine, tmp_path):
        seed_user(engine, "analyst4", "analyst")
        token = get_token(client, "analyst4")
        big = b"x" * (21 * 1024 * 1024)
        with patch("app.agents.doc_qa.router.UPLOAD_DIR", tmp_path):
            resp = client.post(
                "/v1/doc-qa/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("big.pdf", big, "application/pdf")},
            )
        assert resp.status_code == 413
        # No partial files left in upload dir
        leftover = list(tmp_path.iterdir())
        assert all(not f.is_file() for f in leftover) or len(leftover) == 0


# ── Trigger endpoint extra_context tests ──────────────────────────────────────

class TestTriggerExtraContext:
    def _make_agent_config(self):
        from app.agents.builder import AgentConfig
        return AgentConfig(id="doc_qa", name="Doc QA", tools=[], model="ollama/mistral")

    def test_extra_context_stored_in_workflow(self, client, engine):
        from sqlmodel import Session, select
        from app.context.models import Workflow
        seed_user(engine, "trig_user", "analyst")
        token = get_token(client, "trig_user")
        cfg = self._make_agent_config()
        with patch("app.automation.router.get_agent", return_value=cfg), \
             patch("app.automation.router._run_workflow"):
            resp = client.post(
                "/v1/automation/trigger/doc_qa",
                headers={"Authorization": f"Bearer {token}"},
                json={"extra_context": {"file_path": "/tmp/x.pdf", "question": "What?"}},
            )
        assert resp.status_code == 202
        wf_id = resp.json()["workflow_id"]
        with Session(engine) as session:
            wf = session.get(Workflow, wf_id)
        assert wf.extra_context_json is not None
        ctx = json.loads(wf.extra_context_json)
        assert ctx["file_path"] == "/tmp/x.pdf"

    def test_no_body_backward_compat(self, client, engine):
        from sqlmodel import Session
        from app.context.models import Workflow
        seed_user(engine, "trig_user2", "analyst")
        token = get_token(client, "trig_user2")
        cfg = self._make_agent_config()
        with patch("app.automation.router.get_agent", return_value=cfg), \
             patch("app.automation.router._run_workflow"):
            resp = client.post(
                "/v1/automation/trigger/doc_qa",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 202
        wf_id = resp.json()["workflow_id"]
        with Session(engine) as session:
            wf = session.get(Workflow, wf_id)
        assert wf.extra_context_json is None
