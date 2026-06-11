"""Tests for the Document Q&A module: upload endpoint and trigger integration."""
import json
from pathlib import Path
from unittest.mock import patch

from tests.conftest import get_token, seed_user


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
        assert all(not f.is_file() for f in leftover)


# ── Trigger endpoint extra_context tests ──────────────────────────────────────

class TestTriggerExtraContext:
    def _make_agent_config(self):
        from app.agents.builder import AgentConfig
        return AgentConfig(id="doc_qa", name="Doc QA", tools=[], model="ollama/mistral")

    def test_extra_context_stored_in_workflow(self, client, engine):
        from sqlmodel import Session
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
