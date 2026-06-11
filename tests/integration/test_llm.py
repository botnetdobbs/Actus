def test_llm_complete_requires_auth(client):
    resp = client.post("/v1/llm/complete", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401


def test_llm_chat_stream_requires_auth(client):
    resp = client.post("/v1/llm/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401
