def test_outcome_map_translates_agent_status_to_filterable_run_log_outcome():
    from app.automation.router import _OUTCOME_MAP
    # These are the AgentRunLog.outcome values used by the /v1/automation `outcome` query filter.
    assert _OUTCOME_MAP["completed"] == "success"
    assert _OUTCOME_MAP["incomplete"] == "incomplete"
    assert _OUTCOME_MAP["error"] == "error"
    assert _OUTCOME_MAP["timeout"] == "timeout"
