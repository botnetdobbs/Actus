from app.auth.models import User


def apply_visibility(query, user: User, *, team_col, owner_col):
    """Scope a query to records the user is allowed to see.

    - Users in a team: see own team's records + team-less global records (NULL team)
    - Users without a team (non-admin): see own records + unowned records
    - Global admin (no team): see everything
    """
    if user.team_id is not None:
        return query.where((team_col == user.team_id) | (team_col == None))  # noqa: E711
    if user.role != "admin":
        return query.where((owner_col == user.id) | (owner_col == None))  # noqa: E711
    return query  # global admin: see all


def check_visibility(record, user: User, *, team_id_attr: str, owner_id_attr: str) -> bool:
    """Single source of truth for by-ID access control. Mirrors apply_visibility logic."""
    record_team = getattr(record, team_id_attr, None)
    record_owner = getattr(record, owner_id_attr, None)
    if user.team_id is not None:
        return record_team == user.team_id or record_team is None
    if user.role == "admin":
        return True
    return record_owner == user.id or record_owner is None
