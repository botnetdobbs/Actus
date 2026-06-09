from datetime import datetime, timedelta, timezone
from typing import Annotated, NoReturn
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select
from app.database import get_session
from app.auth.models import Team, User, hash_password, verify_password
from app.auth.jwt import create_access_token, get_current_user, require_role, write_audit_log
import structlog

log = structlog.get_logger()

router = APIRouter()

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    password: str


class RoleUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str


class PasswordResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int | None
    username: str
    role: str
    is_active: bool
    team_id: int | None = None
    created_at: datetime


class TeamCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class TeamAssignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: int | None


class TeamResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int | None
    name: str
    created_at: datetime
    created_by: int | None


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: Request,
    req: RegisterRequest,
    session: Session = Depends(get_session),
):
    existing = session.exec(
        select(User).where(User.username == req.username, User.is_deleted == False)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(username=req.username, hashed_password=hash_password(req.password), role="viewer")
    try:
        session.add(user)
        session.commit()
        session.refresh(user)
    except Exception as e:
        session.rollback()
        log.error("register_failed", username=req.username, error=str(e))
        raise

    ip = request.client.host if request.client else None
    write_audit_log(username=req.username, action="register", ip=ip, success=True)
    log.info("user_registered", username=req.username, role="viewer")
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    ip = request.client.host if request.client else None

    user = session.exec(
        select(User).where(User.username == form.username, User.is_deleted == False)
    ).first()

    def fail(detail: str) -> NoReturn:
        if user:
            user.failed_login_count += 1
            if user.failed_login_count >= MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
                log.warning("account_locked", username=form.username, until=user.locked_until.isoformat())
            user.touch()
            session.add(user)
            session.commit()
        write_audit_log(username=form.username, action="login", ip=ip, success=False, detail=detail)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user or not user.is_active:
        fail("user not found or inactive")

    if user and user.is_locked():
        write_audit_log(username=form.username, action="login", ip=ip, success=False, detail="account locked")
        raise HTTPException(status_code=403, detail="Account temporarily locked")

    if not verify_password(form.password, user.hashed_password):
        fail("wrong password")

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)
    user.touch()
    session.add(user)
    session.commit()

    token = create_access_token({"sub": user.username})
    write_audit_log(username=user.username, action="login", ip=ip, success=True)
    log.info("user_logged_in", username=user.username)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/users/{user_id}/role", response_model=UserResponse)
async def assign_role(
    user_id: int,
    req: RoleUpdateRequest,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    valid_roles = {"viewer", "analyst", "admin"}
    if req.role not in valid_roles:
        raise HTTPException(status_code=422, detail=f"Role must be one of: {', '.join(sorted(valid_roles))}")

    target = session.exec(
        select(User).where(User.id == user_id, User.is_deleted == False)
    ).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.role = req.role
    target.touch()
    session.add(target)
    session.commit()
    session.refresh(target)

    log.info("role_assigned", by=admin.username, target=target.username, role=req.role)
    write_audit_log(username=admin.username, action="assign_role", resource=target.username, success=True, detail=req.role)
    return target


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("admin")),
):
    return session.exec(
        select(User).where(User.is_deleted == False).offset(offset).limit(limit)
    ).all()


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    target = session.exec(
        select(User).where(User.id == user_id, User.is_deleted == False)
    ).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    target.soft_delete(deleted_by=admin.id)
    session.add(target)
    session.commit()

    log.info("user_deleted", by=admin.username, target=target.username)
    write_audit_log(username=admin.username, action="delete_user", resource=target.username, success=True)


@router.patch("/users/{user_id}/password", response_model=UserResponse)
async def reset_password(
    user_id: int,
    req: PasswordResetRequest,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    target = session.exec(
        select(User).where(User.id == user_id, User.is_deleted == False)
    ).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.hashed_password = hash_password(req.new_password)
    target.touch()
    session.add(target)
    session.commit()
    session.refresh(target)

    log.info("password_reset", by=admin.username, target=target.username)
    write_audit_log(username=admin.username, action="reset_password", resource=target.username, success=True)
    return target


# ── Team management ───────────────────────────────────────────────────────────

@router.post("/teams", response_model=TeamResponse, status_code=201)
async def create_team(
    req: TeamCreateRequest,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    existing = session.exec(
        select(Team).where(Team.name == req.name, Team.is_deleted == False)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Team name already taken")
    team = Team(name=req.name, created_by=admin.id)
    session.add(team)
    session.commit()
    session.refresh(team)
    log.info("team_created", by=admin.username, team=req.name)
    write_audit_log(username=admin.username, action="create_team", resource=req.name, success=True)
    return team


@router.get("/teams", response_model=list[TeamResponse])
async def list_teams(
    session: Session = Depends(get_session),
    _: User = Depends(require_role("admin")),
):
    return session.exec(select(Team).where(Team.is_deleted == False)).all()


@router.patch("/users/{user_id}/team", response_model=UserResponse)
async def assign_team(
    user_id: int,
    req: TeamAssignRequest,
    session: Session = Depends(get_session),
    admin: User = Depends(require_role("admin")),
):
    target = session.exec(
        select(User).where(User.id == user_id, User.is_deleted == False)
    ).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if req.team_id is not None:
        team = session.exec(
            select(Team).where(Team.id == req.team_id, Team.is_deleted == False)
        ).first()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

    target.team_id = req.team_id
    target.touch()
    session.add(target)
    session.commit()
    session.refresh(target)

    log.info("team_assigned", by=admin.username, target=target.username, team_id=req.team_id)
    write_audit_log(username=admin.username, action="assign_team", resource=target.username, success=True,
                    detail=str(req.team_id))
    return target
