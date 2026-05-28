from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select
from app.database import get_session
from app.auth.models import User, hash_password, verify_password
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int | None
    username: str
    role: str
    is_active: bool
    created_at: datetime


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

    def fail(detail: str) -> None:
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
