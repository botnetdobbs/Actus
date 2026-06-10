from datetime import datetime, timedelta, timezone
import uuid as _uuid
import jwt
from jwt import PyJWTError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
from app.config import get_settings
from app.database import get_engine, get_session
from app.auth.models import User, AuditLog, VALID_ROLES
import structlog

log = structlog.get_logger()

_settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=_settings.access_token_expire_minutes))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "jti": str(_uuid.uuid4())})
    return jwt.encode(to_encode, _settings.secret_key, algorithm=_settings.algorithm)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=_settings.refresh_token_expire_minutes)
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(_uuid.uuid4()),
        "type": "refresh",
    })
    return jwt.encode(to_encode, _settings.secret_key, algorithm=_settings.algorithm)


async def _revoke_token(jti: str, expires_in: int) -> None:
    from app import pubsub
    if pubsub._redis is None:
        return
    try:
        await pubsub._redis.setex(f"jti:revoked:{jti}", expires_in, "1")
    except Exception as e:
        log.warning("token_revocation_failed", jti=jti, error=str(e))


async def _is_revoked(jti: str) -> bool:
    from app import pubsub
    if pubsub._redis is None:
        return False
    try:
        return bool(await pubsub._redis.exists(f"jti:revoked:{jti}"))
    except Exception:
        return False


async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _settings.secret_key, algorithms=[_settings.algorithm])
        username: str | None = payload.get("sub")
        if username is None:
            raise exc
        jti: str | None = payload.get("jti")
        if jti and await _is_revoked(jti):
            raise exc
        token_version = payload.get("tv", 0)
    except PyJWTError:
        raise exc
    user = session.exec(select(User).where(User.username == username, User.is_deleted == False)).first()
    if not user or not user.is_active:
        raise exc
    if token_version != user.token_version:
        raise exc
    if user.is_locked():
        raise HTTPException(status_code=403, detail="Account temporarily locked")
    request.state.user_id = user.id
    return user


_ROLES_HIERARCHY: dict[str, int] = {"viewer": 0, "analyst": 1, "admin": 2}


def require_role(role: str):
    async def checker(user: User = Depends(get_current_user)):
        required = _ROLES_HIERARCHY.get(role, 99)
        actual = _ROLES_HIERARCHY.get(user.role, 0)
        if actual < required:
            raise HTTPException(status_code=403, detail=f"Role '{role}' required")
        return user
    return checker


def write_audit_log(
    username: str,
    action: str,
    resource: str | None = None,
    ip: str | None = None,
    success: bool = True,
    detail: str | None = None,
) -> None:
    entry = AuditLog(
        username=username,
        action=action,
        resource=resource,
        ip_address=ip,
        success=success,
        detail=detail,
    )
    with Session(get_engine()) as session:
        try:
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("audit_log_write_failed", username=username, action=action, error=str(e))
