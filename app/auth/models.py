from datetime import datetime, timezone
from sqlalchemy import DateTime as SADateTime, Column
from sqlmodel import SQLModel, Field
import bcrypt as _bcrypt


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    role: str = "viewer"
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = Field(
        default=None, sa_column=Column(SADateTime(timezone=True))
    )
    failed_login_count: int = 0
    locked_until: datetime | None = Field(
        default=None, sa_column=Column(SADateTime(timezone=True))
    )
    is_deleted: bool = Field(default=False, index=True)
    deleted_at: datetime | None = None
    deleted_by: int | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def is_locked(self) -> bool:
        if not self.locked_until:
            return False
        # Guard for SQLite in tests: it ignores timezone=True and returns naive datetimes
        locked_until = (
            self.locked_until.replace(tzinfo=timezone.utc)
            if self.locked_until.tzinfo is None
            else self.locked_until
        )
        return locked_until > datetime.now(timezone.utc)

    def soft_delete(self, deleted_by: int | None = None) -> None:
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = deleted_by


class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    action: str = Field(index=True)
    resource: str | None = None
    ip_address: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    detail: str | None = None


def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode(), hashed.encode())
