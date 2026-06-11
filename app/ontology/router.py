from typing import cast
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlmodel import Session, col, select
from app.database import get_session
from app.ontology.models import OntologyObjectBase
from app.ontology.registry import get_type, list_types
from app.auth.models import User
from app.auth.jwt import get_current_user, require_role, write_audit_log
from app.auth.visibility import apply_visibility, check_visibility
from app.rag.indexer import delete_from_index, index_object
import structlog

log = structlog.get_logger()
router = APIRouter()

_PROTECTED_FIELDS = {"id", "created_at", "created_by", "is_deleted", "deleted_at", "deleted_by"}


@router.get("/types")
def get_ontology_types():
    return {"types": list_types()}


@router.get("/objects/{type_name}")
def list_objects(
    type_name: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    query = select(cls).where(col(cls.is_deleted).is_(False))
    query = apply_visibility(query, user, team_col=cls.team_id, owner_col=cls.created_by)
    results = session.exec(query.offset(offset).limit(limit)).all()
    return {"type": type_name, "count": len(results), "items": results}


@router.get("/objects/{type_name}/{object_id}")
def get_object(
    type_name: str,
    object_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    if not check_visibility(obj, user, team_id_attr="team_id", owner_id_attr="created_by"):
        raise HTTPException(status_code=404, detail="Object not found")
    return obj


@router.post("/objects/{type_name}", status_code=201)
def create_object(
    type_name: str,
    payload: dict,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(require_role("analyst")),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    unknown = set(payload.keys()) - set(cls.model_fields.keys())
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown fields: {unknown}. Valid: {list(cls.model_fields.keys())}",
        )
    try:
        obj = cls(**payload)
    except (ValidationError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    obj.created_by = user.id
    obj.team_id = user.team_id
    try:
        session.add(obj)
        session.commit()
        session.refresh(obj)
    except Exception as e:
        session.rollback()
        log.error("ontology_create_failed", type=type_name, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    background_tasks.add_task(index_object, type_name, obj.id, obj)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_create", resource=f"{type_name}:{obj.id}", ip=ip)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    log.info("ontology_object_created", type=type_name, id=obj.id, by=user.username)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    return obj


@router.put("/objects/{type_name}/{object_id}")
def update_object(
    type_name: str,
    object_id: int,
    payload: dict,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(require_role("analyst")),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    unknown = [k for k in payload if k not in cls.model_fields]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {unknown}")
    protected = [k for k in payload if k in _PROTECTED_FIELDS]
    if protected:
        raise HTTPException(status_code=422, detail=f"Cannot modify protected fields: {protected}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    if not check_visibility(obj, user, team_id_attr="team_id", owner_id_attr="created_by"):
        raise HTTPException(status_code=404, detail="Object not found")
    try:
        for key, value in payload.items():
            setattr(obj, key, value)
        obj.touch()
        session.add(obj)
        session.commit()
        session.refresh(obj)
    except Exception as e:
        session.rollback()
        log.error("ontology_update_failed", type=type_name, id=object_id, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    background_tasks.add_task(index_object, type_name, obj.id, obj)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_update", resource=f"{type_name}:{object_id}", ip=ip)
    log.info("ontology_object_updated", type=type_name, id=object_id, by=user.username)
    return obj


@router.delete("/objects/{type_name}/{object_id}", status_code=204)
def delete_object(
    type_name: str,
    object_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(require_role("analyst")),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    if not check_visibility(obj, user, team_id_attr="team_id", owner_id_attr="created_by"):
        raise HTTPException(status_code=404, detail="Object not found")
    obj.soft_delete(deleted_by=user.id)
    session.add(obj)
    session.commit()
    background_tasks.add_task(delete_from_index, type_name, object_id)
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_delete", resource=f"{type_name}:{object_id}", ip=ip)
    log.info("ontology_object_deleted", type=type_name, id=object_id, by=user.username)
