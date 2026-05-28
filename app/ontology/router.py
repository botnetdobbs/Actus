from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlmodel import Session, select
from app.database import get_session
from app.ontology.registry import get_type, list_types
import structlog

log = structlog.get_logger()
router = APIRouter()


@router.get("/types")
def get_ontology_types():
    return {"types": list_types()}


@router.get("/objects/{type_name}")
def list_objects(
    type_name: str,
    session: Session = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    try:
        cls = get_type(type_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    results = session.exec(
        select(cls).where(cls.is_deleted == False).offset(offset).limit(limit)
    ).all()
    return {"type": type_name, "count": len(results), "items": results}


@router.get("/objects/{type_name}/{object_id}")
def get_object(
    type_name: str,
    object_id: int,
    session: Session = Depends(get_session),
):
    try:
        cls = get_type(type_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    return obj


@router.post("/objects/{type_name}", status_code=201)
def create_object(
    type_name: str,
    payload: dict,
    session: Session = Depends(get_session),
):
    try:
        cls = get_type(type_name)
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
    try:
        session.add(obj)
        session.commit()
        session.refresh(obj)
    except Exception as e:
        session.rollback()
        log.error("ontology_create_failed", type=type_name, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    log.info("ontology_object_created", type=type_name, id=obj.id)
    return obj


@router.put("/objects/{type_name}/{object_id}")
def update_object(
    type_name: str,
    object_id: int,
    payload: dict,
    session: Session = Depends(get_session),
):
    try:
        cls = get_type(type_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    unknown = [k for k in payload if k not in cls.model_fields]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {unknown}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
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
    return obj


@router.delete("/objects/{type_name}/{object_id}", status_code=204)
def delete_object(
    type_name: str,
    object_id: int,
    session: Session = Depends(get_session),
):
    try:
        cls = get_type(type_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    obj.soft_delete()
    session.add(obj)
    session.commit()
    log.info("ontology_object_deleted", type=type_name, id=object_id)
