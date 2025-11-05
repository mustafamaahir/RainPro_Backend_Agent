# app/routers/auth.py
# Simple signup & login (no password hashing as requested).

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models, schemas

router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/signup", status_code=201)
def signup(payload: schemas.SignupIn, db: Session = Depends(get_db)):
    """
    Register a new user (simple, no hashing).
    Returns stored user id and username.
    """
    if db.query(models.User).filter(models.User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = models.User(username=payload.username, password=payload.password, email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "email": user.email, "username": user.username, "created_at": user.created_at.isoformat()}


@router.post("/login")
def login(payload: schemas.LoginIn, db: Session = Depends(get_db)):
    """
    Simple login returning user id (frontend must store user_id).
    """
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or user.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"id": user.id, "username": user.username, "email": user.email}
