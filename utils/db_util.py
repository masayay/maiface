# -*- coding: utf-8 -*-
import conf
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, String, DATETIME, func
from sqlalchemy.orm import Session, sessionmaker
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
"""
Load Configuration
"""
DBUSER=conf.DBUSER
DBPASS=conf.DBPASS
DBHOST=conf.DBHOST
DBPORT=conf.DBPORT
DBNAME=conf.DBNAME
"""
Connection
"""
# Define the MariaDB engine using MariaDB Connector
url = f"mariadb+mariadbconnector://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}"
engine = create_engine(
    url,
    pool_recycle=3600,)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=True, autoflush=True, bind=engine)

# ORM create table 
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
Models
"""
class AuthHistoryModel(Base):
    __tablename__ = "t_auth_history"
    authtime = Column(DATETIME, nullable=False, server_default=func.current_timestamp(6), primary_key=True)
    faceid = Column(String, nullable=False, primary_key=True)
    deviceid = Column(String, nullable=False, server_default="noinfo")
    distance = Column(Float, nullable=False)

"""
Schemas
"""
class AuthHistoryBase(BaseModel):
    faceid: Optional[str]
    deviceid: Optional[str]
    distance: Optional[float]

class AuthHistoryCreate(AuthHistoryBase):
    pass

class AuthHistory(AuthHistoryBase):
    authtime: Optional[datetime]
    faceid: Optional[str]
    deviceid: Optional[str]
    distance: Optional[float]
    
    class Config:
        orm_mode = True

"""
CRUD
"""
def get_auth_history(db: Session, skip: int = 0, limit: int = 100):
    return db.query(AuthHistoryModel).offset(skip).limit(limit).all()


def add_auth_history(db: Session, auth_history: AuthHistoryCreate):
    db_auth_history = AuthHistoryModel(faceid=auth_history.faceid,
                                  deviceid=auth_history.deviceid,
                                  distance=auth_history.distance)
    db.add(db_auth_history)
    return db_auth_history
