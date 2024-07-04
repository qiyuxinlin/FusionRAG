from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from server.utils.sql_utils import Base


class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, index=True)
    object = Column(String, default="thread")
    created_at = Column(Integer)

    tool_resources = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=True)

    runs = relationship("Run", back_populates="thread")
    messages = relationship("Message", back_populates="thread")
