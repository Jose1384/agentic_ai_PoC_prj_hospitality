import uuid

from sqlalchemy import Column, Text, TIMESTAMP
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

from sqlalchemy.orm import declarative_base

# Base model that will be used in the models you declare
Base = declarative_base()
class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True)   # VARCHAR en SQL
    content = Column(Text, nullable=True)      # TEXT en SQL
    embedding = Column(Vector(768))        # VECTOR en pgvector (ajusta 768 a tu modelo real)
    source = Column(Text, nullable=True)  # VARCHAR en SQL
    date = Column(TIMESTAMP(timezone=True), nullable=True)