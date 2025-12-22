CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE documents (
    doc_id uuid PRIMARY KEY,
    title text,
    content text,
    embedding vector(768), -- Dimensión de embedding a usar
    date timestamptz,
    source text
);