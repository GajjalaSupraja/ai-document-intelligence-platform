
import json
from uuid import uuid4
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import Response, StreamingResponse

from app.config import DATA_DIR, settings
from app.document_store import DocumentStore
from app.pdf_processor import extract_document
from app.text_chunker import chunk_text
from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService
from app.answer_service import AnswerService
from app.media_processor import extract_media_text

app = FastAPI(
    title=settings.app_name,
    description="Upload documents and search their content using AI embeddings",
    version="0.3.0",
)

document_store = DocumentStore(DATA_DIR / "documents.sqlite3")
documents: dict[str, dict] = {}
embedding_service = None
answer_service = AnswerService(model_name=settings.ollama_model)


def get_embedding_service():
    global embedding_service

    if embedding_service is None:
        embedding_service = EmbeddingService()

    return embedding_service


def get_document(document_id: str) -> dict | None:
    cached_document = documents.get(document_id)
    if cached_document is not None:
        return cached_document

    saved_document = document_store.get(document_id)
    if saved_document is None:
        return None

    retrieval_service = RetrievalService(get_embedding_service())
    retrieval_service.index_chunks(saved_document["chunks"])
    cached_document = {
        "filename": saved_document["filename"],
        "retrieval_service": retrieval_service,
        "chunk_count": saved_document["chunk_count"],
        "chunks": saved_document["chunks"],
    }
    documents[document_id] = cached_document
    return cached_document


class QuestionRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    output_language: str = Field(default="English", min_length=1, max_length=64)


class SummaryRequest(BaseModel):
    document_id: str
    output_language: str = Field(default="English", min_length=1, max_length=64)


@app.get("/")
def home():
    return {
        "message": settings.app_name,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if "." not in filename:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or DOCX file.",
        )

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported files are PDF, DOCX, PNG, JPG, JPEG, WEBP, "
                "MP4, MOV, MKV, and WEBM."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    video_extensions = {"mp4", "mov", "mkv", "webm"}
    max_size_mb = (
        settings.max_video_size_mb
        if extension in video_extensions
        else settings.max_file_size_mb
    )
    max_size = max_size_mb * 1024 * 1024

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {max_size_mb} MB limit.",
        )

    try:
        if extension in {"pdf", "docx"}:
            extracted = extract_document(filename, content)
        else:
            extracted = extract_media_text(filename, content)

        sections = extracted["extracted_content"]
        full_text = "\n\n".join(
            section["text"]
            for section in sections
            if section.get("text", "").strip()
        )

        chunks_text = chunk_text(
            full_text,
            chunk_size=500,
            overlap=100,
        )

        if not chunks_text:
            raise ValueError("No text available to index.")

        chunks = [
            {
                "text": text,
                "source": filename,
                "chunk_index": index,
            }
            for index, text in enumerate(chunks_text)
        ]

        service = RetrievalService(get_embedding_service())
        indexed_count = service.index_chunks(chunks)

        document_id = str(uuid4())

        document_store.save(
            document_id=document_id,
            filename=filename,
            file_type=extension,
            chunks=chunks,
            original_file=content,
        )
        documents[document_id] = {
            "filename": filename,
            "retrieval_service": service,
            "chunk_count": indexed_count,
            "chunks": chunks,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        print(f"Media processing could not start: {error}")
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Document processing or indexing failed: {error}")
        raise HTTPException(
            status_code=500,
            detail="Document processing or indexing failed. Check the server logs.",
        ) from error

    return {
        "document_id": document_id,
        "filename": filename,
        "status": "success",
        "file_type": extension,
        "chunks_indexed": indexed_count,
        "message": "Document processed, chunked, and indexed successfully.",
    }


@app.get("/documents")
def list_documents():
    return {"documents": document_store.list_documents()}


@app.get("/documents/{document_id}/file")
def download_document(document_id: str):
    stored_file = document_store.get_original_file(document_id)
    if stored_file is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )
    if stored_file["content"] is None:
        raise HTTPException(
            status_code=404,
            detail="The original file is not available for this document.",
        )

    media_types = {
        "pdf": "application/pdf",
        "docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    }
    encoded_filename = quote(stored_file["filename"])
    return Response(
        content=stored_file["content"],
        media_type=media_types.get(stored_file["file_type"], "application/octet-stream"),
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{encoded_filename}"
            )
        },
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not document_store.delete(document_id):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    documents.pop(document_id, None)
    return {"document_id": document_id, "status": "deleted"}


@app.get("/documents/{document_id}/history")
def get_document_history(document_id: str):
    if document_store.get(document_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "document_id": document_id,
        "history": document_store.list_conversations(document_id),
    }


@app.delete("/documents/{document_id}/history")
def clear_document_history(document_id: str):
    if not document_store.clear_conversations(document_id):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {"document_id": document_id, "status": "history_cleared"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    document = get_document(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload it again before asking questions.",
        )

    try:
        results = document["retrieval_service"].search(
            request.question,
            top_k=request.top_k,
        )

        answer = answer_service.generate_answer(
            question=request.question,
            retrieved_chunks=results,
            output_language=request.output_language,
        )

        document_store.save_conversation(
            document_id=request.document_id,
            question=request.question,
            answer=answer,
            sources=results,
        )

        return {
            "document_id": request.document_id,
            "filename": document["filename"],
            "question": request.question,
            "output_language": request.output_language,
            "answer": answer,
            "sources": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        print(f"AI answer generation failed: {error}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not generate an answer. "
                "Check that Ollama is running and the model is available."
            ),
        ) from error


@app.post("/ask/stream")
def ask_question_stream(request: QuestionRequest):
    document = get_document(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload it again before asking questions.",
        )

    try:
        results = document["retrieval_service"].search(
            request.question,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    def generate_events():
        yield json.dumps({"type": "sources", "sources": results}) + "\n"
        answer_parts = []
        try:
            for content in answer_service.stream_answer(
                question=request.question,
                retrieved_chunks=results,
                output_language=request.output_language,
            ):
                answer_parts.append(content)
                yield json.dumps({"type": "token", "content": content}) + "\n"

            answer = "".join(answer_parts).strip()
            if not answer:
                raise RuntimeError("The AI model returned an empty answer.")

            document_store.save_conversation(
                document_id=request.document_id,
                question=request.question,
                answer=answer,
                sources=results,
            )
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as error:
            yield json.dumps(
                {
                    "type": "error",
                    "detail": str(error),
                }
            ) + "\n"

    return StreamingResponse(
        generate_events(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/summarize")
def summarize_document(request: SummaryRequest):
    document = get_document(request.document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload it again before summarizing.",
        )

    try:
        summary = answer_service.summarize_document(
            chunks=document["chunks"],
            output_language=request.output_language,
        )
        document_store.save_conversation(
            document_id=request.document_id,
            question=f"Summarize this document in {request.output_language}",
            answer=summary,
            sources=[],
        )
        return {
            "document_id": request.document_id,
            "filename": document["filename"],
            "output_language": request.output_language,
            "summary": summary,
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        print(f"Document summarization failed: {error}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not summarize the document. "
                "Check that Ollama is running and the configured model is available."
            ),
        ) from error
