import json

import httpx
import streamlit as st

from app.config import settings


def _stream_answer(
    api_url: str,
    payload: dict,
    answer_placeholder,
) -> tuple[str, list[dict]]:
    answer_parts = []
    sources: list[dict] = []
    completed = False

    with httpx.stream(
        "POST",
        f"{api_url}/ask/stream",
        json=payload,
        timeout=1800,
    ) as response:
        if response.is_error:
            response.read()
            _read_response(response)

        for line in response.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("The streaming API returned invalid data.") from error

            event_type = event.get("type")
            if event_type == "sources":
                sources = event.get("sources", [])
            elif event_type == "token":
                content = event.get("content", "")
                answer_parts.append(content)
                answer_placeholder.markdown("".join(answer_parts) + " ▌")
            elif event_type == "error":
                raise RuntimeError(event.get("detail", "Answer generation failed."))
            elif event_type == "done":
                completed = True

    answer = "".join(answer_parts).strip()
    if not completed:
        raise RuntimeError("The answer stream ended before completion.")
    if not answer:
        raise RuntimeError("The AI model returned an empty answer.")

    answer_placeholder.markdown(answer)
    return answer, sources


def _format_conversation_markdown(
    filename: str,
    chat_history: list[dict],
) -> str:
    lines = [
        f"# Document Q&A: {filename}",
        "",
        "Exported from DocuMind.",
        "",
    ]

    for message in chat_history:
        if message["role"] == "user":
            lines.extend(["## Question", message["content"], ""])
        elif message["role"] == "assistant":
            lines.extend(["## Answer", message["content"], ""])
            sources = message.get("sources", [])
            if sources:
                lines.extend(["### Retrieved sources", ""])
                for index, source in enumerate(sources, start=1):
                    source_name = source.get("source", "Unknown source")
                    score = source.get("score", "N/A")
                    source_text = source.get("text", "").replace("\n", "\n  ")
                    lines.extend(
                        [
                            f"{index}. **{source_name}** "
                            f"(relevance: {score})",
                            f"  {source_text}",
                            "",
                        ]
                    )

    return "\n".join(lines).rstrip() + "\n"


PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #17231f;
    --muted: #74817b;
    --green: #18795b;
    --green-dark: #116248;
    --mint: #eaf5ef;
    --line: #e7ece8;
    --surface: #ffffff;
    --background: #f6f8f6;
}

.stApp {
    background: var(--background);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: rgba(246, 248, 246, 0.88); }
[data-testid="stToolbar"] { right: 1.5rem; }
[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
[data-testid="stMain"] .block-container {
    max-width: 1160px;
    padding: 2rem 2rem 5rem;
}

h1, h2, h3, p, label { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 2px 3px 22px;
}
.brand-mark {
    display: grid;
    place-items: center;
    width: 40px;
    height: 40px;
    border-radius: 13px;
    background: var(--green);
    color: white;
    font-size: 19px;
    box-shadow: 0 7px 18px rgba(24, 121, 91, .2);
}
.brand-name {
    color: var(--ink);
    font: 800 16px 'Manrope', sans-serif;
    letter-spacing: -.04em;
}
.brand-caption { color: var(--muted); font-size: 11px; margin-top: 1px; }

.hero {
    position: relative;
    overflow: hidden;
    padding: 33px 38px;
    border-radius: 24px;
    background: linear-gradient(115deg, #123f32 0%, #18795b 60%, #42a47c 100%);
    color: white;
    box-shadow: 0 16px 36px rgba(20, 86, 62, .14);
    margin-bottom: 25px;
}
.hero:after {
    content: '';
    position: absolute;
    width: 270px;
    height: 270px;
    right: -52px;
    top: -100px;
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 50%;
    box-shadow: 0 0 0 32px rgba(255,255,255,.045), 0 0 0 65px rgba(255,255,255,.035);
}
.hero-kicker {
    color: #bde8d1;
    font: 700 11px 'Manrope', sans-serif;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero h1 {
    color: white;
    font: 800 clamp(27px, 4vw, 39px) 'Manrope', sans-serif;
    letter-spacing: -.055em;
    line-height: 1.14;
    margin: 0 0 10px;
}
.hero p {
    color: rgba(255,255,255,.78);
    font-size: 14px;
    line-height: 1.6;
    max-width: 600px;
    margin: 0;
}

.section-title {
    color: var(--ink);
    font: 800 19px 'Manrope', sans-serif;
    letter-spacing: -.04em;
    margin: 0 0 4px;
}
.section-subtitle { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
.soft-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 21px 23px;
    box-shadow: 0 4px 14px rgba(27, 49, 39, .035);
}
.eyebrow {
    color: var(--green);
    font: 700 10px 'Manrope', sans-serif;
    text-transform: uppercase;
    letter-spacing: .13em;
}
.document-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #15583f;
    background: var(--mint);
    border: 1px solid #d5ebdd;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 8px;
}
.empty-state {
    text-align: center;
    background: white;
    border: 1px dashed #cbd8d0;
    border-radius: 19px;
    padding: 40px 20px;
    margin-top: 20px;
}
.empty-icon {
    display: inline-grid;
    place-items: center;
    width: 54px;
    height: 54px;
    border-radius: 17px;
    background: var(--mint);
    font-size: 24px;
    margin-bottom: 12px;
}
.empty-title { color: var(--ink); font: 800 17px 'Manrope', sans-serif; }
.empty-copy { color: var(--muted); font-size: 13px; max-width: 420px; margin: 6px auto 0; }
.how-step {
    color: #53625a;
    font-size: 12px;
    line-height: 1.55;
    padding: 10px 0;
    border-bottom: 1px solid #eff2ef;
}
.step-number {
    display: inline-grid;
    place-items: center;
    width: 22px;
    height: 22px;
    border-radius: 8px;
    color: var(--green);
    background: var(--mint);
    font-size: 11px;
    font-weight: 800;
    margin-right: 8px;
}

div.stButton > button[kind="primary"] {
    background: var(--green);
    border: 1px solid var(--green);
    border-radius: 11px;
    min-height: 44px;
    font-weight: 700;
    box-shadow: 0 5px 12px rgba(24, 121, 91, .13);
}
div.stButton > button[kind="primary"]:hover {
    background: var(--green-dark);
    border-color: var(--green-dark);
}
div[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 15px 18px;
}
div[data-testid="stChatInput"] {
    border-radius: 15px;
}
div[data-testid="stFileUploader"] section {
    background: #fbfdfb;
    border: 1px dashed #b9d1c2;
    border-radius: 14px;
}
div[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 13px;
}
.footer {
    color: #98a39d;
    border-top: 1px solid var(--line);
    font-size: 11px;
    margin-top: 46px;
    padding-top: 17px;
    text-align: center;
}
@media (max-width: 700px) {
    [data-testid="stMain"] .block-container { padding: 1.2rem 1rem 4rem; }
    .hero { padding: 26px 23px; border-radius: 19px; }
}
</style>
"""


def _read_response(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("The API returned an invalid JSON response.") from error

    if response.is_error:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    if not isinstance(payload, dict):
        raise RuntimeError("The API returned an unexpected response.")

    return payload


def _init_state() -> None:
    defaults = {
        "document_id": None,
        "filename": None,
        "chunk_count": 0,
        "chat_history": [],
        "confirm_delete_document_id": None,
        "original_file_content": None,
        "original_file_filename": None,
        "original_file_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚  View {len(sources)} retrieved source passages"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source.get('source', 'Unknown source')}** "
                f"&nbsp; · &nbsp; relevance `{source.get('score', 'N/A')}`"
            )
            st.write(source.get("text", ""))
            if index < len(sources):
                st.divider()


def main() -> None:
    st.set_page_config(
        page_title="DocuMind — AI Document Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    _init_state()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">✦</div>
                <div>
                    <div class="brand-name">DocuMind</div>
                    <div class="brand-caption">YOUR DOCUMENT, UNDERSTOOD</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Workspace")
        api_url = st.text_input(
            "API server address",
            value="http://127.0.0.1:8000",
            help="The FastAPI server address.",
        ).strip().rstrip("/")
        api_is_ready = False
        if api_url:
            try:
                health_response = httpx.get(f"{api_url}/health", timeout=4)
                _read_response(health_response)
                api_is_ready = True
                st.success("API connected", icon="✅")
            except (httpx.RequestError, RuntimeError) as error:
                st.error(f"API unavailable: {error}", icon="⚠️")
        else:
            st.warning("Enter the API server address.")

        saved_documents = []
        if api_is_ready:
            try:
                saved_response = httpx.get(f"{api_url}/documents", timeout=10)
                saved_documents = _read_response(saved_response).get("documents", [])
            except (httpx.RequestError, RuntimeError) as error:
                st.error(f"Could not load saved documents: {error}")

        if saved_documents:
            st.markdown("---")
            st.markdown("#### Saved documents")
            saved_by_id = {
                item["document_id"]: item for item in saved_documents
            }
            selected_saved_id = st.selectbox(
                "Choose a saved document",
                options=list(saved_by_id),
                format_func=lambda document_id: saved_by_id[document_id]["filename"],
                label_visibility="collapsed",
            )
            if st.button("Open saved document", use_container_width=True):
                try:
                    history_response = httpx.get(
                        f"{api_url}/documents/{selected_saved_id}/history",
                        timeout=10,
                    )
                    saved_history = _read_response(history_response).get("history", [])
                    chat_history = []
                    for turn in saved_history:
                        chat_history.extend(
                            [
                                {"role": "user", "content": turn["question"]},
                                {
                                    "role": "assistant",
                                    "content": turn["answer"],
                                    "sources": turn.get("sources", []),
                                },
                            ]
                        )

                    selected_document = saved_by_id[selected_saved_id]
                    st.session_state.document_id = selected_document["document_id"]
                    st.session_state.filename = selected_document["filename"]
                    st.session_state.chunk_count = selected_document["chunk_count"]
                    st.session_state.chat_history = chat_history
                    st.session_state.original_file_content = None
                    st.rerun()
                except (httpx.RequestError, RuntimeError, KeyError) as error:
                    st.error(f"Could not open saved document: {error}")

            if st.button("Delete selected document", use_container_width=True):
                st.session_state.confirm_delete_document_id = selected_saved_id
                st.rerun()

            if (
                st.session_state.confirm_delete_document_id
                == selected_saved_id
            ):
                st.warning("Delete this document and its saved Q&A history?")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button(
                        "Confirm delete",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            delete_response = httpx.delete(
                                f"{api_url}/documents/{selected_saved_id}",
                                timeout=10,
                            )
                            _read_response(delete_response)
                            if st.session_state.document_id == selected_saved_id:
                                st.session_state.document_id = None
                                st.session_state.filename = None
                                st.session_state.chunk_count = 0
                                st.session_state.chat_history = []
                            st.session_state.confirm_delete_document_id = None
                            st.session_state.original_file_content = None
                            st.success("Document and its history deleted.")
                            st.rerun()
                        except (httpx.RequestError, RuntimeError) as error:
                            st.error(f"Could not delete the document: {error}")
                with cancel_col:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.confirm_delete_document_id = None
                        st.rerun()

        st.markdown("---")
        st.markdown("#### Add a document")
        uploaded_file = st.file_uploader(
            "Document, image, or video",
            type=[
                "pdf", "docx", "png", "jpg", "jpeg", "webp",
                "mp4", "mov", "mkv", "webm",
            ],
            help=(
                "Images are described by the local Ollama vision model. "
                "Videos are transcribed from their audio track."
            ),
            label_visibility="collapsed",
        )
        st.caption(
            "Image uploads require the Ollama vision model. "
            f"Install it with: ollama pull {settings.ollama_vision_model}"
        )
        if st.button(
            "↥  Upload and analyze",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or not api_is_ready,
        ):
            try:
                with st.spinner("Reading and indexing your document..."):
                    response = httpx.post(
                        f"{api_url}/upload-document",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=300,
                    )
                    result = _read_response(response)
                st.session_state.document_id = result["document_id"]
                st.session_state.filename = result["filename"]
                st.session_state.chunk_count = result["chunks_indexed"]
                st.session_state.chat_history = []
                st.session_state.original_file_content = None
                st.success("Document is ready to explore.")
            except (httpx.RequestError, RuntimeError, KeyError) as error:
                st.error(f"Upload failed: {error}")

        if st.session_state.document_id:
            st.markdown("---")
            st.markdown("#### Current document")
            st.markdown(
                f'<div class="document-chip">📄 '
                f'{st.session_state.filename}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{st.session_state.chunk_count} passages indexed")
            if st.button("Prepare original file download", use_container_width=True):
                try:
                    file_response = httpx.get(
                        f"{api_url}/documents/"
                        f"{st.session_state.document_id}/file",
                        timeout=30,
                    )
                    if file_response.is_error:
                        _read_response(file_response)
                    st.session_state.original_file_content = file_response.content
                    st.session_state.original_file_filename = (
                        st.session_state.filename
                    )
                    st.session_state.original_file_type = file_response.headers.get(
                        "content-type",
                        "application/octet-stream",
                    )
                except (httpx.RequestError, RuntimeError) as error:
                    st.error(f"Could not prepare original file: {error}")
            if st.session_state.original_file_content is not None:
                st.download_button(
                    "↓  Download original file",
                    data=st.session_state.original_file_content,
                    file_name=st.session_state.original_file_filename,
                    mime=st.session_state.original_file_type,
                    use_container_width=True,
                )
            if st.button("＋  Start with another document", use_container_width=True):
                st.session_state.document_id = None
                st.session_state.filename = None
                st.session_state.chunk_count = 0
                st.session_state.chat_history = []
                st.session_state.original_file_content = None
                st.rerun()

        st.markdown("---")
        st.markdown("#### How it works")
        st.markdown(
            """
            <div class="how-step"><span class="step-number">1</span>Upload a PDF or DOCX</div>
            <div class="how-step"><span class="step-number">2</span>DocuMind finds relevant passages</div>
            <div class="how-step"><span class="step-number">3</span>Ask questions and explore sources</div>
            """,
            unsafe_allow_html=True,
        )
        top_k = st.slider(
            "Source passages per answer",
            min_value=1,
            max_value=10,
            value=3,
        )
        language_choice = st.selectbox(
            "Answer language",
            ["English", "Telugu", "Hindi", "Custom..."],
        )
        output_language = (
            st.text_input("Custom language")
            if language_choice == "Custom..."
            else language_choice
        )
        st.caption("Answers are generated from your document context.")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI-powered document intelligence</div>
            <h1>Make sense of your documents.</h1>
            <p>Upload a report, study, or project file. Ask questions in plain language
            and get clear answers grounded in the content you shared.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">Document assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">A focused workspace for your document questions.</div>',
            unsafe_allow_html=True,
        )
    with right:
        if st.session_state.document_id:
            st.markdown(
                f'<div style="text-align:right;padding-top:5px">'
                f'<span class="document-chip">● &nbsp; {st.session_state.filename}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if not st.session_state.document_id:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📑</div>
                <div class="empty-title">Your next insight starts with a document</div>
                <div class="empty-copy">Upload a PDF or DOCX from the sidebar. Once it is
                indexed, you can ask questions and see which passages support each answer.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                _render_sources(message.get("sources", []))

        summarize_col, spacer_col = st.columns([1, 2])
        with summarize_col:
            if st.button(
                f"Summarize in {output_language or 'English'}",
                use_container_width=True,
                disabled=not output_language.strip(),
            ):
                summary_question = (
                    f"Summarize this document in {output_language.strip()}"
                )
                try:
                    with st.spinner("Creating a document summary..."):
                        summary_response = httpx.post(
                            f"{api_url}/summarize",
                            json={
                                "document_id": st.session_state.document_id,
                                "output_language": output_language.strip(),
                            },
                            timeout=900,
                        )
                        summary_result = _read_response(summary_response)
                    st.session_state.chat_history.extend(
                        [
                            {"role": "user", "content": summary_question},
                            {
                                "role": "assistant",
                                "content": summary_result["summary"],
                                "sources": [],
                            },
                        ]
                    )
                    st.rerun()
                except (httpx.RequestError, RuntimeError, KeyError) as error:
                    st.error(f"Could not summarize the document: {error}")

        if st.session_state.chat_history:
            export_col, clear_col = st.columns([1, 1])
            with export_col:
                export_content = _format_conversation_markdown(
                    st.session_state.filename,
                    st.session_state.chat_history,
                )
                st.download_button(
                    "↓  Download conversation",
                    data=export_content,
                    file_name="documind-conversation.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with clear_col:
                if st.button(
                    "Clear conversation",
                    use_container_width=True,
                ):
                    try:
                        clear_response = httpx.delete(
                            f"{api_url}/documents/"
                            f"{st.session_state.document_id}/history",
                            timeout=10,
                        )
                        _read_response(clear_response)
                        st.session_state.chat_history = []
                        st.success("Conversation history cleared.")
                        st.rerun()
                    except (httpx.RequestError, RuntimeError) as error:
                        st.error(f"Could not clear conversation: {error}")

        question = st.chat_input(
            f"Ask a question about {st.session_state.filename}..."
        )
        if question:
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                answer_placeholder.caption(
                    "Searching the document and generating an answer..."
                )
                try:
                    answer, sources = _stream_answer(
                        api_url,
                        {
                            "document_id": st.session_state.document_id,
                            "question": question,
                            "top_k": top_k,
                            "output_language": output_language.strip() or "English",
                        },
                        answer_placeholder,
                    )
                    _render_sources(sources)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except (httpx.RequestError, RuntimeError, KeyError) as error:
                    answer_placeholder.error(
                        f"I couldn't answer that right now: {error}"
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"I couldn't answer that right now: {error}",
                        }
                    )
            st.rerun()

    st.markdown(
        '<div class="footer">DocuMind · AI Document Intelligence '
        '· Answers may need verification against the original document.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
