
from collections.abc import Iterator


try:
    import ollama
except ImportError:
    class _OllamaFallback:
        def __init__(self):
            self.chat = None

    ollama = _OllamaFallback()


class AnswerService:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        output_language: str = "English",
    ) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return "I could not find relevant information in the uploaded document."

        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            if not isinstance(chunk, dict):
                continue

            source = chunk.get("source") or "Unknown source"
            text = chunk.get("text") or ""

            context_parts.append(f"[Source {index}: {source}]\n{text}")

        if not context_parts:
            return "I could not find relevant information in the uploaded document."

        context = "\n\n".join(context_parts)

        chat_callable = getattr(ollama, "chat", None)
        if chat_callable is None:
            raise RuntimeError(
                "The Ollama client is not available. Install the 'ollama' package."
            )

        response = chat_callable(
            model=self.model_name,
            messages=self._messages(question, context, output_language),
        )

        answer = response.get("message", {}).get("content", "")
        answer = str(answer).strip() if answer is not None else ""

        if not answer:
            raise RuntimeError("The AI model returned an empty answer.")

        return answer

    def stream_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        output_language: str = "English",
    ) -> Iterator[str]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            yield "I could not find relevant information in the uploaded document."
            return

        context_parts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            source = chunk.get("source") or "Unknown source"
            text = chunk.get("text") or ""
            context_parts.append(f"[Source {index}: {source}]\n{text}")

        if not context_parts:
            yield "I could not find relevant information in the uploaded document."
            return

        chat_callable = getattr(ollama, "chat", None)
        if chat_callable is None:
            raise RuntimeError(
                "The Ollama client is not available. Install the 'ollama' package."
            )

        response_stream = chat_callable(
            model=self.model_name,
            messages=self._messages(
                question,
                "\n\n".join(context_parts),
                output_language,
            ),
            stream=True,
        )

        has_content = False
        for response_chunk in response_stream:
            message = response_chunk.get("message", {})
            content = message.get("content", "")
            if content:
                has_content = True
                yield str(content)

        if not has_content:
            raise RuntimeError("The AI model returned an empty answer.")

    def summarize_document(
        self,
        chunks: list[dict],
        output_language: str = "English",
    ) -> str:
        if not chunks:
            raise ValueError("No document content is available to summarize.")

        chat_callable = getattr(ollama, "chat", None)
        if chat_callable is None:
            raise RuntimeError(
                "The Ollama client is not available. Install the 'ollama' package."
            )

        summaries = []
        for index, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            response = chat_callable(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize only the supplied document passage. "
                            f"Write the summary in {output_language}. "
                            "Preserve key facts and do not invent information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Passage {index} of {len(chunks)}:\n{text}"
                        ),
                    },
                ],
            )
            summary = response.get("message", {}).get("content", "")
            summary = str(summary).strip() if summary is not None else ""
            if summary:
                summaries.append(summary)

        if not summaries:
            raise ValueError("No readable document content is available to summarize.")

        while len(summaries) > 8:
            reduced_summaries = []
            for start in range(0, len(summaries), 8):
                reduced_summaries.append(
                    self._combine_summaries(
                        chat_callable,
                        summaries[start : start + 8],
                        output_language,
                    )
                )
            summaries = reduced_summaries

        final_summary = self._combine_summaries(
            chat_callable,
            summaries,
            output_language,
        )
        if not final_summary:
            raise RuntimeError("The AI model returned an empty summary.")
        return final_summary

    def _combine_summaries(
        self,
        chat_callable,
        summaries: list[str],
        output_language: str,
    ) -> str:
        response = chat_callable(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create a concise, well-structured summary from the "
                        "provided passage summaries. Combine repeated ideas, "
                        "preserve important findings and limitations, and "
                        f"write in {output_language}. Do not add facts."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n\n".join(summaries),
                },
            ],
        )
        final_summary = response.get("message", {}).get("content", "")
        final_summary = (
            str(final_summary).strip()
            if final_summary is not None
            else ""
        )
        return final_summary

    @staticmethod
    def _messages(
        question: str,
        context: str,
        output_language: str = "English",
    ) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an AI document assistant. "
                    "Answer the user's question using only the "
                    "provided document excerpts. "
                    "If the excerpts do not contain enough information, "
                    "clearly say so. Do not invent facts. "
                    "Use clear, professional language and refer to "
                    "the relevant source labels when possible. "
                    f"Write your answer in {output_language}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Provide a clear answer based on the excerpts."
                ),
            },
        ]
