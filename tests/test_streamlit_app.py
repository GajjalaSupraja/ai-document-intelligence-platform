from streamlit_app import _format_conversation_markdown


def test_format_conversation_markdown_includes_answers_and_sources():
    markdown = _format_conversation_markdown(
        "report.docx",
        [
            {"role": "user", "content": "What is the main topic?"},
            {
                "role": "assistant",
                "content": "The report covers project management.",
                "sources": [
                    {
                        "source": "report.docx",
                        "score": 0.91,
                        "text": "Project management is the main topic.",
                    }
                ],
            },
        ],
    )

    assert "# Document Q&A: report.docx" in markdown
    assert "What is the main topic?" in markdown
    assert "The report covers project management." in markdown
    assert "**report.docx** (relevance: 0.91)" in markdown
    assert "Project management is the main topic." in markdown
