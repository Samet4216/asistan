"""
JSON Schemas for Structured Answer Pipeline (Faz 7).
Enforces quote-first attribution before generating final synthetic answer text.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "has_evidence": {
            "type": "boolean",
            "description": "True if sufficient evidence exists in retrieved context, False if unsure or insufficient."
        },
        "quotes": {
            "type": "array",
            "description": "Verbatim snippets quoted directly from the retrieved context.",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Section or Paper ID citation"},
                    "text": {"type": "string", "description": "Exact quoted sentence from context"}
                },
                "required": ["source", "text"]
            }
        },
        "answer": {
            "type": "string",
            "description": "Concise, technical synthesis strictly grounded in the extracted quotes."
        },
        "key_takeaways": {
            "type": "array",
            "description": "Key findings, mathematical insights, or tensor dimension constraints.",
            "items": {"type": "string"}
        }
    },
    "required": ["has_evidence", "quotes", "answer", "key_takeaways"]
}
