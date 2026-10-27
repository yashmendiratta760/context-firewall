
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)


def sanitize_text(text: str, sensitive_items: list[dict]) -> str:
    """
    Replace detected sensitive spans with placeholders.
    Each span must refer to offsets in the exact input text.
    """
    replacements = []

    for item in sensitive_items:
        span = item.get("span")
        replacement = item.get("replaced_with")

        if not span or not replacement or len(span) != 2:
            continue

        start, end = span

        if not isinstance(start, int) or not isinstance(end, int):
            continue

        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Invalid sensitive span: {span}")

        replacements.append((start, end, replacement))

    # Reject overlapping spans rather than risk incorrect masking.
    ordered = sorted(replacements, key=lambda x: x[0])

    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError("Overlapping sensitive spans detected")

    # Replace from right to left so earlier offsets remain valid.
    for start, end, replacement in reversed(ordered):
        text = text[:start] + replacement + text[end:]

    return text


def create_sanitized_pdf(
    extracted_text: str,
    sensitive_items: list[dict],
    output_path: str,
) -> str:
    """Create a new PDF containing sanitized extracted text."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sanitized = sanitize_text(extracted_text, sensitive_items)

    styles = getSampleStyleSheet()
    line_style = ParagraphStyle(
        "SanitizedLine",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=2,
        wordWrap="CJK",
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Sanitized Document",
    )

    story = []

    for line in sanitized.splitlines():
        # Escape extracted text so PDF content cannot be interpreted
        # as ReportLab paragraph markup.
        story.append(Paragraph(escape(line) or " ", line_style))
        story.append(Spacer(1, 2))

    if not story:
        story.append(Paragraph("No text found.", line_style))

    document.build(story)

    return str(output)
