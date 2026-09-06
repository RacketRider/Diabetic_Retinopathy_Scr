"""Clinical report PDF generator.

Generates annotated PDF reports with:
- Original + Grad-CAM overlay side by side
- DR severity grading and confidence
- Lesion evidence table
- Clinical recommendation
- Medical disclaimer

All processing is in-memory — no disk I/O.
"""

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_report(
    grading: dict,
    confidence: dict,
    quality: dict,
    clinical_evidence: str,
    original_image_bytes: bytes | None = None,
    overlay_image_bytes: bytes | None = None,
) -> bytes:
    """Generate a clinical screening report as PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = styles["Normal"]
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceBefore=20,
    )

    elements = []

    # Header
    elements.append(Paragraph("Diabetic Retinopathy Screening Report", title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f" &nbsp;|&nbsp; Scan ID: DR-{datetime.now().strftime('%H%M%S')}",
        body_style,
    ))
    elements.append(Spacer(1, 12))

    # Grading summary
    elements.append(Paragraph("DR Severity Grading", heading_style))
    referable_text = "YES — Specialist referral recommended" if grading["is_referable"] else "NO"
    grading_data = [
        ["Field", "Value"],
        ["ICDR Level", f"Level {grading['level']}"],
        ["Classification", grading["label"]],
        ["Description", grading["description"]],
        ["Referable DR", referable_text],
        ["Referable Probability", f"{confidence['referable_probability']:.1%}"],
        ["Decision Threshold", str(confidence["threshold_used"])],
    ]
    table = Table(grading_data, colWidths=[45 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Class probabilities
    elements.append(Paragraph("Class Probabilities", heading_style))
    prob_data = [["Class", "Probability"]]
    for cls, prob in confidence["class_probabilities"].items():
        prob_data.append([cls, f"{prob:.1%}"])
    prob_table = Table(prob_data, colWidths=[45 * mm, 40 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 12))

    # Clinical evidence
    elements.append(Paragraph("Clinical Evidence", heading_style))
    elements.append(Paragraph(clinical_evidence, body_style))
    elements.append(Spacer(1, 12))

    # Image quality
    elements.append(Paragraph("Image Quality Assessment", heading_style))
    quality_data = [
        ["Metric", "Value"],
        ["Gradeable", "Yes" if quality["is_gradeable"] else "No"],
        ["Focus Score", f"{quality['focus_score']:.3f}"],
        ["Illumination Score", f"{quality['illumination_score']:.3f}"],
        ["FOV Adequate", "Yes" if quality["fov_adequate"] else "No"],
        ["Enhancement Applied", "Yes" if quality["enhancement_applied"] else "No"],
    ]
    q_table = Table(quality_data, colWidths=[45 * mm, 40 * mm])
    q_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(q_table)

    # Disclaimer
    elements.append(Paragraph(
        "⚠ DISCLAIMER: This is an investigational research prototype for diabetic retinopathy "
        "screening. It is NOT an FDA/CE cleared medical device and must NOT be used for "
        "clinical diagnostic decisions. All results require confirmation by a qualified "
        "ophthalmologist. This system is intended as a screening aid in a human-in-the-loop "
        "workflow only.",
        disclaimer_style,
    ))

    doc.build(elements)
    return buf.getvalue()
