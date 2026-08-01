#!/usr/bin/env python3
"""Build data/cv.pdf from data shared with the Jekyll website."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

try:
    import yaml
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        CondPageBreak,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as error:
    missing = getattr(error, "name", "a required package")
    raise SystemExit(
        f"Missing {missing}. Install the resume dependencies with:\n"
        f"  {sys.executable} -m pip install -r tools/resume/requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "_data"
DEFAULT_OUTPUT = ROOT / "data" / "cv.pdf"
BUILD_DIR = ROOT / "tmp" / "pdfs"
SOURCE_PATHS = [
    DATA_DIR / "profile.yml",
    DATA_DIR / "experience.yml",
    DATA_DIR / "publications.yml",
    DATA_DIR / "education.yml",
    DATA_DIR / "code.yml",
    Path(__file__).resolve(),
]

INK = colors.HexColor("#1E1D1A")
MUTED = colors.HexColor("#68635C")
ACCENT = colors.HexColor("#7A2E2A")
RULE = colors.HexColor("#CFC6B8")


def read_yaml(filename: str) -> Any:
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def plain(value: Any) -> str:
    """Convert the small amount of HTML used by Jekyll data to PDF-safe text."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00b7": " | ",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def markup(value: Any, *, bold_name: bool = False) -> str:
    text = escape(plain(value))
    if bold_name:
        for name in ("Luis A. Ortega", "Luis A Ortega"):
            escaped_name = escape(name)
            text = text.replace(escaped_name, f"<b>{escaped_name}</b>")
    return text


def register_fonts() -> dict[str, str]:
    font_sets = [
        {
            "sans": Path("C:/Windows/Fonts/arial.ttf"),
            "sans_bold": Path("C:/Windows/Fonts/arialbd.ttf"),
            "sans_italic": Path("C:/Windows/Fonts/ariali.ttf"),
            "sans_bold_italic": Path("C:/Windows/Fonts/arialbi.ttf"),
            "serif": Path("C:/Windows/Fonts/georgia.ttf"),
            "serif_bold": Path("C:/Windows/Fonts/georgiab.ttf"),
        },
        {
            "sans": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            "sans_bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            "sans_italic": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
            "sans_bold_italic": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
            "serif": Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
            "serif_bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        },
    ]

    for fonts in font_sets:
        if all(path.exists() for path in fonts.values()):
            pdfmetrics.registerFont(TTFont("ResumeSans", str(fonts["sans"])))
            pdfmetrics.registerFont(TTFont("ResumeSans-Bold", str(fonts["sans_bold"])))
            pdfmetrics.registerFont(TTFont("ResumeSans-Italic", str(fonts["sans_italic"])))
            pdfmetrics.registerFont(TTFont("ResumeSans-BoldItalic", str(fonts["sans_bold_italic"])))
            pdfmetrics.registerFont(TTFont("ResumeSerif", str(fonts["serif"])))
            pdfmetrics.registerFont(TTFont("ResumeSerif-Bold", str(fonts["serif_bold"])))
            pdfmetrics.registerFontFamily(
                "ResumeSans",
                normal="ResumeSans",
                bold="ResumeSans-Bold",
                italic="ResumeSans-Italic",
                boldItalic="ResumeSans-BoldItalic",
            )
            pdfmetrics.registerFontFamily(
                "ResumeSerif",
                normal="ResumeSerif",
                bold="ResumeSerif-Bold",
            )
            return {"sans": "ResumeSans", "serif": "ResumeSerif"}

    return {"sans": "Helvetica", "serif": "Times-Roman"}


def make_styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "ResumeName",
            parent=styles["Normal"],
            fontName=fonts["serif"],
            fontSize=25,
            leading=27,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "role": ParagraphStyle(
            "ResumeRole",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=9.5,
            leading=12,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "contact": ParagraphStyle(
            "ResumeContact",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=8.1,
            leading=10,
            textColor=MUTED,
        ),
        "section": ParagraphStyle(
            "ResumeSection",
            parent=styles["Normal"],
            fontName=fonts["serif"],
            fontSize=14.5,
            leading=16.5,
            textColor=INK,
            spaceAfter=0,
        ),
        "entry": ParagraphStyle(
            "ResumeEntry",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=9,
            leading=10.9,
            textColor=INK,
        ),
        "entry_small": ParagraphStyle(
            "ResumeEntrySmall",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=8,
            leading=9.7,
            textColor=MUTED,
        ),
        "meta": ParagraphStyle(
            "ResumeMeta",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=8.1,
            leading=9.8,
            alignment=TA_RIGHT,
            textColor=ACCENT,
        ),
        "footer": ParagraphStyle(
            "ResumeFooter",
            parent=styles["Normal"],
            fontName=fonts["sans"],
            fontSize=6.8,
            leading=8,
            textColor=MUTED,
        ),
    }


def link_markup(label: str, url: str) -> str:
    safe_url = escape(plain(url), {'"': "&quot;"})
    return f'<link href="{safe_url}" color="#7A2E2A"><b>{markup(label)}</b></link>'


def section_heading(title: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    table = Table([[Paragraph(markup(title), styles["section"])]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, -1), 0.65, RULE),
            ]
        )
    )
    return table


def entry_table(
    left: str,
    right: str,
    detail: str,
    styles: dict[str, ParagraphStyle],
    width: float,
    *,
    left_markup: bool = True,
) -> Table:
    left_text = left if left_markup else markup(left)
    rows: list[list[Any]] = [
        [Paragraph(left_text, styles["entry"]), Paragraph(markup(right), styles["meta"])],
    ]
    if detail:
        rows.append([Paragraph(detail, styles["entry_small"]), ""])

    table = Table(rows, colWidths=[width * 0.79, width * 0.21], splitByRow=0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]
    if detail:
        commands.append(("SPAN", (0, 1), (1, 1)))
    table.setStyle(TableStyle(commands))
    return table


def skills_table(
    skills: dict[str, list[str]],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> Table:
    categories = [
        ("Languages", "languages"),
        ("ML frameworks", "ml_frameworks"),
        ("Research methods", "research_methods"),
        ("Research practice", "research_practice"),
    ]
    rows = [
        [
            Paragraph(f"<b>{markup(label)}</b>", styles["entry"]),
            Paragraph(markup(", ".join(skills.get(key, []))), styles["entry_small"]),
        ]
        for label, key in categories
        if skills.get(key)
    ]
    table = Table(rows, colWidths=[width * 0.2, width * 0.8], splitByRow=0)
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    return table


def grouped_section(
    title: str,
    entries: Iterable[Any],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    entries = list(entries)
    if not entries:
        return []
    first = KeepTogether([section_heading(title, styles, width), Spacer(1, 4), entries[0]])
    flowables: list[Any] = [Spacer(1, 7), first]
    for entry in entries[1:]:
        flowables.extend([Spacer(1, 3.2), entry])
    return flowables


def publication_actions(publication: dict[str, Any]) -> str:
    actions = []
    for link in publication.get("links") or []:
        label = plain(link.get("label"))
        if label in {"PDF", "Preprint", "Draft", "Code"}:
            actions.append(link_markup(label, link.get("url", "")))
    return " &nbsp; ".join(actions)


def build_story(
    profile: dict[str, Any],
    experience: list[dict[str, Any]],
    publications: list[dict[str, Any]],
    education: dict[str, Any],
    code: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    story: list[Any] = []

    story.append(Paragraph(markup(profile["name"]), styles["name"]))
    role_line = f'<b>{markup(profile["role"])}</b> &nbsp; | &nbsp; {markup(profile["institution"])} &nbsp; | &nbsp; {markup(profile["location"])}'
    story.append(Paragraph(role_line, styles["role"]))

    contacts = [
        link_markup(profile["email"], f'mailto:{profile["email"]}'),
        link_markup("Website", profile["website"]),
    ]
    contacts.extend(
        link_markup(link["label"], link["url"])
        for link in profile.get("links", [])
        if link.get("resume", False)
    )
    story.append(Paragraph(" &nbsp; | &nbsp; ".join(contacts), styles["contact"]))
    story.append(Spacer(1, 5))

    experience_entries = []
    for position in experience:
        left = f'<b>{markup(position["title"])}</b>, <i>{markup(position["institution"])}</i>'
        experience_entries.append(
            entry_table(
                left,
                plain(position.get("date")),
                markup(position.get("description")),
                styles,
                width,
            )
        )
    story.extend(grouped_section("Experience", experience_entries, styles, width))

    published = [
        publication
        for publication in publications
        if not publication.get("ongoing") and publication.get("include_in_resume", True)
    ]
    publication_entries = []
    for publication in published:
        left = f'<b>{markup(publication["title"])}</b>'
        authors = markup(publication.get("authors"), bold_name=True)
        actions = publication_actions(publication)
        detail_parts = [part for part in (authors, actions) if part]
        publication_entries.append(
            entry_table(
                left,
                plain(publication.get("venue") or publication.get("year")),
                " &nbsp; | &nbsp; ".join(detail_parts),
                styles,
                width,
            )
        )
    story.extend(grouped_section("Publications", publication_entries, styles, width))

    ongoing = [
        publication
        for publication in publications
        if publication.get("ongoing") and publication.get("include_in_resume", True)
    ]
    ongoing_entries = []
    for publication in ongoing:
        left = f'<b>{markup(publication["title"])}</b>'
        details = []
        if publication.get("summary"):
            details.append(markup(publication["summary"]))
        actions = publication_actions(publication)
        if actions:
            details.append(f"<br/>{actions}" if details else actions)
        ongoing_entries.append(
            entry_table(
                left,
                "Ongoing",
                "".join(details),
                styles,
                width,
            )
        )
    story.extend(grouped_section("Ongoing Research", ongoing_entries, styles, width))

    education_entries = []
    for degree in education.get("degrees", []):
        left = f'<b>{markup(degree["title"])}</b>, <i>{markup(degree["institution"])}</i>'
        education_entries.append(
            entry_table(
                left,
                plain(degree.get("date")),
                markup(degree.get("detail")),
                styles,
                width,
            )
        )
    story.append(CondPageBreak(80 * mm))
    story.extend(grouped_section("Education", education_entries, styles, width))

    honor_entries = []
    for honor in education.get("honors", []):
        left = f'<b>{markup(honor["title"])}</b>'
        honor_entries.append(
            entry_table(
                left,
                plain(honor.get("date")),
                markup(honor.get("description")),
                styles,
                width,
            )
        )
    story.extend(grouped_section("Honors & Awards", honor_entries, styles, width))

    code_entries = []
    for repo in code:
        left = link_markup(repo["title"], repo["url"])
        code_entries.append(
            entry_table(
                left,
                "",
                markup(repo.get("description")),
                styles,
                width,
            )
        )
    story.extend(grouped_section("Open Source Contributions", code_entries, styles, width))

    skills = profile.get("skills", {})
    skill_entries = [skills_table(skills, styles, width)]
    story.extend(grouped_section("Skills", skill_entries, styles, width))

    return story


def set_pdf_metadata(
    canvas: Any,
    profile: dict[str, Any],
    content_hash: str,
) -> None:
    canvas.saveState()
    canvas.setTitle(f'{plain(profile["name"])}: CV')
    canvas.setAuthor(plain(profile["name"]))
    canvas.setSubject(f'Resume of {plain(profile["name"])}')
    canvas.setKeywords(f"resume-source-sha256:{content_hash}")
    canvas.restoreState()


def validate_pdf(
    path: Path,
    expected: Iterable[str],
    expected_hash: str,
) -> tuple[int, int]:
    reader = PdfReader(str(path))
    if not 1 <= len(reader.pages) <= 4:
        raise ValueError(f"Unexpected page count: {len(reader.pages)}")

    page_text = [page.extract_text() or "" for page in reader.pages]
    if any(not text.strip() for text in page_text):
        raise ValueError("The generated PDF contains a blank page")

    full_text = "\n".join(page_text)
    missing = [plain(value) for value in expected if plain(value) not in full_text]
    if missing:
        raise ValueError(f"Generated PDF is missing expected text: {missing}")

    link_count = 0
    for page in reader.pages:
        for annotation_ref in page.get("/Annots", []):
            annotation = annotation_ref.get_object()
            if annotation.get("/Subtype") == "/Link":
                link_count += 1
    if link_count == 0:
        raise ValueError("Generated PDF contains no clickable links")

    keywords = plain((reader.metadata or {}).get("/Keywords", ""))
    if keywords != f"resume-source-sha256:{expected_hash}":
        raise ValueError("The PDF was not generated from the current resume sources")

    return len(reader.pages), link_count


def expected_text(
    profile: dict[str, Any],
    experience: list[dict[str, Any]],
    publications: list[dict[str, Any]],
    education: dict[str, Any],
) -> list[str]:
    return [
        profile["name"],
        profile["email"],
        *[position["title"] for position in experience],
        *[
            publication["title"]
            for publication in publications
            if publication.get("include_in_resume", True)
        ],
        *[degree["title"] for degree in education.get("degrees", [])],
    ]


def build_resume(output: Path) -> tuple[int, int]:
    profile = read_yaml("profile.yml")
    experience = read_yaml("experience.yml")
    publications = read_yaml("publications.yml")
    education = read_yaml("education.yml")
    code = read_yaml("code.yml")
    content_hash = source_hash()

    fonts = register_fonts()
    styles = make_styles(fonts)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    temporary_output = BUILD_DIR / f"{output.stem}.build.pdf"
    document = SimpleDocTemplate(
        str(temporary_output),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title=f'{plain(profile["name"])}: CV',
        author=plain(profile["name"]),
        subject=f'Resume of {plain(profile["name"])}',
        pageCompression=1,
    )

    story = build_story(
        profile,
        experience,
        publications,
        education,
        code,
        styles,
        document.width,
    )
    document.build(
        story,
        onFirstPage=lambda canvas, doc: set_pdf_metadata(canvas, profile, content_hash),
        onLaterPages=lambda canvas, doc: set_pdf_metadata(canvas, profile, content_hash),
    )

    expected = expected_text(profile, experience, publications, education)
    page_count, link_count = validate_pdf(temporary_output, expected, content_hash)

    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, output)
    return page_count, link_count


def check_resume(output: Path) -> tuple[int, int]:
    if not output.exists():
        raise ValueError(f"Resume PDF does not exist: {output}")
    profile = read_yaml("profile.yml")
    experience = read_yaml("experience.yml")
    publications = read_yaml("publications.yml")
    education = read_yaml("education.yml")
    return validate_pdf(
        output,
        expected_text(profile, experience, publications, education),
        source_hash(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PDF output path (default: data/cv.pdf)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the existing PDF matches the current data and generator",
    )
    args = parser.parse_args()
    output = args.output.resolve()

    if args.check:
        pages, links = check_resume(output)
        print(f"Verified {output} ({pages} pages, {links} clickable links)")
    else:
        pages, links = build_resume(output)
        print(f"Built {output} ({pages} pages, {links} clickable links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
