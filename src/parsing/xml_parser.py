"""
AeroLex — eCFR XML Parser

Parses raw eCFR XML files downloaded by ECFRIngestor into
structured, clean text chunks with hierarchy metadata.

Input:  data/raw/ecfr/part_91_2026-04-14.xml
Output: List of ParsedSection objects, each containing:
        - part number, subpart, section number, section title
        - clean text content
        - full citation (e.g., "14 CFR Part 91, Section 91.103")
        - hierarchy path (for metadata filtering in RAG)

Why lxml over built-in xml.etree?
- Faster parsing for large XML files
- Better handling of malformed XML (common in government docs)
- XPath support for precise element selection
- Handles XML namespaces cleanly

Usage:
    from src.parsing.xml_parser import ECFRXMLParser
    parser = ECFRXMLParser()
    sections = parser.parse_part_file("data/raw/ecfr/part_91_2026-04-14.xml")
    for section in sections[:3]:
        print(section)
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from lxml import etree
from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, ParsingError

logger = get_logger(__name__)

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ParsedSection:
    """
    Represents one parsed section from eCFR regulations.
    This is the unit of text that will be chunked and embedded.

    Example:
        part_number:  "91"
        part_title:   "General Operating and Flight Rules"
        subpart:      "B"
        subpart_title:"Flight Rules"
        section:      "91.103"
        section_title:"Preflight action"
        text:         "Each pilot in command shall, before beginning a flight..."
        citation:     "14 CFR Part 91, Section 91.103"
        hierarchy:    "Title 14 > Part 91 > Subpart B > Section 91.103"
        source:       "ecfr"
        doc_type:     "regulation"
    """
    part_number:    str
    part_title:     str
    subpart:        str
    subpart_title:  str
    section:        str
    section_title:  str
    text:           str
    citation:       str
    hierarchy:      str
    source:         str = "ecfr"
    doc_type:       str = "regulation"

    def __str__(self):
        return (
            f"[{self.citation}]\n"
            f"Hierarchy: {self.hierarchy}\n"
            f"Text preview: {self.text[:200]}..."
        )


# ── Parser ────────────────────────────────────────────────────────────────────

class ECFRXMLParser:
    """
    Parses eCFR XML files into structured ParsedSection objects.

    XML Structure of eCFR Title 14:
        DIV5 (PART)
            HEAD (Part title)
            DIV6 (SUBPART)
                HEAD (Subpart title)
                DIV8 (SECTION)
                    HEAD (Section number + title)
                    P (Paragraph text)
                    ...
    """

    def __init__(self):
        logger.info("ECFRXMLParser initialized")

    def _clean_text(self, text: str) -> str:
        """
        Clean raw XML text content.
        Removes extra whitespace, fixes encoding issues.

        Args:
            text: Raw text from XML element

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix common HTML entities that slip through
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#160;', ' ')  # Non-breaking space

        return text

    def _extract_all_text(self, element) -> str:
        """
        Extract all text content from an XML element and its children.
        Preserves paragraph structure with newlines.

        Args:
            element: lxml element

        Returns:
            str: All text content concatenated
        """
        texts = []

        # Get direct text
        if element.text:
            cleaned = self._clean_text(element.text)
            if cleaned:
                texts.append(cleaned)

        # Get text from all child elements
        for child in element:
            child_tag = child.tag if isinstance(child.tag, str) else ""

            # Skip HEAD elements — already captured as titles
            if child_tag == "HEAD":
                continue

            child_text = self._extract_all_text(child)
            if child_text:
                texts.append(child_text)

            # Get tail text (text after closing tag)
            if child.tail:
                cleaned = self._clean_text(child.tail)
                if cleaned:
                    texts.append(cleaned)

        return " ".join(texts)

    def _parse_part_title(self, root_element) -> str:
        """Extract part title from HEAD element."""
        head = root_element.find("HEAD")
        if head is not None and head.text:
            return self._clean_text(head.text)
        return f"Part {root_element.get('N', 'Unknown')}"

    def parse_part_file(self, xml_file_path: str) -> list[ParsedSection]:
        """
        Parse a complete eCFR Part XML file into ParsedSection objects.

        Args:
            xml_file_path: Path to the XML file

        Returns:
            list[ParsedSection]: List of parsed sections
        """
        xml_path = Path(xml_file_path)

        if not xml_path.exists():
            raise ParsingError(
                message=f"XML file not found: {xml_file_path}",
                context="ECFRXMLParser.parse_part_file()"
            )

        logger.info(f"Parsing XML file: {xml_path.name}")

        try:
            # Parse XML with lxml
            # recover=True handles minor XML errors gracefully
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            tree = etree.parse(str(xml_path), parser)
            root = tree.getroot()

        except Exception as e:
            raise ParsingError(
                message=f"Failed to parse XML: {xml_path.name}",
                context="ECFRXMLParser.parse_part_file()",
                original_error=e
            )

        # Extract part number from filename or XML
        part_number = root.get("N", xml_path.stem.split("_")[1])
        part_title = self._parse_part_title(root)

        logger.info(f"Parsing Part {part_number}: {part_title}")

        sections = []

        # ── Walk the XML tree ─────────────────────────────────────────────
        # DIV5 = PART (root)
        # DIV6 = SUBPART
        # DIV8 = SECTION

        # Find all subparts (DIV6)
        subparts = root.findall(".//DIV6[@TYPE='SUBPART']")

        if subparts:
            # Parse with subpart structure
            for subpart_elem in subparts:
                subpart_id = subpart_elem.get("N", "")
                subpart_head = subpart_elem.find("HEAD")
                subpart_title = self._clean_text(
                    subpart_head.text if subpart_head is not None else ""
                )

                # Find all sections in this subpart (DIV8)
                section_elems = subpart_elem.findall(".//DIV8[@TYPE='SECTION']")

                for section_elem in section_elems:
                    parsed = self._parse_section(
                        section_elem=section_elem,
                        part_number=part_number,
                        part_title=part_title,
                        subpart=subpart_id,
                        subpart_title=subpart_title
                    )
                    if parsed:
                        sections.append(parsed)
        else:
            # Some parts have no subparts — sections directly under part
            section_elems = root.findall(".//DIV8[@TYPE='SECTION']")
            for section_elem in section_elems:
                parsed = self._parse_section(
                    section_elem=section_elem,
                    part_number=part_number,
                    part_title=part_title,
                    subpart="",
                    subpart_title=""
                )
                if parsed:
                    sections.append(parsed)

        logger.info(
            f"Part {part_number} parsed | "
            f"Sections found: {len(sections)}"
        )
        return sections

    def _parse_section(
        self,
        section_elem,
        part_number: str,
        part_title: str,
        subpart: str,
        subpart_title: str
    ) -> Optional[ParsedSection]:
        """
        Parse a single section element into a ParsedSection.

        Args:
            section_elem: lxml element for DIV8 SECTION
            part_number: Parent part number
            part_title: Parent part title
            subpart: Parent subpart identifier
            subpart_title: Parent subpart title

        Returns:
            ParsedSection or None if section has no useful content
        """
        try:
            # Section number (e.g., "91.103")
            section_number = section_elem.get("N", "")

            # Section title from HEAD element
            head_elem = section_elem.find("HEAD")
            if head_elem is not None:
                section_title = self._clean_text(head_elem.text or "")
            else:
                section_title = section_number

            # Extract all text content
            text = self._extract_all_text(section_elem)

            # Skip empty or very short sections (likely reserved/placeholder)
            if len(text.strip()) < 20:
                logger.debug(f"Skipping short section {section_number}: too short")
                return None

            # Build citation and hierarchy
            citation = f"14 CFR Part {part_number}, Section {section_number}"
            if subpart:
                hierarchy = (
                    f"Title 14 > Part {part_number} ({part_title}) > "
                    f"Subpart {subpart} ({subpart_title}) > "
                    f"Section {section_number}"
                )
            else:
                hierarchy = (
                    f"Title 14 > Part {part_number} ({part_title}) > "
                    f"Section {section_number}"
                )

            return ParsedSection(
                part_number=part_number,
                part_title=part_title,
                subpart=subpart,
                subpart_title=subpart_title,
                section=section_number,
                section_title=section_title,
                text=text,
                citation=citation,
                hierarchy=hierarchy
            )

        except Exception as e:
            handle_exception(
                e,
                context=f"ECFRXMLParser._parse_section({section_number})"
            )
            return None

    def save_parsed_sections(
        self,
        sections: list[ParsedSection],
        output_path: str
    ) -> None:
        """
        Save parsed sections to a JSON file for downstream processing.

        Args:
            sections: List of ParsedSection objects
            output_path: Where to save the JSON file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(s) for s in sections]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(sections)} parsed sections to: {output}")


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n--- Testing eCFR XML Parser ---\n")

    parser = ECFRXMLParser()

    # Test with Part 1 (small file — good for testing)
    import glob
    part1_files = glob.glob("data/raw/ecfr/part_1_*.xml")
    part91_files = glob.glob("data/raw/ecfr/part_91_*.xml")

    if not part1_files:
        print("❌ No Part 1 XML found — run ecfr_ingestor.py first!")
        exit(1)

    # Parse Part 1
    print("Test 1: Parsing Part 1 (Definitions)...")
    sections_1 = parser.parse_part_file(part1_files[0])
    print(f"  Sections found: {len(sections_1)}")
    if sections_1:
        print(f"\n  First section:")
        print(f"  {sections_1[0]}")

    # Parse Part 91
    if part91_files:
        print(f"\nTest 2: Parsing Part 91 (General Operating Rules)...")
        sections_91 = parser.parse_part_file(part91_files[0])
        print(f"  Sections found: {len(sections_91)}")
        if sections_91:
            print(f"\n  Sample section (index 5):")
            print(f"  {sections_91[min(5, len(sections_91)-1)]}")

        # Save parsed output
        output_path = "data/processed/part_91_parsed.json"
        parser.save_parsed_sections(sections_91, output_path)
        print(f"\n  ✅ Saved to: {output_path}")

    print("\n✅ XML Parser working!")