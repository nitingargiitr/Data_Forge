"""
Enhanced PDF Loader Module
Track 4: Contextual Compression for Extreme Long Inputs
"""

import fitz
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DocumentMetadata:
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StructuralElement:
    element_type: str
    content: str
    level: int = 0
    page_number: int = 0
    bbox: Optional[tuple] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedPDFLoader:
    def __init__(self, pdf_path: str, extract_structure: bool = True):
        self.pdf_path = pdf_path
        self.extract_structure = extract_structure
        self.metadata = None
        self.structured_content = []
        
        self.patterns = {
            'header': re.compile(r'^(SECTION|APPENDIX|CHAPTER)\s+\d+[:.\s]', re.IGNORECASE),
            'subheader': re.compile(r'^\d+\.\d+\s+', re.IGNORECASE),
            'exception': re.compile(r'\bEXCEPTION[:\s]', re.IGNORECASE),
            'risk': re.compile(r'\bRISK|WARNING|ALERT|CAUTION[:\s]', re.IGNORECASE),
            'contradiction': re.compile(r'\bCONTRADICTION|CONFLICT', re.IGNORECASE),
            'threshold': re.compile(r'\b(minimum|maximum|threshold|limit)\b.*\d+', re.IGNORECASE),
            'date': re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
        }

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        file_size = os.path.getsize(self.pdf_path)
        
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {str(e)}")
        
        self.metadata = self._extract_metadata(doc, file_size)
        
        pages_data = []
        all_text = []
        
        print(f"[Enhanced Loader] Processing {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            dict_data = page.get_text("dict")
            rect = page.rect
            
            page_info = {
                "page_number": page_num + 1,
                "text": text.strip(),
                "width": rect.width,
                "height": rect.height,
                "word_count": len(text.split()),
                "blocks": self._extract_blocks(dict_data, page_num + 1)
            }
            
            pages_data.append(page_info)
            all_text.append(text)
            
            if self.extract_structure:
                elements = self._analyze_structure(text, page_num + 1)
                self.structured_content.extend(elements)
        
        doc.close()
        
        full_text = "\n".join(all_text)
        cross_refs = self._detect_cross_references(full_text)
        
        result = {
            "metadata": self.metadata.to_dict(),
            "pages": pages_data,
            "structure": [self._element_to_dict(e) for e in self.structured_content],
            "cross_references": cross_refs,
            "stats": {
                "total_pages": len(pages_data),
                "total_words": sum(p["word_count"] for p in pages_data),
                "structural_elements": len(self.structured_content),
                "processing_timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"[Enhanced Loader] ✓ Loaded {result['stats']['total_pages']} pages, "
              f"{result['stats']['total_words']} words, "
              f"{result['stats']['structural_elements']} structural elements")
        
        return result

    def _extract_metadata(self, doc: fitz.Document, file_size: int) -> DocumentMetadata:
        meta = doc.metadata
        
        return DocumentMetadata(
            title=meta.get('title'),
            author=meta.get('author'),
            subject=meta.get('subject'),
            creator=meta.get('creator'),
            producer=meta.get('producer'),
            creation_date=meta.get('creationDate'),
            modification_date=meta.get('modDate'),
            page_count=len(doc),
            file_size=file_size
        )

    def _extract_blocks(self, dict_data: Dict, page_num: int) -> List[Dict]:
        blocks = []
        if "blocks" in dict_data:
            for block in dict_data["blocks"]:
                if "lines" in block:
                    text = "\n".join(
                        " ".join(span["text"] for span in line["spans"])
                        for line in block["lines"]
                    )
                    if text.strip():
                        blocks.append({
                            "text": text.strip(),
                            "bbox": block.get("bbox"),
                            "page": page_num
                        })
        return blocks

    def _analyze_structure(self, text: str, page_num: int) -> List[StructuralElement]:
        elements = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            element_type = 'paragraph'
            level = 0
            
            if self.patterns['header'].match(line):
                element_type = 'header'
                level = 1
            elif self.patterns['subheader'].match(line):
                element_type = 'header'
                level = 2
            elif line.isupper() and len(line) < 100 and len(line) > 10:
                element_type = 'header'
                level = 2
            
            elif self.patterns['exception'].search(line):
                element_type = 'exception'
            elif self.patterns['risk'].search(line):
                element_type = 'risk'
            elif self.patterns['contradiction'].search(line):
                element_type = 'contradiction'
            elif self.patterns['threshold'].search(line):
                element_type = 'threshold'
            
            elements.append(StructuralElement(
                element_type=element_type,
                content=line,
                level=level,
                page_number=page_num,
                metadata={
                    'line_number': i,
                    'has_numbers': bool(re.search(r'\d+', line)),
                    'word_count': len(line.split())
                }
            ))
        
        return elements

    def _detect_cross_references(self, text: str) -> List[Dict]:
        refs = []
        pattern = re.compile(r'(?:Section|Appendix|Chapter)\s+(\d+(?:\.\d+)*)', re.IGNORECASE)
        matches = pattern.findall(text)
        
        for match in set(matches):
            refs.append({
                'type': 'section_reference',
                'target': match,
                'context': 'internal'
            })
        
        return refs

    def _element_to_dict(self, element: StructuralElement) -> Dict[str, Any]:
        return {
            'element_type': element.element_type,
            'content': element.content,
            'level': element.level,
            'page_number': element.page_number,
            'metadata': element.metadata
        }

    def get_decision_critical_content(self) -> List[Dict]:
        if not self.structured_content:
            raise RuntimeError("Must call load() first")
        
        critical_types = {'exception', 'risk', 'contradiction', 'threshold'}
        return [
            self._element_to_dict(e) 
            for e in self.structured_content 
            if e.element_type in critical_types
        ]


if __name__ == "__main__":
    print("Enhanced PDF Loader Module")