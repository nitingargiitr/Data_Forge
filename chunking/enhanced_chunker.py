"""
Enhanced Smart Chunker Module - FIXED VERSION
Track 4: Contextual Compression for Extreme Long Inputs
"""

import re
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ChunkType(Enum):
    STANDARD = "standard"
    CRITICAL = "critical"
    HEADER = "header"
    REFERENCE = "reference"


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page_number: int
    word_count: int
    chunk_type: ChunkType = ChunkType.STANDARD
    section_id: Optional[str] = None
    parent_chunk_id: Optional[int] = None
    contains_numbers: bool = False
    contains_dates: bool = False
    contains_exceptions: bool = False
    contains_risks: bool = False
    contains_contradictions: bool = False
    header_level: int = 0
    source_range: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'page_number': self.page_number,
            'word_count': self.word_count,
            'chunk_type': self.chunk_type.value,
            'section_id': self.section_id,
            'parent_chunk_id': self.parent_chunk_id,
            'contains_numbers': self.contains_numbers,
            'contains_dates': self.contains_dates,
            'contains_exceptions': self.contains_exceptions,
            'contains_risks': self.contains_risks,
            'contains_contradictions': self.contains_contradictions,
            'header_level': self.header_level,
            'source_range': self.source_range
        }


class EnhancedSmartChunker:
    def __init__(self, min_words=50, max_words=300, overlap_words=30, preserve_critical=True):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.preserve_critical = preserve_critical
        
        self.patterns = {
            'number': re.compile(r'\b\d+(?:\.\d+)?\b'),
            'date': re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}\b', re.IGNORECASE),
            'exception': re.compile(r'\bEXCEPTION|EXCEPTION:|unless|only if|however\b', re.IGNORECASE),
            'risk': re.compile(r'\bRISK|WARNING|ALERT|CAUTION|DANGER\b', re.IGNORECASE),
            'contradiction': re.compile(r'\bCONTRADICTION|CONFLICT|vs\.|versus\b', re.IGNORECASE),
            'section_header': re.compile(r'^(?:SECTION|APPENDIX|CHAPTER)\s+(\d+)', re.IGNORECASE),
            'subsection_header': re.compile(r'^(\d+\.\d+)\s+'),
        }

    def chunk_document(self, document_data: Dict[str, Any]) -> List[Chunk]:
        pages = document_data.get('pages', [])
        structure = document_data.get('structure', [])
        
        print(f"[Chunker] Processing {len(pages)} pages...")
        print(f"[Chunker] Structural elements: {len(structure)}")
        
        section_map = self._build_section_map(structure)
        chunks = []
        chunk_id = 0
        current_section = None
        global_char_offset = 0
        
        for page in pages:
            page_num = page['page_number']
            text = page['text']
            paragraphs = self._split_into_paragraphs(text)
            
            buffer = ""
            buffer_start = global_char_offset
            current_header_level = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                para_start = text.find(para, buffer_start - global_char_offset if buffer else 0) + global_char_offset
                
                header_level, section_id = self._classify_header(para)
                
                if header_level > 0 and section_id:
                    current_section = section_id
                    print(f"[Chunker] Found Section {section_id} on page {page_num}")
                
                if header_level > 0:
                    if buffer and len(buffer.split()) >= self.min_words:
                        chunk = self._create_chunk(chunk_id, buffer, page_num, current_section, buffer_start, current_header_level)
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    current_header_level = header_level
                    buffer = para
                    buffer_start = para_start
                    
                    if len(para.split()) > 20:
                        chunk = self._create_chunk(chunk_id, buffer, page_num, current_section, buffer_start, header_level)
                        chunk.chunk_type = ChunkType.HEADER
                        chunks.append(chunk)
                        chunk_id += 1
                        buffer = ""
                        buffer_start = para_start + len(para)
                else:
                    test_buffer = buffer + "\n" + para if buffer else para
                    test_word_count = len(test_buffer.split())
                    is_critical = self._is_critical_content(para)
                    
                    if test_word_count > self.max_words and buffer and not is_critical:
                        chunk = self._create_chunk(chunk_id, buffer, page_num, current_section, buffer_start, current_header_level)
                        chunks.append(chunk)
                        chunk_id += 1
                        
                        overlap_text = self._get_overlap_text(buffer)
                        if overlap_text:
                            buffer = overlap_text + "\n" + para
                            buffer_start = para_start - len(overlap_text)
                        else:
                            buffer = para
                            buffer_start = para_start
                    else:
                        buffer = test_buffer
                        if not buffer or buffer_start == global_char_offset:
                            buffer_start = para_start
            
            if buffer and len(buffer.split()) >= 30:
                chunk = self._create_chunk(chunk_id, buffer, page_num, current_section, buffer_start, current_header_level)
                chunks.append(chunk)
                chunk_id += 1
                buffer = ""
            
            global_char_offset += len(text) + 1
        
        if buffer and len(buffer.split()) >= 30:
            chunk = self._create_chunk(chunk_id, buffer, page_num, current_section, buffer_start, current_header_level)
            chunks.append(chunk)
        
        chunks = self._link_chunks(chunks)
        
        print(f"[Chunker] ✓ Created {len(chunks)} chunks")
        print(f"[Chunker] Critical: {sum(1 for c in chunks if c.chunk_type == ChunkType.CRITICAL)}")
        print(f"[Chunker] Headers: {sum(1 for c in chunks if c.chunk_type == ChunkType.HEADER)}")
        
        sections = {}
        for c in chunks:
            sid = c.section_id or "none"
            sections[sid] = sections.get(sid, 0) + 1
        print(f"[Chunker] Sections found: {sections}")
        
        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n|\n(?=(?:SECTION|APPENDIX|CHAPTER|\d+\.\d+))', text)
        return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10]

    def _classify_header(self, text: str) -> Tuple[int, Optional[str]]:
        match = self.patterns['section_header'].match(text)
        if match:
            return 1, match.group(1)
        
        match = self.patterns['subsection_header'].match(text)
        if match:
            return 2, match.group(1)
        
        if text.isupper() and 10 < len(text) < 150 and len(text.split()) < 15:
            return 2, None
        
        if text.startswith('# ') and len(text) < 150:
            return 2, None
        if text.startswith('## ') and len(text) < 150:
            return 3, None
            
        return 0, None

    def _is_critical_content(self, text: str) -> bool:
        has_exception = self.patterns['exception'].search(text) is not None
        has_risk = self.patterns['risk'].search(text) is not None
        has_contradiction = self.patterns['contradiction'].search(text) is not None
        
        word_count = len(text.split())
        is_substantial = word_count > 15
        
        has_specifics = (
            re.search(r'\d+', text) is not None or
            'unless' in text.lower() or
            'only if' in text.lower() or
            'must' in text.lower() or
            'required' in text.lower() or
            'mandatory' in text.lower()
        )
        
        return (has_exception or has_risk or has_contradiction) and is_substantial and has_specifics

    def _create_chunk(self, chunk_id: int, text: str, page_num: int, section_id: Optional[str], char_start: int, header_level: int) -> Chunk:
        word_count = len(text.split())
        
        chunk_type = ChunkType.STANDARD
        
        if header_level > 0:
            chunk_type = ChunkType.HEADER
        elif self._is_critical_content(text) and word_count > 20:
            chunk_type = ChunkType.CRITICAL
        
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text.strip(),
            page_number=page_num,
            word_count=word_count,
            chunk_type=chunk_type,
            section_id=section_id,
            header_level=header_level,
            source_range=(char_start, char_start + len(text)),
            contains_numbers=self.patterns['number'].search(text) is not None,
            contains_dates=self.patterns['date'].search(text) is not None,
            contains_exceptions=self.patterns['exception'].search(text) is not None,
            contains_risks=self.patterns['risk'].search(text) is not None,
            contains_contradictions=self.patterns['contradiction'].search(text) is not None
        )
        
        return chunk

    def _get_overlap_text(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.overlap_words:
            return ""
        return " ".join(words[-self.overlap_words:])

    def _build_section_map(self, structure: List[Dict]) -> Dict[str, List[int]]:
        section_map = {}
        for i, elem in enumerate(structure):
            content = elem.get('content', '')
            match = re.search(r'(?:SECTION|APPENDIX|CHAPTER)?\s*(\d+(?:\.\d+)*)', content, re.IGNORECASE)
            if match:
                section_id = match.group(1)
                if section_id not in section_map:
                    section_map[section_id] = []
                section_map[section_id].append(i)
        return section_map

    def _link_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        section_chunks = {}
        
        for chunk in chunks:
            if chunk.section_id:
                if chunk.section_id not in section_chunks:
                    section_chunks[chunk.section_id] = []
                section_chunks[chunk.section_id].append(chunk)
        
        for section_id, sec_chunks in section_chunks.items():
            header_chunk = None
            for chunk in sec_chunks:
                if chunk.chunk_type == ChunkType.HEADER:
                    header_chunk = chunk
                elif header_chunk and chunk.header_level == 0:
                    chunk.parent_chunk_id = header_chunk.chunk_id
        
        return chunks

    def get_critical_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        return [c for c in chunks if c.chunk_type == ChunkType.CRITICAL]


if __name__ == "__main__":
    print("Enhanced Smart Chunker Module - FIXED")
