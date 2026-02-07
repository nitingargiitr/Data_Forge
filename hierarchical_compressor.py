import os
import sys
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field, replace
from datetime import datetime
from collections import defaultdict

# Add project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ingestion.enhanced_pdf_loader import EnhancedPDFLoader
from chunking.enhanced_chunker import EnhancedSmartChunker, ChunkType
from summarizer.enhanced_ai_summarizer import EnhancedAISummarizer, SummaryStrategy


@dataclass
class CompressionReport:
    document_name: str
    compression_date: str
    original_stats: Dict[str, Any]
    levels: List[Dict[str, Any]] = field(default_factory=list)
    information_loss_score: float = 0.0
    critical_preservation_rate: float = 1.0
    contradiction_count: int = 0
    traceability_links: List[Dict] = field(default_factory=list)
    compression_decisions: List[Dict[str, str]] = field(default_factory=list)
    critical_facts_summary: List[Dict] = field(default_factory=list)  # NEW

    def to_dict(self):
        return {
            "document_name": self.document_name,
            "compression_date": self.compression_date,
            "original_stats": self.original_stats,
            "levels": self.levels,
            "quality_metrics": {
                "information_loss_score": self.information_loss_score,
                "critical_preservation_rate": self.critical_preservation_rate,
                "contradiction_count": self.contradiction_count
            },
            "critical_facts_summary": self.critical_facts_summary,  # NEW
            "compression_decisions": self.compression_decisions
        }


class HierarchicalCompressor:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.document_data = None
        self.chunks = None
        self.chunk_summaries = None
        self.section_summaries = None
        self.document_summary = None
        self.report = None


    def compress_document(self, pdf_path: str, config: Dict[str, Any] = None):
            """
            Compress document with configurable parameters
            
            Args:
                pdf_path: Path to PDF file
                config: Configuration dict with keys:
                    - min_words: int (default 30)
                    - max_words: int (default 250)
                    - overlap_words: int (default 30)
                    - doc_max_length: int (default 300) - NEW!
                    - strategy: str (default 'extractive')
            """
            # Default config
            default_config = {
                'min_words': 75,
                'max_words': 300,
                'overlap_words': 20,
                'doc_max_length': 300,  # NEW!
                'strategy': 'extractive'
            }
            
            # Merge with provided config
            if config:
                default_config.update(config)
            config = default_config
            
            print("=" * 70)
            print("HIERARCHICAL COMPRESSION ENGINE")
            print("=" * 70)
            print(f"PDF: {pdf_path}")
            print(f"Config: {config}\\n")

            # Step 1: Load
            print("[1/5] Loading PDF...")
            loader = EnhancedPDFLoader(pdf_path)
            self.document_data = loader.load()

            original_stats = {
                "pages": self.document_data["stats"]["total_pages"],
                "words": self.document_data["stats"]["total_words"],
                "structural_elements": self.document_data["stats"]["structural_elements"]
            }

            # Step 2: Chunk - USE CONFIG
            print("\\n[2/5] Chunking...")
            chunker = EnhancedSmartChunker(
                min_words=config['min_words'],
                max_words=config['max_words'],
                overlap_words=config['overlap_words']
            )
            self.chunks = chunker.chunk_document(self.document_data)

            # Step 3: Summarize chunks
            print("\\n[3/5] Summarizing chunks...")
            summarizer = EnhancedAISummarizer(strategy=SummaryStrategy.HYBRID)
            
            chunk_dicts = [c.to_dict() for c in self.chunks]
            self.chunk_summaries = summarizer.summarize_chunks(chunk_dicts)

            # Step 4: Section summaries
            print("\\n[4/5] Creating section summaries...")
            self.section_summaries = self._create_section_summaries(summarizer)

            # Step 5: Document summary - PASS MAX_LENGTH
            print("\\n[5/5] Creating document summary...")
            self.document_summary = self._create_document_summary(
                summarizer, 
                max_length=config['doc_max_length']  # NEW!
            )

            # Step 6: Extract critical facts
            print("\\n[6/6] Extracting critical facts...")
            critical_facts = self._extract_critical_facts()

            # Build report
            contradictions = self._detect_contradictions()
            self.report = self._build_report(pdf_path, original_stats, contradictions, critical_facts)

            # Final print
            print("\\n" + "=" * 70)
            print("COMPRESSION COMPLETE")
            print("=" * 70)
            print(f"Original Words: {original_stats['words']:,}")
            print(f"Final Words: {self.document_summary.summary_words:,}")
            
            ratio = self.document_summary.summary_words / original_stats["words"]
            print(f"Compression Ratio: {ratio:.1%}")
            print(f"Critical Preservation: {self.report.critical_preservation_rate:.1%}")
            print(f"Information Loss: {self.report.information_loss_score:.2f}")
            print(f"Total Chunks: {len(self.chunks)}")
            print(f"Critical Facts Extracted: {len(critical_facts)}")
            print("=" * 70)

            return self.report

    def _create_document_summary(self, summarizer, max_length: int = 300):
        """Create document summary with STRICT length enforcement"""

        # Combine section summaries
        combined_text = " ".join([s.summary_text for s in self.section_summaries])
        original_word_count = len(combined_text.split())

        print(f"[Compressor] Creating document summary from {len(self.section_summaries)} sections...")
        print(f"[Compressor] Combined text: {original_word_count} words")
        print(f"[Compressor] Target max length: {max_length} words (STRICT)")

        # STRICT: Calculate exact ratio needed
        if original_word_count <= max_length:
            # If already under max, use as-is but ensure minimum 150 words
            if original_word_count < 150:
                # Need to expand slightly
                target_ratio = 0.9  # Keep most of it
            else:
                summary_text = combined_text
                summary_words = original_word_count
                print(f"[Compressor] Already under max: {summary_words} words")
                # Skip to result creation
                from dataclasses import dataclass
                
                @dataclass
                class DocSummary:
                    summary_text: str
                    summary_words: int
                    original_words: int
                    compression_ratio: float
                    confidence: float
                    level: str
                    source_chunks: list
                    source_pages: list
                    explainability: dict
                    strategy: any
                    processing_time: float
                    preserved_critical: list

                result = DocSummary(
                    summary_text=summary_text,
                    summary_words=summary_words,
                    original_words=original_word_count,
                    compression_ratio=summary_words / original_word_count if original_word_count > 0 else 0,
                    confidence=0.85,
                    level="document",
                    source_chunks=[s.source_chunks[0] for s in self.section_summaries if s.source_chunks],
                    source_pages=list(set([p for s in self.section_summaries for p in s.source_pages])),
                    explainability={
                        'inclusion_reason': f"Aggregated summary of {len(self.section_summaries)} sub-sections",
                        'structural_role': 'Document level overview',
                        'preservation_priority': 'high',
                        'sections_covered': list(set(s.section_id for s in self.section_summaries if s.section_id)),
                        'total_critical_preserved': sum(len(s.preserved_critical) for s in self.section_summaries),
                        'avg_confidence': sum(s.confidence for s in self.section_summaries) / len(self.section_summaries) if self.section_summaries else 0,
                        'compression_note': f"No summarization applied, already under {max_length} words"
                    },
                    strategy=SummaryStrategy.EXTRACTIVE,
                    processing_time=0.0,
                    preserved_critical=[]
                )

                return result
        else:
            # Need to compress - be aggressive
            target_ratio = (max_length - 10) / original_word_count  # -10 for safety margin
            target_ratio = max(0.10, min(0.30, target_ratio))  # Keep between 10-30%

        # Apply summarization with calculated ratio
        summary_text = summarizer._extractive_summarize(combined_text, ratio=target_ratio)
        summary_words = len(summary_text.split())

        print(f"[Compressor] Initial compression: {original_word_count} → {summary_words} words (ratio: {target_ratio:.2%})")

        # STRICT CHECK: If still over max, truncate hard
        if summary_words > max_length:
            print(f"[Compressor] WARNING: Still over max ({summary_words} > {max_length}), forcing truncation...")
            words = summary_text.split()
            truncated = words[:max_length]
            summary_text = ' '.join(truncated) + '...'
            summary_words = len(truncated)
            print(f"[Compressor] ✓ Hard truncated to {summary_words} words")

        # If too short, expand (but not over max)
        elif summary_words < 150 and summary_words < max_length:
            print(f"[Compressor] Expanding from {summary_words} words...")
            expansion_room = max_length - summary_words

            expanded_parts = [summary_text]
            current_count = summary_words

            for sec_summary in self.section_summaries:
                if current_count >= 200 or current_count >= max_length - 50:
                    break
                sentences = sec_summary.summary_text.split('.')
                for sent in sentences[:1]:
                    sent = sent.strip()
                    if len(sent) > 20 and sent not in summary_text:
                        sent_words = len(sent.split())
                        if current_count + sent_words <= max_length - 20:
                            expanded_parts.append(sent)
                            current_count += sent_words

            summary_text = '. '.join(expanded_parts)
            if not summary_text.endswith('.'):
                summary_text += '.'
            summary_words = len(summary_text.split())
            print(f"[Compressor] ✓ Expanded to {summary_words} words (max was {max_length})")

        # FINAL CHECK: Ensure we're at or under max
        if summary_words > max_length:
            words = summary_text.split()
            summary_text = ' '.join(words[:max_length]) + '...'
            summary_words = max_length
            print(f"[Compressor] ✓ Final forced truncation to {max_length} words")

        # Create result
        from dataclasses import dataclass

        @dataclass
        class DocSummary:
            summary_text: str
            summary_words: int
            original_words: int
            compression_ratio: float
            confidence: float
            level: str
            source_chunks: list
            source_pages: list
            explainability: dict
            strategy: any
            processing_time: float
            preserved_critical: list

        result = DocSummary(
            summary_text=summary_text,
            summary_words=summary_words,
            original_words=original_word_count,
            compression_ratio=summary_words / original_word_count if original_word_count > 0 else 0,
            confidence=0.85,
            level="document",
            source_chunks=[s.source_chunks[0] for s in self.section_summaries if s.source_chunks],
            source_pages=list(set([p for s in self.section_summaries for p in s.source_pages])),
            explainability={
                'inclusion_reason': f"Aggregated summary of {len(self.section_summaries)} sub-sections",
                'structural_role': 'Document level overview',
                'preservation_priority': 'high',
                'sections_covered': list(set(s.section_id for s in self.section_summaries if s.section_id)),
                'total_critical_preserved': sum(len(s.preserved_critical) for s in self.section_summaries),
                'avg_confidence': sum(s.confidence for s in self.section_summaries) / len(self.section_summaries) if self.section_summaries else 0,
                'compression_note': f"Extractive summarization ({target_ratio:.0%} ratio) applied to {original_word_count} words, max {max_length} words"
            },
            strategy=SummaryStrategy.EXTRACTIVE,
            processing_time=0.0,
            preserved_critical=[]
        )

        return result


    def _create_section_summaries(self, summarizer):
        """FIXED: Properly group by section_id"""
        section_groups = defaultdict(list)
        
        # Map chunk_id to section_id for lookup
        chunk_to_section = {}
        for chunk in self.chunks:
            chunk_to_section[chunk.chunk_id] = chunk.section_id or "uncategorized"
        
        # Group summaries by section
        for summary in self.chunk_summaries:
            if summary.source_chunks:
                chunk_id = summary.source_chunks[0]
                section_id = chunk_to_section.get(chunk_id, "uncategorized")
                section_groups[section_id].append(summary)
        
        section_summaries = []
        for section_id, summaries in section_groups.items():
            print(f"[Compressor] Section {section_id}: {len(summaries)} chunks")
            if len(summaries) == 1:
                # Single chunk - use its summary but ensure section_id is set
                sec_summary = summaries[0]
                sec_summary.section_id = section_id
                section_summaries.append(sec_summary)
            else:
                # Multiple chunks - summarize them
                sec = summarizer.summarize_summaries(summaries, level="section")
                sec.section_id = section_id
                section_summaries.append(sec)
        
        return section_summaries

    def _extract_critical_facts(self) -> List[Dict]:
        """NEW: Extract all critical facts at document level"""
        critical_facts = []
        
        for chunk, summary in zip(self.chunks, self.chunk_summaries):
            if chunk.chunk_type == ChunkType.CRITICAL or chunk.contains_risks or chunk.contains_exceptions:
                fact = {
                    "section": chunk.section_id or "unknown",
                    "page": chunk.page_number,
                    "type": "critical",
                    "summary": summary.summary_text,
                    "details": {
                        "has_exception": chunk.contains_exceptions,
                        "has_risk": chunk.contains_risks,
                        "has_contradiction": chunk.contains_contradictions,
                        "has_numbers": chunk.contains_numbers
                    },
                    "source_range": chunk.source_range,
                    "chunk_id": chunk.chunk_id
                }
                critical_facts.append(fact)
        
        return critical_facts

    def _detect_contradictions(self):
        count = 0
        for elem in self.document_data.get("structure", []):
            if elem.get("element_type") == "contradiction":
                count += 1
        for c in self.chunks:
            if c.contains_contradictions:
                count += 1
        return count

    def _build_report(self, pdf_path, original_stats, contradictions, critical_facts):
        report = CompressionReport(
            document_name=os.path.basename(pdf_path),
            compression_date=datetime.now().isoformat(),
            original_stats=original_stats,
            contradiction_count=contradictions,
            critical_facts_summary=critical_facts  # NEW
        )

        # Raw level
        report.levels.append({
            "level_name": "raw",
            "item_count": len(self.document_data["pages"]),
            "total_words": original_stats["words"],
            "compression_ratio": 1.0
        })

        # Chunk level - FIXED with proper section_id
        chunk_total = sum(s.summary_words for s in self.chunk_summaries)
        report.levels.append({
            "level_name": "chunk",
            "item_count": len(self.chunk_summaries),
            "total_words": chunk_total,
            "compression_ratio": chunk_total / original_stats["words"],
            "items": [
                {
                    "chunk_id": s.source_chunks[0] if s.source_chunks else i,
                    "section_id": chunk.section_id or "uncategorized",  # FIXED
                    "summary": s.summary_text,
                    "confidence": s.confidence,
                    "explainability": s.explainability,
                    "source_range": chunk.source_range,
                    "page_number": chunk.page_number
                }
                for i, (s, chunk) in enumerate(zip(self.chunk_summaries, self.chunks))
            ]
        })

        # Section level - FIXED with proper section_id
        sec_total = sum(s.summary_words for s in self.section_summaries)
        report.levels.append({
            "level_name": "section",
            "item_count": len(self.section_summaries),
            "total_words": sec_total,
            "compression_ratio": sec_total / chunk_total if chunk_total > 0 else 0,
            "items": [
                {
                    "section_id": getattr(s, 'section_id', 'unknown'),  # FIXED
                    "summary": s.summary_text,
                    "confidence": s.confidence,
                    "explainability": s.explainability
                }
                for s in self.section_summaries
            ]
        })

        # Document level
        report.levels.append({
            "level_name": "document",
            "item_count": 1,
            "total_words": self.document_summary.summary_words,
            "compression_ratio": self.document_summary.summary_words / sec_total if sec_total > 0 else 0,
            "items": [
                {
                    "summary": self.document_summary.summary_text,
                    "confidence": self.document_summary.confidence,
                    "explainability": self.document_summary.explainability
                }
            ]
        })

        # Metrics
        report.critical_preservation_rate = self._calc_critical_preservation()
        report.information_loss_score = self._calc_info_loss()

        # Decisions
        report.compression_decisions = [
            {
                "decision": "Chunking",
                "rationale": f"{len(self.chunks)} semantic chunks with structural awareness (avg {sum(c.word_count for c in self.chunks)//len(self.chunks) if self.chunks else 0} words/chunk)"
            },
            {
                "decision": "Summarization",
                "rationale": "Hybrid approach: extractive for critical, abstractive for narrative"
            },
            {
                "decision": "Critical Content",
                "rationale": f"{sum(1 for c in self.chunks if c.chunk_type == ChunkType.CRITICAL)} critical chunks preserved with {len(critical_facts)} critical facts extracted"
            },
            {
                "decision": "Per-Item Explainability",
                "rationale": "Each chunk includes inclusion reason, priority, content removed, and preservation details"
            },
            {
                "decision": "Document Length Optimization",
                "rationale": f"Ensured {self.document_summary.summary_words} words for readable document summary (target: 200-300 words)"
            },
            {
                "decision": "Section Tracking",
                "rationale": f"Preserved {len(set(c.section_id for c in self.chunks if c.section_id))} unique section identifiers for traceability"
            }
        ]

        return report

    def _calc_critical_preservation(self):
        """Calculate critical preservation rate - FIXED"""
        total_critical = 0
        preserved = 0

        for c, s in zip(self.chunks, self.chunk_summaries):
            # Check if chunk is critical OR contains critical content
            is_critical = (
                c.chunk_type == ChunkType.CRITICAL or 
                c.contains_exceptions or 
                c.contains_risks or 
                c.contains_contradictions
            )

            if is_critical:
                total_critical += 1
                # FIXED: Use >= instead of > to include 0.30 confidence
                if s.confidence >= 0.3:
                    preserved += 1
                print(f"  [Critical Check] Chunk {c.chunk_id}: type={c.chunk_type.value}, "
                    f"exc={c.contains_exceptions}, risk={c.contains_risks}, "
                    f"conf={s.confidence:.2f}, preserved={s.confidence >= 0.3}")

        rate = preserved / total_critical if total_critical > 0 else 1.0
        print(f"[Compressor] Critical preservation: {preserved}/{total_critical} = {rate:.1%}")
        return rate


    def _calc_info_loss(self):
        final_ratio = (
            self.document_summary.summary_words /
            self.document_data["stats"]["total_words"]
        )
        avg_conf = sum(
            s.confidence for s in self.chunk_summaries
        ) / len(self.chunk_summaries) if self.chunk_summaries else 0
        crit_rate = self._calc_critical_preservation()
        
        loss = (
            (1 - final_ratio) * 0.3 +
            (1 - avg_conf) * 0.4 +
            (1 - crit_rate) * 0.3
        )
        return max(0.0, min(1.0, loss))

    def export(self):
        outputs = {}
        
        json_path = os.path.join(self.output_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, indent=2)
        outputs['json'] = json_path
        print(f"[Saved] {json_path}")
        
        # NEW: Export critical facts separately
        if self.report.critical_facts_summary:
            critical_path = os.path.join(self.output_dir, "critical_facts.json")
            with open(critical_path, "w", encoding="utf-8") as f:
                json.dump(self.report.critical_facts_summary, f, indent=2)
            outputs['critical_facts'] = critical_path
            print(f"[Saved] {critical_path}")
        
        return outputs


if __name__ == "__main__":
    pdf_path = r"data/raw_pdfs/check.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
    else:
        compressor = HierarchicalCompressor()
        report = compressor.compress_document(pdf_path)
        compressor.export()