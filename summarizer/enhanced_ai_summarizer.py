import sys
import os
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class SummaryStrategy(Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    CRITICAL_PRESERVING = "critical"


@dataclass
class SummaryResult:
    summary_text: str
    original_words: int
    summary_words: int
    compression_ratio: float
    strategy: SummaryStrategy
    processing_time: float
    source_chunks: List[int] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    confidence: float = 0.0
    preserved_critical: List[str] = field(default_factory=list)
    level: str = "chunk"
    section_id: Optional[str] = None  # NEW: Track section
    explainability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            'summary_text': self.summary_text,
            'original_words': self.original_words,
            'summary_words': self.summary_words,
            'compression_ratio': self.compression_ratio,
            'strategy': self.strategy.value,
            'processing_time': self.processing_time,
            'source_chunks': self.source_chunks,
            'source_pages': self.source_pages,
            'confidence': self.confidence,
            'preserved_critical': self.preserved_critical,
            'level': self.level,
            'section_id': self.section_id,  # NEW
            'explainability': self.explainability
        }


class EnhancedAISummarizer:
    def __init__(self, model_name="t5-base", strategy=SummaryStrategy.HYBRID, device=None):
        self.model_name = model_name
        self.strategy = strategy
        self.device = device or "cpu"
        self.model = None
        self.tokenizer = None
        
        self.critical_patterns = {
            'number': re.compile(r'\\b\\d+(?:\\.\\d+)?\\s*(?:days?|hours?|minutes?|years?|\\$|€|£|%|GB|MB|TB)?\\b', re.IGNORECASE),
            'date': re.compile(r'\\b\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}\\b|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d{1,2},?\\s*\\d{4}', re.IGNORECASE),
            'exception': re.compile(r'\\b(?:EXCEPTION|unless|except|only if|however|but|although)\\b', re.IGNORECASE),
            'risk': re.compile(r'\\b(?:RISK|WARNING|ALERT|CAUTION|DANGER|MUST|REQUIRED|PROHIBITED)\\b', re.IGNORECASE),
            'threshold': re.compile(r'\\b(?:minimum|maximum|threshold|limit|at least|at most|no more than|no less than)\\b', re.IGNORECASE),
        }

    def _load_model(self):
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                
                print(f"[AI] Loading {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                print("[AI] Model ready.\\n")
            except ImportError:
                print("[AI] Warning: transformers not installed. Using extractive mode only.")
                self.strategy = SummaryStrategy.EXTRACTIVE

    def _identify_removed_content(self, original: str, summary: str) -> List[str]:
        """Identify what types of content were removed"""
        removed = []
        
        # Check for examples/cases removed
        if "example" in original.lower() and "example" not in summary.lower():
            removed.append("Detailed examples and case studies")
        
        # Check for elaborations removed  
        if len(original.split('.')) > len(summary.split('.')) * 2:
            removed.append("Elaborative explanations and context")
        
        # Check for redundant info
        if len(set(original.lower().split())) < len(original.split()) * 0.7:
            removed.append("Redundant and repetitive statements")
        
        # Check for narrative removed
        if len(original) > len(summary) * 3:
            removed.append("Narrative background and historical context")
        
        return removed if removed else ["Non-critical supporting details"]

    def summarize_chunk(self, chunk: Dict[str, Any], strategy=None):
        strategy = strategy or self.strategy
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', 0)
        page = chunk.get('page_number', 0)
        chunk_type = chunk.get('chunk_type', 'standard')
        section_id = chunk.get('section_id')  # NEW: Extract section
        
        word_count = len(text.split())
        
        # Generate explainability BEFORE compression
        explainability = self._generate_explainability(chunk, text)
        
        if word_count < 50:
            result = SummaryResult(
                summary_text=text,
                original_words=word_count,
                summary_words=word_count,
                compression_ratio=1.0,
                strategy=strategy,
                processing_time=0.0,
                source_chunks=[chunk_id],
                source_pages=[page],
                confidence=1.0,
                level="chunk",
                section_id=section_id,  # NEW
                explainability=explainability
            )
            return result
        
        start_time = time.time()
        critical_info = self._extract_critical_info(text)
        
        # FORCE extractive for short chunks to prevent expansion
        if word_count < 200:
            strategy = SummaryStrategy.EXTRACTIVE
            explainability['strategy_reason'] = "Short text - using extractive to prevent expansion"
        elif chunk_type == 'critical' or chunk.get('contains_exceptions'):
            strategy = SummaryStrategy.CRITICAL_PRESERVING
            explainability['strategy_reason'] = "Critical content detected - using preservation mode"
        else:
            strategy = SummaryStrategy.EXTRACTIVE
            explainability['strategy_reason'] = "Using extractive summarization for reliable compression"
        
        # Generate summary
        if strategy == SummaryStrategy.EXTRACTIVE:
            summary = self._extractive_summarize(text, ratio=0.35)
        elif strategy == SummaryStrategy.ABSTRACTIVE:
            summary = self._abstractive_summarize(text)
        elif strategy == SummaryStrategy.CRITICAL_PRESERVING:
            summary = self._critical_preserving_summarize(text, critical_info)
        else:
            summary = self._extractive_summarize(text, ratio=0.35)
        
        # SAFETY CHECK: Never allow expansion
        summary_words = len(summary.split())
        if summary_words >= word_count:
            summary = self._extractive_summarize(text, ratio=0.25)
            summary_words = len(summary.split())
            explainability['note'] = "Expansion prevented - forced stricter compression"
        
        processing_time = time.time() - start_time
        
        confidence = self._calculate_confidence(text, summary, critical_info)
        
        # Update explainability with post-compression info
        explainability['compression_applied'] = True
        words_removed = word_count - len(summary.split())
        explainability['words_removed'] = max(0, words_removed)
        explainability['removal_percentage'] = f"{(max(0, words_removed) / word_count * 100):.1f}%"
        explainability['content_removed'] = self._identify_removed_content(text, summary)

        result = SummaryResult(
            summary_text=summary,
            original_words=word_count,
            summary_words=len(summary.split()),
            compression_ratio=len(summary.split()) / word_count if word_count > 0 else 0,
            strategy=strategy,
            processing_time=processing_time,
            source_chunks=[chunk_id],
            source_pages=[page],
            confidence=confidence,
            preserved_critical=critical_info,
            level="chunk",
            section_id=section_id,  # NEW
            explainability=explainability
        )
        return result

    def _generate_explainability(self, chunk: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Generate per-item explainability for WHY this was included"""
        explain = {
            'inclusion_reason': '',
            'critical_content_found': [],
            'structural_role': '',
            'preservation_priority': 'standard'
        }
        
        chunk_type = chunk.get('chunk_type', 'standard')
        header_level = chunk.get('header_level', 0)
        section_id = chunk.get('section_id', 'none')
        
        # Determine structural role
        if header_level == 1:
            explain['structural_role'] = 'Section header'
        elif header_level == 2:
            explain['structural_role'] = 'Subsection header'
        else:
            explain['structural_role'] = 'Content body'
        
        # Determine inclusion reason based on content type
        if chunk_type == 'critical':
            explain['inclusion_reason'] = "Contains decision-critical information that cannot be lost"
            explain['preservation_priority'] = 'critical'
        elif chunk_type == 'header':
            explain['inclusion_reason'] = "Provides structural context and navigation reference"
            explain['preservation_priority'] = 'high'
        elif chunk.get('contains_numbers') and chunk.get('contains_dates'):
            explain['inclusion_reason'] = "Contains temporal and quantitative data relevant to decisions"
            explain['preservation_priority'] = 'high'
        elif chunk.get('contains_exceptions'):
            explain['inclusion_reason'] = "Contains policy exceptions that modify standard rules"
            explain['preservation_priority'] = 'critical'
        elif chunk.get('contains_risks'):
            explain['inclusion_reason'] = "Contains risk warnings or compliance alerts"
            explain['preservation_priority'] = 'critical'
        else:
            explain['inclusion_reason'] = "Standard content providing context and supporting information"
            explain['preservation_priority'] = 'standard'
        
        # List specific critical content found
        if chunk.get('contains_numbers'):
            numbers = re.findall(r'\\b\\d+(?:\\.\\d+)?\\s*(?:days?|hours?|years?|%)?\\b', text, re.IGNORECASE)
            explain['critical_content_found'].extend([f"Number: {n}" for n in numbers[:3]])
        
        if chunk.get('contains_dates'):
            dates = re.findall(r'\\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d{1,2},?\\s*\\d{4}\\b', text, re.IGNORECASE)
            explain['critical_content_found'].extend([f"Date: {d}" for d in dates[:2]])
        
        if chunk.get('contains_exceptions'):
            explain['critical_content_found'].append("Exception clause detected")
        
        if chunk.get('contains_risks'):
            explain['critical_content_found'].append("Risk alert detected")
        
        if chunk.get('contains_contradictions'):
            explain['critical_content_found'].append("Contradiction detected (requires human review)")
        
        # Section context
        if section_id and section_id != 'uncategorized':
            explain['section_context'] = f"Part of Section {section_id}"
        
        return explain

    def summarize_chunks(self, chunks: List[Dict[str, Any]], level="chunk"):
        results = []
        print(f"[AI] Summarizing {len(chunks)} chunks...")
        
        for chunk in chunks:
            result = self.summarize_chunk(chunk)
            result.level = level
            results.append(result)
            
            # Print explainability
            exp = result.explainability
            print(f"\\n  Chunk {chunk.get('chunk_id')} (Section {result.section_id}):")
            print(f"    Why included: {exp.get('inclusion_reason', 'N/A')}")
            print(f"    Priority: {exp.get('preservation_priority', 'N/A')}")
            print(f"    Critical found: {len(exp.get('critical_content_found', []))} items")
            print(f"    {result.original_words} → {result.summary_words} words ({result.compression_ratio:.1%})")
        
        print(f"\\n[AI] Done\\n")
        return results

    def summarize_summaries(self, summaries: List[SummaryResult], level="document"):
        """Create higher-level summary with aggregated explainability"""
        combined_text = " ".join([s.summary_text for s in summaries])
        
        print(f"[AI] Creating {level} summary from {len(summaries)} items...")
        
        # Aggregate explainability from children
        aggregated_explain = {
            'inclusion_reason': f"Aggregated summary of {len(summaries)} sub-sections",
            'structural_role': f'{level.title()} level overview',
            'preservation_priority': 'high',
            'child_summaries_included': [s.source_chunks[0] for s in summaries if s.source_chunks],
            'sections_covered': list(set(s.section_id for s in summaries if s.section_id)),  # NEW
            'total_critical_preserved': sum(len(s.preserved_critical) for s in summaries),
            'avg_confidence': sum(s.confidence for s in summaries) / len(summaries) if summaries else 0
        }
        
        # ALWAYS use extractive for higher levels to prevent expansion
        result = self.summarize_chunk({
            'text': combined_text,
            'chunk_id': -1,
            'page_number': summaries[0].source_pages[0] if summaries else 0,
            'chunk_type': 'standard'
        })
        
        # Ensure minimum length for document
        if level == "document" and result.summary_words < 100:
            fallback = self._extractive_summarize(combined_text, ratio=0.5)
            result.summary_text = fallback
            result.summary_words = len(fallback.split())
            result.compression_ratio = result.summary_words / result.original_words
            aggregated_explain['note'] = "Fallback to extractive due to length constraints"
        
        result.level = level
        result.source_chunks = [s.source_chunks[0] for s in summaries if s.source_chunks]
        result.source_pages = list(set([p for s in summaries for p in s.source_pages]))
        result.explainability = aggregated_explain
        
        print(f"[AI] ✓ {level.title()} summary: {result.original_words} → {result.summary_words} words\\n")
        
        return result

    def _extract_critical_info(self, text: str):
        critical = []
        for pattern_name, pattern in self.critical_patterns.items():
            matches = pattern.findall(text)
            if matches:
                for match in set(matches):
                    idx = text.find(str(match))
                    if idx != -1:
                        start = max(0, idx - 50)
                        end = min(len(text), idx + 50)
                        context = text[start:end].strip()
                        if context not in critical:
                            critical.append(context)
        return critical[:10]

    def _extractive_summarize(self, text: str, ratio=0.35):
        """FORCE compression - never expand"""
        sentences = re.split(r'(?<=[.!?])\\s+', text)
        
        if len(sentences) <= 3:
            return text
        
        target_sentences = max(1, int(len(sentences) * ratio))
        target_words = int(len(text.split()) * ratio)
        
        scores = []
        word_freq = {}
        words = re.findall(r'\\w+', text.lower())
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        for sent in sentences:
            score = 0
            sent_words = re.findall(r'\\w+', sent.lower())
            for word in sent_words:
                score += word_freq.get(word, 0)
            
            if any(p.search(sent) for p in self.critical_patterns.values()):
                score *= 3
            
            scores.append((sent, score, len(sent.split())))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        current_words = 0
        
        for sent, score, word_count in scores:
            if current_words + word_count <= target_words and len(selected) < target_sentences:
                selected.append(sent)
                current_words += word_count
        
        selected.sort(key=lambda s: text.find(s))
        
        summary = " ".join(selected)
        
        if len(summary.split()) >= len(text.split()):
            forced_count = max(1, int(len(sentences) * 0.25))
            forced = [s[0] for s in scores[:forced_count]]
            forced.sort(key=lambda s: text.find(s))
            summary = " ".join(forced)
        
        return summary

    def _abstractive_summarize(self, text: str):
        self._load_model()
        
        if self.model is None:
            return self._extractive_summarize(text)
        
        try:
            import torch
            
            prompt = "summarize: " + text[:1500]
            
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    min_new_tokens=20,
                    num_beams=4,
                    length_penalty=0.5,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"[AI] Abstractive failed: {e}")
            return self._extractive_summarize(text)

    def _critical_preserving_summarize(self, text: str, critical_info: List[str]):
        summary = self._extractive_summarize(text, ratio=0.4)
        
        for info in critical_info[:3]:
            if info not in summary and len(info) > 10:
                summary += " " + info
        
        if len(summary.split()) > len(text.split()) * 0.8:
            summary = self._extractive_summarize(text, ratio=0.3)
        
        return summary

    def _hybrid_summarize(self, text: str, critical_info: List[str]):
        return self._extractive_summarize(text, ratio=0.35)

    def _inject_critical_info(self, summary: str, critical_info: List[str]):
        missing = []
        for info in critical_info[:2]:
            key_terms = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b|\\b[A-Z]{2,}\\b', info)
            if key_terms and not any(term in summary for term in key_terms[:2]):
                missing.append(info)
        
        if missing and len(summary) < 400:
            summary += "\\nKey: " + "; ".join(missing[:2])
        
        return summary

    def _calculate_confidence(self, original: str, summary: str, critical_info: List[str]):
        confidence = 0.5
        
        orig_words = len(original.split())
        sum_words = len(summary.split())
        ratio = sum_words / orig_words if orig_words > 0 else 0
        
        if 0.15 <= ratio <= 0.4:
            confidence += 0.3
        elif ratio > 0.5:
            confidence -= 0.2
        elif ratio < 0.1:
            confidence += 0.1
        
        preserved = 0
        for info in critical_info:
            key_terms = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', info)
            if any(term in summary for term in key_terms):
                preserved += 1
        
        if critical_info:
            preservation_rate = preserved / len(critical_info)
            confidence += preservation_rate * 0.3
        
        return max(0.0, min(1.0, confidence))


if __name__ == "__main__":
    print("Enhanced AI Summarizer with Explainability - FIXED")