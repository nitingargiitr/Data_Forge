<div align="center">

# Contextual Compression Engine

### Hierarchical Document Summarization with Full Traceability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track 4 - DataForge E-Summit 2026**

[Live Demo](https://dataforge-contextual-search-engine.streamlit.app/)

</div>

---

## Overview

The **Contextual Compression Engine** is an advanced document processing system that transforms lengthy documents into hierarchical, traceable summaries while preserving decision-critical information. Unlike traditional one-shot summarizers, this system creates a 4-level compression hierarchy that enables users to explore content at any granularity—from high-level overviews to specific source locations.

### Key Capabilities

- **Hierarchical Compression**: Raw → Chunk → Section → Document levels
- **Critical Content Preservation**: 100% retention of exceptions, risks, and contradictions
- **Full Traceability**: Navigate from summary to source with page/section info
- **Per-Item Explainability**: Know why each element was included or removed
- **Configurable Control**: Adjust compression parameters in real-time
- **Interactive Visualization**: Explore document structure through Sunburst charts

---

## Features

### Hierarchical Summarization
```
Document (1,302 words)
    ↓
Chunks (19-27 summaries, 30-250 words each)
    ↓
Sections (5-15 summaries, aggregated by section)
    ↓
Final Summary (200-700 words, configurable)
```

### Critical Content Detection
Automatically identifies and preserves:
- **Exceptions**: "unless", "only if", "however"
- **Risks**: "WARNING", "ALERT", "CAUTION"
- **Contradictions**: "vs", "CONTRADICTION"
- **Numbers & Dates**: Quantities, thresholds, deadlines

### Explainability Framework
Every chunk includes:
- **Why included**: "Contains policy exceptions that modify standard rules"
- **What was removed**: ["Detailed examples", "Elaborative context"]
- **Removal reason**: "Using extractive summarization for reliable compression"
- **Confidence score**: 0.0-1.0 based on preservation quality

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/contextual-compression-engine.git
cd contextual-compression-engine

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit web interface
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

Alternatively, access the live deployed version at: [https://dataforge-contextual-search-engine.streamlit.app/](https://dataforge-contextual-search-engine.streamlit.app/)

### Basic Usage

1. **Upload**: Drag and drop a PDF document
2. **Configure**: Adjust compression settings in the sidebar (optional)
3. **Process**: Click "Start Compression"
4. **Explore**: Navigate through 6 tabs to explore results
5. **Export**: Download JSON reports or text summaries

---

## Example Output

### Compression Metrics
```
Original Words:        1,302
Final Words:             214
Compression Ratio:     16.4%
Critical Preservation:  100%
Information Loss:      0.25
Total Chunks:            27
Critical Facts:           9
```

### Sample Critical Fact
```json
{
  "section": "1.1",
  "page": 2,
  "type": "critical",
  "summary": "Passwords must be changed every 30 days unless...",
  "details": {
    "has_exception": true,
    "has_risk": false,
    "has_contradiction": false,
    "has_numbers": true
  },
  "chunk_id": 5
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: PDF Document                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │  Document Ingestion   │
           │  • Text extraction    │
           │  • Structure parsing  │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Semantic Chunking    │
           │  • Paragraph split    │
           │  • Header detection   │
           │  • Critical flags     │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Hierarchical Summary │
           │  • Chunk summaries    │
           │  • Section aggregate  │
           │  • Document final     │
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  Report Generation    │
           │  • Metrics calc       │
           │  • Critical extract   │
           │  • JSON export        │
           └───────────────────────┘
```

---

## Configuration

Adjust compression behavior through the sidebar:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Min Words** | 10-100 | 30 | Minimum chunk size |
| **Max Words** | 100-500 | 250 | Maximum chunk size |
| **Overlap** | 0-100 | 30 | Context preservation |
| **Max Length** | 100-1000 | 300 | Final summary limit |
| **Strategy** | Multiple | extractive | Summarization approach |

### Strategy Options

- **Extractive**: Sentence selection (fast, reliable)
- **Abstractive**: Transformer generation (detailed)
- **Hybrid**: Dynamic selection (balanced)
- **Critical**: Prioritizes exceptions/risks (compliance)

---

## Project Structure

```
contextual_compression/
├── app/
│   └── app.py                    # Streamlit web interface
├── chunking/
│   └── enhanced_chunker.py       # Semantic chunking engine
├── ingestion/
│   └── enhanced_pdf_loader.py    # PDF processing
├── summarizer/
│   └── enhanced_ai_summarizer.py # Summarization algorithms
├── data/
│   └── raw_pdfs/                 # Sample documents
├── outputs/                      # Generated reports
├── hierarchical_compressor.py    # Main orchestrator
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Technical Details

### Chunking Algorithm

1. **Paragraph Detection**: Split on double newlines
2. **Header Classification**: Regex patterns for section identification
3. **Section Tracking**: Propagate section IDs through document
4. **Critical Detection**: Pattern matching for exceptions/risks
5. **Overlap Management**: Preserve context between chunks

### Summarization Methods

**Extractive (Primary)**
```python
# TF-IDF based sentence scoring
# Critical content boost (3x multiplier)
# Sentence selection by score
# Original order restoration
```

**Critical-Preserving**
```python
# Higher compression ratio (40%)
# Post-process injection of critical sentences
# Length re-checking
```

### Quality Metrics

- **Compression Ratio**: Final words / Original words
- **Critical Preservation**: % of critical chunks preserved
- **Information Loss**: Composite score (0=perfect, 1=total loss)
  - 30% weight: Compression ratio
  - 40% weight: Average confidence
  - 30% weight: Critical preservation

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_chunker.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Coverage

- Document ingestion (PDF loading)
- Semantic chunking (boundary detection)
- Critical content detection (patterns)
- Summarization (compression quality)
- Hierarchical aggregation (section grouping)
- Report generation (JSON output)

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write tests for new features
- Update documentation

---

## Roadmap

### Phase 1: Core Enhancement (Current)
- [x] Hierarchical compression
- [x] Critical content preservation
- [x] Full traceability
- [x] Interactive visualization

### Phase 2: Advanced Features (Q2 2026)
- [ ] Multi-language support
- [ ] REST API
- [ ] Batch processing
- [ ] Advanced visualizations

### Phase 3: Enterprise Integration (Q3 2026)
- [ ] Cloud deployment
- [ ] SSO authentication
- [ ] Audit logging
- [ ] Custom model training

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **DataForge E-Summit 2026** for organizing the competition
- **Hugging Face** for transformer models
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations

---

## Contact

**Project Team**: DataForge E-Summit 2026 Participants

**Project Link**: [https://github.com/yourusername/contextual-compression-engine](https://github.com/yourusername/contextual-compression-engine)

**Live Demo**: [https://dataforge-contextual-search-engine.streamlit.app/](https://dataforge-contextual-search-engine.streamlit.app/)

---

<div align="center">

**Star this repository if you find it helpful!**

Made for the DataForge E-Summit 2026

</div>
