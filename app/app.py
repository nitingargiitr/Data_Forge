import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

# FIX: Add parent directory to path for imports
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

st.set_page_config(
    page_title="Contextual Compression Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - OPTIMIZED AND CLEAN
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4dabf7;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #adb5bd;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chunk cards */
    .chunk-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Badges */
    .critical-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .header-badge {
        background-color: #4b8bff;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .standard-badge {
        background-color: #7f7f7f;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Better text area styling */
    .stTextArea textarea {
        background-color: #1e2530 !important;
        color: #e0e0e0 !important;
        border: 2px solid #4dabf7 !important;
        font-size: 1.05rem !important;
        line-height: 1.8 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Info boxes */
    .info-card {
        background-color: rgba(77, 171, 247, 0.1);
        border-left: 4px solid #4dabf7;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Level detail boxes */
    .level-box {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Session State
def init_session_state():
    defaults = {
        'report': None,
        'file_processed': False,
        'uploaded_file': None,
        'compression_config': {
            'min_words': 30,
            'max_words': 250,
            'overlap_words': 30,
            'doc_max_length': 300,
            'strategy': 'extractive'
        },
        'current_view': 'upload',
        'selected_chunk': 0,
        'selected_section': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def render_header():
    st.markdown('<div class="main-header">Contextual Compression Engine</div>', unsafe_allow_html=True)
    st.divider()

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Compression Settings")
        
        st.subheader("Chunking Parameters")
        min_words = st.slider(
            "Min words per chunk",
            10, 100,
            st.session_state.compression_config['min_words']
        )
        
        max_words = st.slider(
            "Max words per chunk",
            100, 500,
            st.session_state.compression_config['max_words']
        )
        
        overlap = st.slider(
            "Overlap words",
            0, 100,
            st.session_state.compression_config['overlap_words']
        )
        
        st.subheader("Summary Control")
        doc_max_length = st.slider(
            "Max document summary length (words)",
            100, 2000,
            st.session_state.compression_config['doc_max_length'],
            step=50
        )
        
        strategy = st.selectbox(
            "Summarization Strategy",
            ["hybrid", "abstractive", "extractive", "critical"],
            index=["extractive", "abstractive", "hybrid", "critical"].index(
                st.session_state.compression_config['strategy']
            )
        )
        
        st.session_state.compression_config = {
            'min_words': min_words,
            'max_words': max_words,
            'overlap_words': overlap,
            'doc_max_length': doc_max_length,
            'strategy': strategy
        }
        
        st.divider()
        
        with st.expander("Current Configuration"):
            st.json(st.session_state.compression_config)
        
        st.info("""
        **How it works:**
        1. Document is split into semantic chunks
        2. Each chunk is summarized individually
        3. Section summaries are created
        4. Final document summary is generated
        5. All critical facts are extracted
        """)
        
        if st.session_state.file_processed:
            st.divider()
            if st.button("Process New Document", use_container_width=True):
                st.session_state.file_processed = False
                st.session_state.report = None
                st.session_state.uploaded_file = None
                st.rerun()

def process_uploaded_file(uploaded_file, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process the uploaded PDF with given configuration"""
    
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        try:
            from hierarchical_compressor import HierarchicalCompressor
            from chunking.enhanced_chunker import EnhancedSmartChunker
            from summarizer.enhanced_ai_summarizer import SummaryStrategy
        except ImportError as e:
            st.error(f"Import error: {e}")
            st.error("Python path: " + str(sys.path))
            return None
        
        compressor = HierarchicalCompressor(output_dir="outputs")
        
        with st.spinner("Processing document..."):
            progress_bar = st.progress(0)
            
            progress_bar.progress(10)
            st.text("Loading PDF...")
            
            progress_bar.progress(25)
            st.text(f"Chunking (min={config['min_words']}, max={config['max_words']})...")
            
            progress_bar.progress(50)
            st.text("Summarizing chunks...")
            
            progress_bar.progress(75)
            st.text("Creating hierarchy...")
            
            report = compressor.compress_document(temp_path, config)
            
            progress_bar.progress(100)
            st.text("Complete!")
            
            return report.to_dict()
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def upload_section():
    st.header("Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"]
        )
        
        if uploaded_file:
            st.success(f"File selected: **{uploaded_file.name}**")
            st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")
            
            with st.expander("📋 Preview Configuration"):
                st.json(st.session_state.compression_config)
            
            if st.button("Start Compression", type="primary", use_container_width=True):
                with st.spinner("Processing... This may take a minute"):
                    report = process_uploaded_file(
                        uploaded_file, 
                        st.session_state.compression_config
                    )
                    
                    if report:
                        st.session_state.report = report
                        st.session_state.file_processed = True
                        st.session_state.uploaded_file = uploaded_file.name
                        st.success("Compression complete!")
                        st.balloons()
                        st.rerun()
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload** a PDF document
        2. **Adjust** settings in sidebar
        3. **Click** "Start Compression"
        4. **Explore** the results
       
        """)

def show_metrics(report: Dict[str, Any]):
    st.header("Compression Metrics")
    
    original_words = report['original_stats']['words']
    final_words = report['levels'][-1]['total_words']
    compression_ratio = final_words / original_words if original_words > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Original Words", f"{original_words:,}")
    
    with col2:
        st.metric("Final Words", f"{final_words:,}", f"-{original_words - final_words:,}")
    
    with col3:
        st.metric("Compression", f"{compression_ratio:.1%}")
    
    with col4:
        crit_rate = report['quality_metrics']['critical_preservation_rate']
        st.metric("Critical Preservation", f"{crit_rate:.0%}")
    
    with col5:
        info_loss = report['quality_metrics']['information_loss_score']
        st.metric("Info Loss Score", f"{info_loss:.2f}")
    
    st.divider()

# SIMPLIFIED: Hierarchy view with fallback to table
def show_hierarchy(report: Dict[str, Any]):
    st.header("Compression Hierarchy")
    
    # Show level statistics in a clean table format
    st.subheader("Hierarchy Levels Overview")
    
    levels_data = []
    for lvl in report['levels']:
        levels_data.append({
            "Level": lvl['level_name'].title(),
            "Items": lvl['item_count'],
            "Total Words": f"{lvl['total_words']:,}",
            "Compression": f"{lvl['compression_ratio']:.1%}"
        })
    
    df = pd.DataFrame(levels_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Detailed level information
    st.subheader("Detailed Level Information")
    
    col1, col2 = st.columns(2)
    
    for idx, lvl in enumerate(report['levels']):
        with (col1 if idx % 2 == 0 else col2):
            st.markdown(f"""
            <div class="level-box">
                <h4>🔹 {lvl['level_name'].title()} Level</h4>
                <p><b>Items:</b> {lvl['item_count']}<br>
                <b>Total Words:</b> {lvl['total_words']:,}<br>
                <b>Compression:</b> {lvl['compression_ratio']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show compression flow
    st.divider()
    st.subheader("Bar Graph")
    
    # Create simple bar chart for word count at each level
    level_names = [lvl['level_name'].title() for lvl in report['levels']]
    word_counts = [lvl['total_words'] for lvl in report['levels']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=level_names,
            y=word_counts,
            marker=dict(
                color=['#4dabf7', '#ffa94d', '#69db7c', '#ff8787'][:len(level_names)],
                line=dict(color='#ffffff', width=2)
            ),
            text=[f"{wc:,}" for wc in word_counts],
            textposition='outside',
            textfont=dict(size=14, color='#ffffff')
        )
    ])
    
    fig.update_layout(
        title="Word Count Reduction Across Levels",
        xaxis_title="Hierarchy Level",
        yaxis_title="Total Words",
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#ffffff'),
        showlegend=False,
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# FIXED: Single, clean summary display
def show_summary(report: Dict[str, Any]):
    st.header("Final Document Summary")
    
    doc_level = report['levels'][-1]
    
    if doc_level.get("items") and len(doc_level["items"]) > 0:
        doc_item = doc_level["items"][0]
        summary_text = doc_item.get("summary", "")
        word_count = len(summary_text.split())
        
        # Metrics row
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.metric("Current Length", f"{word_count} words")
        
        with col2:
            target = st.session_state.compression_config['doc_max_length']
            st.metric("Target Max", f"{target} words")
        
        st.divider()
        
        # SINGLE summary display - text area only
        st.markdown("###Summary Content")
        
        st.text_area(
            "Document Summary",
            summary_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
            key="summary_display"
        )
        
        # Action buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.download_button(
                "Download Summary",
                summary_text,
                f"summary_{st.session_state.uploaded_file or 'document'}.txt",
                "text/plain",
                use_container_width=True
            )
        
        with col_b:
            with st.expander("View Explainability Details"):
                explainability = doc_item.get("explainability", {})
                if explainability:
                    st.json(explainability)
                else:
                    st.info("No explainability data available")
        
    else:
        st.error("No summary available in the report")
        st.info("This might indicate an issue with document processing. Please try reprocessing the document.")

def show_chunk_explorer(report: Dict[str, Any]):
    st.header("Chunk Explorer")
    
    chunk_level = report['levels'][1] if len(report['levels']) > 1 else None
    
    if not chunk_level or "items" not in chunk_level:
        st.warning("No chunks available")
        return
    
    chunks = chunk_level["items"]
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        section_filter = st.selectbox(
            "Filter by Section",
            ["All"] + sorted(list(set(str(c.get("section_id", "unknown")) for c in chunks)))
        )
    
    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All", "critical", "header", "standard"]
        )
    
    with col3:
        page_filter = st.selectbox(
            "Filter by Page",
            ["All"] + sorted(list(set(str(c.get("page_number", 0)) for c in chunks)))
        )
    
    # Apply filters
    filtered_chunks = chunks
    if section_filter != "All":
        filtered_chunks = [c for c in filtered_chunks if str(c.get("section_id")) == section_filter]
    if type_filter != "All":
        filtered_chunks = [c for c in filtered_chunks if c.get("chunk_type") == type_filter]
    if page_filter != "All":
        filtered_chunks = [c for c in filtered_chunks if str(c.get("page_number", 0)) == page_filter]
    
    st.write(f"Showing **{len(filtered_chunks)}** of **{len(chunks)}** chunks")
    st.divider()
    
    # Display chunks
    for i, chunk in enumerate(filtered_chunks):
        chunk_type = chunk.get("chunk_type", "standard")
        
        if chunk_type == "critical":
            badge = '<span class="critical-badge">CRITICAL</span>'
        elif chunk_type == "header":
            badge = '<span class="header-badge">HEADER</span>'
        else:
            badge = '<span class="standard-badge">STANDARD</span>'
        
        with st.container():
            st.markdown(f"""
            <div class="chunk-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4>Chunk {chunk.get("chunk_id", i)} {badge}</h4>
                    <span style="color: #adb5bd;">Page {chunk.get("page_number", "N/A")} | 
                          Section {chunk.get("section_id", "N/A")}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Details", expanded=(i == 0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("Summary")
                    st.write(chunk.get("summary", "No summary"))
                    
                    st.subheader("Location")
                    col_loc1, col_loc2, col_loc3 = st.columns(3)
                    with col_loc1:
                        st.metric("Page", chunk.get("page_number", "N/A"))
                    with col_loc2:
                        st.metric("Section", chunk.get("section_id", "N/A"))
                    with col_loc3:
                        st.metric("Chunk ID", chunk.get("chunk_id", "N/A"))
                    
                    word_count = len(chunk.get("summary", "").split())
                    st.write(f"**Summary Words:** {word_count}")
                    
                    if "original_words" in chunk:
                        orig_words = chunk["original_words"]
                        st.write(f"**Original Words:** {orig_words}")
                        if orig_words > 0:
                            st.write(f"**Compression:** {(word_count/orig_words)*100:.1f}%")
                
                with col2:
                    st.subheader("Metadata")
                    st.write(f"**Confidence:** {chunk.get('confidence', 0):.2f}")
                    
                    explain = chunk.get("explainability", {})
                    st.write(f"**Priority:** {explain.get('preservation_priority', 'N/A')}")
                    st.write(f"**Role:** {explain.get('structural_role', 'N/A')}")
                    
                    if explain.get('critical_content_found'):
                        st.write("**Critical Items:**")
                        for item in explain['critical_content_found'][:3]:
                            st.write(f"• {item}")

def show_critical_facts(report: Dict[str, Any]):
    st.header("Critical Facts & Exceptions")
    
    facts = report.get("critical_facts_summary", [])
    
    if not facts:
        st.info("ℹNo critical facts detected in this document")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Critical Facts", len(facts))
    with col2:
        exceptions = sum(1 for f in facts if f.get("details", {}).get("has_exception"))
        st.metric("Exceptions", exceptions)
    with col3:
        risks = sum(1 for f in facts if f.get("details", {}).get("has_risk"))
        st.metric("Risks", risks)
    with col4:
        contradictions = sum(1 for f in facts if f.get("details", {}).get("has_contradiction"))
        st.metric("Contradictions", contradictions)
    
    st.divider()
    
    # Filter
    fact_type = st.radio(
        "Filter by type",
        ["All", "Exceptions", "Risks", "Contradictions"],
        horizontal=True
    )
    
    filtered_facts = facts
    if fact_type == "Exceptions":
        filtered_facts = [f for f in facts if f.get("details", {}).get("has_exception")]
    elif fact_type == "Risks":
        filtered_facts = [f for f in facts if f.get("details", {}).get("has_risk")]
    elif fact_type == "Contradictions":
        filtered_facts = [f for f in facts if f.get("details", {}).get("has_contradiction")]
    
    st.write(f"Showing **{len(filtered_facts)}** facts")
    st.divider()
    
    # Display facts
    for i, fact in enumerate(filtered_facts):
        details = fact.get("details", {})
        
        if details.get("has_contradiction"):
            icon = "⚠️"
            color = "#ff4b4b"
        elif details.get("has_risk"):
            icon = "🔴"
            color = "#ff6b6b"
        elif details.get("has_exception"):
            icon = "📋"
            color = "#4b8bff"
        else:
            icon = "📌"
            color = "#7f7f7f"
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 15px; margin: 15px 0; 
                        background-color: rgba(255,255,255,0.03); border-radius: 4px;">
                <h4>{icon} {fact.get('section', 'Unknown Section')} - Page {fact.get('page', 'N/A')}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Details", expanded=(i == 0)):
                st.write(fact.get("summary", "No summary"))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Flags")
                    st.write(f"• Exception: {'✅' if details.get('has_exception') else '❌'}")
                    st.write(f"• Risk: {'✅' if details.get('has_risk') else '❌'}")
                    st.write(f"• Contradiction: {'✅' if details.get('has_contradiction') else '❌'}")
                    st.write(f"• Numbers: {'✅' if details.get('has_numbers') else '❌'}")
                
                with col2:
                    st.subheader("Source")
                    st.write(f"**Chunk ID:** {fact.get('chunk_id', 'N/A')}")
                    st.write(f"**Page:** {fact.get('page', 'N/A')}")
                    st.write(f"**Section:** {fact.get('section', 'N/A')}")

def show_export(report: Dict[str, Any]):
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Full Report (JSON)")
        json_str = json.dumps(report, indent=2)
        st.download_button(
            "Download Full Report",
            json_str,
            "compression_report.json",
            "application/json",
            use_container_width=True,
            key="download_full"
        )
        
        with st.expander("Preview JSON"):
            preview_text = json_str[:2000] + "\n..." if len(json_str) > 2000 else json_str
            st.code(preview_text, language="json")
    
    with col2:
        st.subheader("Summary (TXT)")
        doc_level = report['levels'][-1]
        if doc_level.get("items"):
            summary_text = doc_level["items"][0].get("summary", "")
            st.download_button(
                "Download Summary",
                summary_text,
                "document_summary.txt",
                "text/plain",
                use_container_width=True,
                key="download_summary"
            )
        else:
            st.info("No summary available")
        
        st.subheader("Critical Facts (JSON)")
        facts = report.get("critical_facts_summary", [])
        if facts:
            facts_json = json.dumps(facts, indent=2)
            st.download_button(
                "Download Critical Facts",
                facts_json,
                "critical_facts.json",
                "application/json",
                use_container_width=True,
                key="download_facts"
            )
        else:
            st.info("No critical facts to export")
    
    st.divider()
    
    # Compression decisions
    st.subheader("Compression Decisions")
    
    decisions = report.get("compression_decisions", [])
    if decisions:
        for idx, decision in enumerate(decisions):
            with st.expander(f"Decision {idx + 1}: {decision.get('decision', 'Unknown')}", expanded=(idx == 0)):
                st.write(decision.get('rationale', 'No rationale provided'))
    else:
        st.info("No compression decisions recorded")

def main():
    render_header()
    render_sidebar()
    
    if not st.session_state.file_processed:
        upload_section()
    else:
        report = st.session_state.report
        
        tabs = st.tabs([
            "Metrics",
            "Hierarchy",
            "Summary",
            "Chunks",
            "Critical Facts",
            "Export"
        ])
        
        with tabs[0]:
            show_metrics(report)
        
        with tabs[1]:
            show_hierarchy(report)
        
        with tabs[2]:
            show_summary(report)
        
        with tabs[3]:
            show_chunk_explorer(report)
        
        with tabs[4]:
            show_critical_facts(report)
        
        with tabs[5]:
            show_export(report)

if __name__ == "__main__":
    main()