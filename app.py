import streamlit as st
import pandas as pd

from nlp_engine import (
    discover_topics, extract_entities, extract_pdf, keyword_scores, most_common_terms,
    retrieve, sentences, textrank_summary, tokenize, stem_tokens, pos_tag_text,
    classify_papers, compare_papers, detect_research_gaps, extract_research_signals,
    research_analysis,
)

st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ---------- ResearchMate visual system ---------- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --navy:#102a43; --blue:#1d4ed8; --cyan:#0ea5e9; --ink:#18324a;
  --muted:#66788a; --paper:#ffffff; --soft:#f5f8fc; --line:#dce6f0;
  --accent:#6d5dfc; --green:#16a34a;
}

.stApp { background:linear-gradient(180deg,#f7faff 0%,#ffffff 42%,#f6f9fc 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(247,250,255,.82); }
.block-container { max-width:1240px; padding-top:1.2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--navy) !important; letter-spacing:-.025em; }

/* top navigation */
.rm-nav { display:flex; align-items:center; justify-content:space-between; padding:.45rem 0 1.1rem; }
.rm-logo { font-family:'Manrope',sans-serif; font-size:1.25rem; font-weight:800; color:var(--navy); letter-spacing:-.03em; }
.rm-logo span { color:var(--accent); }
.rm-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.38rem .75rem; border:1px solid var(--line); background:#fff; border-radius:999px; color:#52677b; font-size:.78rem; font-weight:600; }

/* hero */
.rm-hero { position:relative; overflow:hidden; min-height:360px; border-radius:30px; margin:.3rem 0 1.4rem;
  background-image:linear-gradient(90deg,rgba(8,24,45,.96) 0%,rgba(10,35,62,.86) 48%,rgba(10,35,62,.30) 100%),
  url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=85');
  background-size:cover; background-position:center; box-shadow:0 24px 60px rgba(24,50,74,.14); }
.rm-hero-content { max-width:700px; padding:3.5rem 3.3rem; color:#fff; }
.rm-eyebrow { display:inline-block; padding:.4rem .72rem; border:1px solid rgba(255,255,255,.22); background:rgba(255,255,255,.10); border-radius:999px; font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.rm-hero h1 { color:#fff !important; font-size:clamp(2.25rem,5vw,4.1rem); line-height:1.02; margin:.9rem 0 .8rem; }
.rm-hero p { color:#dce8f5; font-size:1.05rem; line-height:1.65; max-width:620px; margin:0 0 1.25rem; }
.rm-stats { display:flex; gap:.65rem; flex-wrap:wrap; }
.rm-stat { padding:.55rem .75rem; border-radius:12px; background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.14); color:#eef6ff; font-size:.78rem; }

/* capability cards */
.rm-section-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:#7890a5; font-weight:800; margin:1.4rem 0 .65rem; }
.rm-card { min-height:126px; padding:1rem 1.05rem; border:1px solid var(--line); border-radius:18px; background:rgba(255,255,255,.92); box-shadow:0 8px 28px rgba(24,50,74,.055); }
.rm-icon { width:38px; height:38px; display:grid; place-items:center; border-radius:12px; background:#eef2ff; font-size:1.15rem; margin-bottom:.65rem; }
.rm-card strong { display:block; color:var(--navy); font-family:'Manrope',sans-serif; font-size:.92rem; margin-bottom:.28rem; }
.rm-card span { color:var(--muted); font-size:.78rem; line-height:1.45; }

/* uploader */
.rm-upload { padding:1.25rem; border:1px solid #d7e3ee; border-radius:22px; background:linear-gradient(135deg,#ffffff,#f7fbff); box-shadow:0 10px 32px rgba(24,50,74,.06); }
[data-testid="stFileUploader"] section { background:#fff!important; border:1.5px dashed #9eb5c9!important; border-radius:16px!important; min-height:150px; }
[data-testid="stFileUploader"] section:hover { border-color:var(--accent)!important; background:#fbfcff!important; }
[data-testid="stFileUploader"] button { border-radius:10px!important; border:1px solid #cdd9e5!important; background:#fff!important; }


/* ---------- homepage expansion ---------- */
.rm-home-intro { display:flex; justify-content:space-between; align-items:end; gap:1rem; margin:1.8rem 0 .85rem; }
.rm-home-intro h2 { margin:0; font-size:1.55rem; }
.rm-home-intro p { margin:.25rem 0 0; color:var(--muted); font-size:.88rem; max-width:620px; }
.rm-cap-grid .rm-card { min-height:150px; transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.rm-cap-grid .rm-card:hover { transform:translateY(-4px); box-shadow:0 16px 34px rgba(24,50,74,.10); border-color:#c8d7e6; }
.rm-card .rm-tag { display:inline-block; margin-top:.7rem; padding:.24rem .5rem; border-radius:999px; background:#f1f5ff; color:#4b5fc0; font-size:.68rem; font-weight:700; }
.rm-how { margin:1.35rem 0; padding:1.35rem; border-radius:24px; background:#0d2238; color:#fff; box-shadow:0 18px 42px rgba(13,34,56,.14); }
.rm-how h3 { color:#fff !important; margin:0 0 .2rem; font-size:1.15rem; }
.rm-how p { color:#b9cbe0; font-size:.82rem; margin:0 0 1rem; }
.rm-step { min-height:120px; padding:1rem; border:1px solid rgba(255,255,255,.10); border-radius:16px; background:rgba(255,255,255,.055); }
.rm-step-num { font-family:'Manrope',sans-serif; font-size:.72rem; font-weight:800; color:#9fb6cf; letter-spacing:.08em; }
.rm-step-icon { font-size:1.35rem; margin:.35rem 0 .3rem; }
.rm-step strong { display:block; font-family:'Manrope',sans-serif; font-size:.9rem; }
.rm-step span { display:block; margin-top:.25rem; color:#b9cbe0; font-size:.74rem; line-height:1.45; }
.rm-flow { margin:1.25rem 0 1.5rem; padding:1rem 1.1rem; border:1px solid var(--line); border-radius:20px; background:linear-gradient(135deg,#fff,#f7faff); }
.rm-flow-title { font-size:.74rem; text-transform:uppercase; letter-spacing:.11em; font-weight:800; color:#7890a5; margin-bottom:.75rem; }
.rm-flow-row { display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; }
.rm-flow-node { padding:.52rem .68rem; border-radius:11px; background:#eef3ff; border:1px solid #dce4ff; color:#354aa1; font-size:.72rem; font-weight:700; }
.rm-flow-arrow { color:#9aaabd; font-weight:800; }
.rm-feature-band { display:grid; grid-template-columns:1.15fr .85fr; gap:1rem; margin:1.35rem 0; }
.rm-feature-copy { padding:1.45rem; border:1px solid var(--line); border-radius:22px; background:#fff; box-shadow:0 10px 30px rgba(24,50,74,.05); }
.rm-feature-copy h3 { margin:.2rem 0 .5rem; font-size:1.35rem; }
.rm-feature-copy p { color:var(--muted); font-size:.84rem; line-height:1.65; }
.rm-check { margin:.45rem 0; color:#42566b; font-size:.8rem; }
.rm-feature-image { min-height:245px; border-radius:22px; overflow:hidden; background-image:linear-gradient(135deg,rgba(13,34,56,.15),rgba(13,34,56,.45)),url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=1200&q=85'); background-size:cover; background-position:center; box-shadow:0 10px 30px rgba(24,50,74,.08); }
@media (max-width: 800px) { .rm-feature-band { grid-template-columns:1fr; } .rm-home-intro { display:block; } }

/* results */
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.9rem; box-shadow:0 6px 20px rgba(24,50,74,.04); }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; padding:.4rem; background:#eef3f8; border-radius:15px; overflow-x:auto; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:.48rem .82rem; color:#53697d; font-weight:600; }
.stTabs [aria-selected="true"] { background:#fff; color:#3347a8; box-shadow:0 2px 10px rgba(24,50,74,.08); }
.stButton>button { background:var(--accent); border:0; border-radius:10px; color:#fff; font-weight:700; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }

.rm-footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:#8293a3; font-size:.75rem; text-align:center; }

/* persistent workspace navigation */
div[role='radiogroup'] { gap:.35rem; padding:.4rem; background:#eef3f8; border-radius:15px; overflow-x:auto; }
div[role='radiogroup'] label { background:transparent; border-radius:10px; padding:.15rem .25rem; }
div[role='radiogroup'] label:has(input:checked) { background:#fff; box-shadow:0 2px 10px rgba(24,50,74,.08); }
div[role='radiogroup'] label p { color:#53697d; font-weight:600; font-size:.78rem; }



/* ---------- Keywords showcase ---------- */
.rm-keyword-hero {
  padding:1.35rem 1.45rem; border-radius:24px; margin:.25rem 0 1rem;
  background:linear-gradient(135deg,#101f3d 0%,#243f7d 62%,#635bd8 100%);
  color:#fff; box-shadow:0 18px 42px rgba(24,50,74,.13);
}
.rm-keyword-kicker { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; color:#b8c8f4; }
.rm-keyword-hero h2 { color:#fff !important; margin:.25rem 0 .35rem; font-size:1.75rem; }
.rm-keyword-hero p { color:#d7e0f3; margin:0; max-width:760px; line-height:1.55; font-size:.86rem; }
.rm-kpi { padding:1rem 1.05rem; border:1px solid var(--line); border-radius:17px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); min-height:86px; }
.rm-kpi-label { color:#7890a5; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
.rm-kpi-value { color:var(--navy); font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; margin-top:.2rem; }
.rm-kpi-note { color:#74879a; font-size:.7rem; margin-top:.12rem; }
.rm-keyword-card { padding:.95rem 1rem; border:1px solid var(--line); border-radius:17px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); min-height:112px; margin-bottom:.7rem; transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
.rm-keyword-card:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(24,50,74,.09); border-color:#c8d7e6; }
.rm-keyword-rank { color:#7f91a4; font-size:.68rem; font-weight:800; letter-spacing:.08em; }
.rm-keyword-name { color:var(--navy); font-family:'Manrope',sans-serif; font-size:1rem; font-weight:800; margin:.22rem 0 .42rem; text-transform:capitalize; }
.rm-keyword-score { display:flex; justify-content:space-between; align-items:center; color:#687b8e; font-size:.68rem; font-weight:700; }
.rm-keyword-bar { height:7px; margin-top:.42rem; border-radius:999px; background:#edf1f5; overflow:hidden; }
.rm-keyword-bar span { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,#6d5dfc,#0ea5e9); }
.rm-keyword-section { margin:1.25rem 0 .65rem; font-family:'Manrope',sans-serif; color:var(--navy); font-size:1.05rem; font-weight:800; }
.rm-keyword-note { padding:.8rem 1rem; border-radius:14px; background:#f5f8fc; border:1px solid #e1e8ef; color:#66788a; font-size:.76rem; line-height:1.5; margin-bottom:1rem; }
@media (max-width: 700px) { .rm-keyword-card { min-height:100px; } }

/* ---------- Ask the Papers showcase ---------- */
.rm-ask-hero {
  padding:1.35rem 1.45rem; border-radius:24px; margin:.25rem 0 1rem;
  background:linear-gradient(135deg,#0d2238 0%,#173d63 65%,#314a91 100%);
  color:#fff; box-shadow:0 18px 42px rgba(13,34,56,.13);
}
.rm-ask-kicker { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; color:#a9c4df; }
.rm-ask-hero h2 { color:#fff !important; margin:.25rem 0 .35rem; font-size:1.75rem; }
.rm-ask-hero p { color:#c9d9e8; margin:0; max-width:760px; line-height:1.55; font-size:.86rem; }
.rm-ask-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.7rem; margin:1rem 0; }
.rm-ask-tip { padding:.85rem 1rem; border:1px solid var(--line); border-radius:16px; background:#fff; color:#4e6275; font-size:.78rem; line-height:1.45; box-shadow:0 7px 22px rgba(24,50,74,.045); }
.rm-ask-tip b { color:var(--navy); display:block; margin-bottom:.2rem; }
.rm-evidence { padding:1.05rem 1.1rem; border:1px solid var(--line); border-radius:18px; background:#fff; margin:.65rem 0; box-shadow:0 8px 24px rgba(24,50,74,.055); }
.rm-evidence-meta { display:flex; gap:.45rem; flex-wrap:wrap; align-items:center; margin-bottom:.55rem; }
.rm-badge { display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#eef3ff; color:#4054ad; font-size:.68rem; font-weight:800; }
.rm-badge.source { background:#edf8f1; color:#26734a; }
.rm-evidence-text { color:#263d52; font-size:.88rem; line-height:1.65; }
.rm-score { margin-top:.7rem; height:6px; background:#edf1f5; border-radius:999px; overflow:hidden; }
.rm-score span { display:block; height:100%; background:linear-gradient(90deg,#6d5dfc,#0ea5e9); border-radius:999px; }
.rm-verify { padding:.8rem 1rem; margin:1rem 0; border-radius:14px; background:#f7f9fc; border:1px solid #e2e9f0; color:#66788a; font-size:.76rem; }
@media (max-width: 700px) { .rm-ask-grid { grid-template-columns:1fr; } }


/* ---------- v7 analysis dashboards ---------- */
.rm-dashboard-hero { padding:1.25rem 1.4rem; border-radius:23px; margin:.2rem 0 1rem; background:linear-gradient(135deg,#102a43,#214d73 58%,#665dd7); color:#fff; box-shadow:0 16px 38px rgba(16,42,67,.12); }
.rm-dashboard-kicker { font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; color:#bdd0e6; }
.rm-dashboard-hero h2 { color:#fff !important; margin:.25rem 0 .3rem; font-size:1.7rem; }
.rm-dashboard-hero p { color:#d9e6f2; margin:0; max-width:800px; font-size:.84rem; line-height:1.55; }
.rm-panel { padding:1rem 1.05rem; border:1px solid var(--line); border-radius:18px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); margin-bottom:.8rem; }
.rm-panel-title { color:var(--navy); font-family:'Manrope',sans-serif; font-weight:800; font-size:1rem; margin-bottom:.3rem; }
.rm-panel-sub { color:var(--muted); font-size:.74rem; line-height:1.5; }
.rm-chip { display:inline-block; margin:.25rem .22rem .25rem 0; padding:.35rem .58rem; border-radius:999px; background:#eef3ff; border:1px solid #dce4ff; color:#3d51a6; font-size:.72rem; font-weight:700; }
.rm-topic-card { min-height:190px; padding:1.05rem; border:1px solid var(--line); border-radius:19px; background:#fff; box-shadow:0 8px 24px rgba(24,50,74,.055); margin-bottom:.8rem; }
.rm-topic-num { color:#7a8da0; font-size:.67rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.rm-topic-name { color:var(--navy); font-family:'Manrope',sans-serif; font-weight:800; font-size:1.05rem; margin:.25rem 0 .65rem; }
.rm-entity-card { min-height:145px; padding:1rem; border:1px solid var(--line); border-radius:18px; background:linear-gradient(135deg,#fff,#f8fbff); box-shadow:0 7px 22px rgba(24,50,74,.045); }
.rm-entity-count { color:var(--navy); font-family:'Manrope',sans-serif; font-size:1.5rem; font-weight:800; }
.rm-summary-card { padding:1.05rem 1.1rem; border:1px solid var(--line); border-radius:18px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); margin:.65rem 0; }
.rm-summary-index { display:inline-block; width:28px; height:28px; border-radius:9px; background:#eef3ff; color:#4054ad; text-align:center; line-height:28px; font-weight:800; font-size:.72rem; margin-right:.5rem; }
.rm-summary-text { color:#263d52; font-size:.88rem; line-height:1.65; margin-top:.55rem; }
.rm-gap-card { padding:1.05rem 1.1rem; border:1px solid #e1e7ef; border-radius:18px; background:#fff; box-shadow:0 8px 24px rgba(24,50,74,.05); margin:.7rem 0; }
.rm-gap-title { color:var(--navy); font-family:'Manrope',sans-serif; font-size:1rem; font-weight:800; }
.rm-gap-evidence { margin-top:.65rem; padding:.7rem .8rem; border-radius:12px; background:#f6f8fb; color:#687b8e; font-size:.75rem; line-height:1.5; }
.rm-compare-card { padding:1rem; border:1px solid var(--line); border-radius:18px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); }
.rm-compare-title { color:var(--navy); font-family:'Manrope',sans-serif; font-weight:800; font-size:.92rem; margin-bottom:.55rem; }
.rm-process-box { padding:1rem; border:1px solid var(--line); border-radius:18px; background:#fff; box-shadow:0 7px 22px rgba(24,50,74,.045); }
.rm-process-step { padding:.72rem .8rem; margin:.4rem 0; border-radius:12px; background:#f7f9fc; border:1px solid #e5ebf1; color:#4d6275; font-size:.77rem; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rm-nav">
  <div class="rm-logo">Research<span>Mate</span></div>
  <div class="rm-pill">✦ NLP Research Intelligence</div>
</div>
<div class="rm-hero">
  <div class="rm-hero-content">
    <div class="rm-eyebrow">AI-assisted research workspace</div>
    <h1>Turn research papers into structured knowledge.</h1>
    <p>Upload your papers and explore keywords, entities, topics, summaries, similarities, research signals and cited evidence from one focused workspace.</p>
    <div class="rm-stats"><span class="rm-stat">📄 Page-aware PDFs</span><span class="rm-stat">🧠 NLP analysis</span><span class="rm-stat">🔎 Evidence retrieval</span></div>
  </div>
</div>
<div class="rm-home-intro">
  <div><div class="rm-section-label" style="margin:0 0 .25rem">Research intelligence</div><h2>Everything you need to explore a paper</h2></div>
  <p>Move from raw PDF text to structured evidence, themes and research signals without leaving the workspace.</p>
</div>
""", unsafe_allow_html=True)

capabilities = st.columns(3)
for column, icon, title, description, tag in [
    (capabilities[0], "🔑", "Keywords", "Find distinctive terms and keyphrases with TF-IDF scoring.", "TF-IDF"),
    (capabilities[1], "🧬", "Entities", "Surface datasets, models, metrics and institutions from research text.", "Research entities"),
    (capabilities[2], "🧠", "Topics", "Discover recurring themes across uploaded papers with NMF.", "NMF topics"),
    (capabilities[0], "📝", "Summaries", "Rank original sentences so important findings remain traceable.", "TextRank"),
    (capabilities[1], "🔗", "Compare papers", "Compare pages, keywords, datasets, models and evaluation metrics.", "Multi-paper"),
    (capabilities[2], "💬", "Ask the papers", "Retrieve relevant excerpts with paper and page metadata.", "Evidence-first"),
]:
    with column:
        st.markdown(f"<div class='rm-card'><div class='rm-icon'>{icon}</div><strong>{title}</strong><span>{description}</span><div class='rm-tag'>{tag}</div></div>", unsafe_allow_html=True)

st.markdown("""
<div class="rm-how">
  <h3>From paper to insight</h3>
  <p>A simple research workflow powered by the NLP modules already inside ResearchMate.</p>
  <div class="rm-step-grid">
""", unsafe_allow_html=True)
steps = st.columns(3)
for col, num, icon, title, desc in [
    (steps[0], "01", "📄", "Upload", "Add one or more research-paper PDFs and keep their page context."),
    (steps[1], "02", "⚙️", "Analyze", "Preprocess text, extract features, discover topics and rank evidence."),
    (steps[2], "03", "💡", "Discover", "Explore summaries, comparisons, research signals and cited answers."),
]:
    with col:
        st.markdown(f"<div class='rm-step'><div class='rm-step-num'>{num}</div><div class='rm-step-icon'>{icon}</div><strong>{title}</strong><span>{desc}</span></div>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("""
<div class="rm-flow">
  <div class="rm-flow-title">NLP analysis pipeline</div>
  <div class="rm-flow-row">
    <span class="rm-flow-node">PDF</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Extraction</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Preprocessing</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Features</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Topics</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Summary</span><span class="rm-flow-arrow">→</span>
    <span class="rm-flow-node">Evidence</span>
  </div>
</div>
<div class="rm-feature-band">
  <div class="rm-feature-copy">
    <div class="rm-section-label" style="margin:0 0 .3rem">Built for research work</div>
    <h3>Keep the evidence connected to the source.</h3>
    <p>ResearchMate keeps page-level metadata with extracted text, so analysis can lead you back to the original paper rather than presenting unsupported generated claims.</p>
    <div class="rm-check">✓ Page-aware PDF extraction</div>
    <div class="rm-check">✓ Original-sentence summaries</div>
    <div class="rm-check">✓ Paper and page evidence for retrieval</div>
  </div>
  <div class="rm-feature-image"></div>
</div>
<div class="rm-section-label">Start a research session</div>
<div class="rm-upload">
""", unsafe_allow_html=True)
_, upload_center, _ = st.columns([1, 2.6, 1])
with upload_center:
    uploads = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True,
                               label_visibility="collapsed")

if not uploads:
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='rm-footer'>ResearchMate · Evidence-first NLP for research paper exploration</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)

chunks = []
for upload in uploads:
    try:
        chunks.extend(extract_pdf(upload))
    except Exception as error:
        st.warning(f"Could not read {upload.name}: {error}")

if not chunks:
    st.error("No readable research-paper text was found in the uploaded files.")
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)
st.success(f"Analysis complete: {len(uploads)} paper(s) and {len(chunks)} pages indexed.")

# Persistent workspace navigation.
# st.tabs() resets to the first tab after every widget interaction. A horizontal
# radio navigation keeps the selected workspace active across Streamlit reruns.
# Two-level workspace: start with a researcher-friendly overview, then open the technical NLP tools.
workspace_mode = st.radio(
    "Research workspace",
    ["🔎 Research Assistant", "🧠 NLP Analysis"],
    key="workspace_mode",
    horizontal=True,
    label_visibility="collapsed",
)

if "active_section" not in st.session_state:
    st.session_state.active_section = "Preprocessing"

if workspace_mode == "🔎 Research Assistant":
    # Clean researcher-facing reading experience. Technical retrieval details stay in NLP Analysis.
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">RESEARCH ASSISTANT</div>
      <h2>Understand your paper in minutes.</h2>
      <p>ResearchMate first gives you a clear reading brief: what the paper studies, how the research was done, what was found, and what the authors say should happen next.</p>
    </div>
    """, unsafe_allow_html=True)

    paper_names = list(dict.fromkeys(c.paper for c in chunks))
    selected_paper = st.selectbox(
        "Choose a paper to understand",
        paper_names,
        key="assistant_paper",
        label_visibility="visible",
    )
    paper_chunks = [c for c in chunks if c.paper == selected_paper]
    paper_text = " ".join(c.text for c in paper_chunks)
    paper_words = len(tokenize(paper_text))
    paper_entities = extract_entities(paper_chunks)
    paper_keywords = keyword_scores(paper_chunks, limit=8)
    paper_summary = textrank_summary(paper_chunks, sentence_limit=5)
    paper_classification, _ = classify_papers(paper_chunks)
    domain = paper_classification[0]["Predicted domain"] if paper_classification else "Not detected"

    # Step 1: quick orientation
    st.markdown("### 1 · Paper at a glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pages", len(paper_chunks))
    k2.metric("Words", f"{paper_words:,}")
    k3.metric("Domain", domain)
    k4.metric("Entities", sum(len(v) for v in paper_entities.values()))

    top_terms = ", ".join(k for k,_ in paper_keywords[:6]) or "Not detected"
    datasets = ", ".join(paper_entities.get("Datasets", [])[:5]) or "Not detected"
    models = ", ".join(paper_entities.get("Models / methods", [])[:5]) or "Not detected"
    metrics = ", ".join(paper_entities.get("Metrics", [])[:5]) or "Not detected"
    glance = st.columns(2)
    with glance[0]:
        st.markdown(f"<div class='rm-panel'><div class='rm-panel-title'>Main research terms</div><div class='rm-panel-sub'>{top_terms}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rm-panel'><div class='rm-panel-title'>Datasets</div><div class='rm-panel-sub'>{datasets}</div></div>", unsafe_allow_html=True)
    with glance[1]:
        st.markdown(f"<div class='rm-panel'><div class='rm-panel-title'>Methods / models</div><div class='rm-panel-sub'>{models}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rm-panel'><div class='rm-panel-title'>Evaluation metrics</div><div class='rm-panel-sub'>{metrics}</div></div>", unsafe_allow_html=True)

    # Step 2: readable summary first
    st.markdown("### 2 · What is this paper about?")
    st.caption("ResearchMate selects important original sentences from the paper. Page numbers let you verify the summary in the source.")
    if paper_summary:
        summary_text = " ".join(sentence for sentence,_,_ in paper_summary)
        st.markdown(f"<div class='rm-summary-card'><div class='rm-summary-text'>{summary_text}</div></div>", unsafe_allow_html=True)
        with st.expander("Show the original summary sentences and page references"):
            for i, (sentence, _, page) in enumerate(paper_summary, 1):
                st.markdown(f"**{i}.** {sentence}")
                st.caption(f"Source: Page {page}")
    else:
        st.info("A summary could not be extracted from this paper.")

    # Step 3: actual research analysis — compact interpretations, not just copied paragraphs
    st.markdown("### 3 · ResearchMate analysis")
    analysis = research_analysis(paper_chunks)

    st.markdown("#### 🧠 Executive insight")
    st.markdown(f"<div class='rm-summary-card'><div class='rm-summary-text'>{analysis['executive_insight']}</div></div>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### ⚙️ Methodology analysis")
        if analysis["method_roles"]:
            for item in analysis["method_roles"]:
                st.markdown(f"**{item['role']}**  \\n{item['evidence']}")
        else:
            st.info("No clear methodology roles were detected from the available text.")
    with a2:
        st.markdown("#### 🧪 Experiment analysis")
        if analysis["experiment"]:
            st.dataframe(pd.DataFrame(analysis["experiment"]), hide_index=True, use_container_width=True)
        else:
            st.info("No explicit train/validation/test quantities were detected.")

    st.markdown("#### 📊 Result analysis")
    if analysis["comparison"]:
        result_df = pd.DataFrame([{
            "Model": r["Model"],
            "Primary reported score": r["Primary score"],
        } for r in analysis["comparison"]])
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        q = analysis["quantitative_insight"]
        if q:
            st.success(
                f"ResearchMate comparison: {q['best_model']} reports the highest detected primary score "
                f"({q['best_score']:.2f}), {q['difference']:.2f} points above {q['baseline_model']} ({q['baseline_score']:.2f})."
            )
        with st.expander("View the original result evidence"):
            for r in analysis["comparison"]:
                st.markdown(f"**{r['Model']}** — {r['Evidence']}")
    else:
        st.info("A comparable numeric result was not detected. ResearchMate will not invent a performance comparison.")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### 💡 Key research insights")
        insights = []
        if analysis["method_roles"]:
            insights.append("The paper combines multiple processing/model stages rather than relying on a single NLP operation.")
        if analysis["quantitative_insight"]:
            q = analysis["quantitative_insight"]
            insights.append(f"The detected comparison shows {q['best_model']} with the highest reported primary score.")
            insights.append(f"The highest-to-lowest detected score difference is {q['difference']:.2f} points in this paper's reported experiment.")
        if datasets:
            insights.append(f"The reported evaluation uses: {datasets}.")
        for item in insights[:4]:
            st.markdown(f"• {item}")
        if not insights:
            st.info("Not enough structured evidence was detected for additional insights.")
    with b2:
        st.markdown("#### ⚠️ Critical analysis")
        if analysis["limitations"]:
            for row in analysis["limitations"][:4]:
                st.markdown(f"• **Page {row['Page']}:** {row['Evidence']}")
        else:
            st.info("No explicit limitation signal was detected. This does not mean the paper has no limitations.")

    st.markdown("#### 🚀 Future research analysis")
    if analysis["future"]:
        for row in analysis["future"][:4]:
            st.markdown(f"• **Page {row['Page']}:** {row['Evidence']}")
    else:
        st.info("No explicit future-work signal was detected.")

    with st.expander("📚 View source evidence used for the analysis"):
        assistant_sections = [
            ("Research problem & objective", "What problem does this paper address and what is its main objective?"),
            ("Methodology", "What method, model, framework or approach does the paper propose or use?"),
            ("Dataset & experimental setup", "What datasets, experimental setup or evaluation procedure are described?"),
            ("Key findings & results", "What are the main findings, results or conclusions reported by the authors?"),
            ("Limitations", "What limitations, challenges, drawbacks or constraints are explicitly reported?"),
            ("Future work", "What future research directions or next steps are explicitly mentioned?"),
        ]
        for title, query in assistant_sections:
            st.markdown(f"**{title}**")
            results = retrieve(query, paper_chunks, limit=2)
            if results:
                for c, _, _ in results:
                    st.markdown(f"- **Page {c.page}:** {c.text}")
            else:
                st.caption("No supporting passage found.")

    # Step 4: questions
    st.markdown("### 4 · Ask about the paper")
    st.markdown("<div class='rm-panel'><div class='rm-panel-title'>Need a specific answer?</div><div class='rm-panel-sub'>Ask ResearchMate about the methods, datasets, results, limitations or any other information contained in the uploaded papers. Answers stay linked to the original page evidence.</div></div>", unsafe_allow_html=True)
    if st.button("💬 Open Ask ResearchMate", type="primary", use_container_width=True):
        st.session_state.workspace_mode = "🧠 NLP Analysis"
        st.session_state.active_section = "Ask the Papers"
        st.rerun()

    st.caption("Tip: Start here for understanding. Use NLP Analysis when you want to inspect how ResearchMate processes the paper.")

else:
    st.markdown("<div class='rm-section-label' style='margin-top:1.6rem'>NLP analysis workspace</div>", unsafe_allow_html=True)
    nlp_sections = [
        "Preprocessing", "POS & Linguistic Analysis", "Keywords", "Entities",
        "Classification", "Topics", "Summary", "Compare Papers",
        "Research Gaps", "Ask the Papers", "NLP Pipeline"
    ]
    active_section = st.radio(
        "NLP analysis workspace",
        nlp_sections,
        key="active_section",
        horizontal=True,
        label_visibility="collapsed",
    )

if workspace_mode == "🧠 NLP Analysis" and active_section == "Preprocessing":
    st.subheader("Live NLP preprocessing")
    sentence_count = sum(len(sentences(c.text)) for c in chunks)
    token_count = sum(len(tokenize(c.text)) for c in chunks)
    stem_count = sum(len(stem_tokens(c.text)) for c in chunks)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Papers", len(uploads)); m2.metric("Pages", len(chunks))
    m3.metric("Sentences", sentence_count); m4.metric("Normalized tokens", token_count)

    options = [f"{c.paper} - page {c.page}" for c in chunks]
    selected_page = st.selectbox("Inspect page", options)
    chunk = chunks[options.index(selected_page)]
    text, sent_col = st.columns(2)
    with text:
        st.markdown("**1. Extracted PDF text**")
        st.text_area("Original", chunk.text, height=260, disabled=True, label_visibility="collapsed")
    with sent_col:
        st.markdown("**2. Sentence segmentation**")
        st.dataframe(
            [{"Sentence":i+1,"Text":s} for i,s in enumerate(sentences(chunk.text))],
            use_container_width=True, hide_index=True, height=260
        )

    st.markdown("**3. Tokenization + stopword removal + lemmatization**")
    st.code(" | ".join(tokenize(chunk.text)[:100]) or "No tokens extracted")
    st.markdown("**4. Stemming**")
    st.code(" | ".join(stem_tokens(chunk.text)[:100]) or "No stems extracted")
    st.caption(f"Normalized tokens: {token_count} · Stemmed tokens: {stem_count}")

    st.markdown("**Most frequent normalized terms**")
    st.write(" · ".join(f"{t} ({n})" for t,n in most_common_terms(chunks)))

elif workspace_mode == "🧠 NLP Analysis" and active_section == "POS & Linguistic Analysis":
    st.subheader("POS tagging and linguistic analysis")
    st.caption("POS tags identify grammatical roles such as nouns, verbs, adjectives and adverbs.")
    options = [f"{c.paper} - page {c.page}" for c in chunks]
    selected_page = st.selectbox("Select page for POS analysis", options, key="pos_page")
    chunk = chunks[options.index(selected_page)]
    tagged = pos_tag_text(chunk.text)
    st.dataframe(tagged, use_container_width=True, hide_index=True, height=350)
    if tagged:
        counts = pd.DataFrame(tagged).groupby("POS").size().reset_index(name="Count").sort_values("Count", ascending=False)
        st.markdown("**POS distribution**")
        st.dataframe(counts, use_container_width=True, hide_index=True)

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Keywords":
    keyword_rows = keyword_scores(chunks)
    st.markdown("""
    <div class="rm-keyword-hero">
      <div class="rm-keyword-kicker">TF-IDF research vocabulary</div>
      <h2>Keywords & Keyphrases</h2>
      <p>Explore the terms that are most distinctive across your uploaded papers. Higher TF-IDF scores indicate terms that carry more distinguishing weight within the current paper collection.</p>
    </div>
    """, unsafe_allow_html=True)

    if not keyword_rows:
        st.info("No keywords could be extracted from the uploaded papers.")
    else:
        max_score = max(score for _, score in keyword_rows) or 1.0
        top_term, top_score = keyword_rows[0]
        phrase_count = sum(1 for term, _ in keyword_rows if " " in term)
        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, note in [
            (k1, "Keyphrases", str(len(keyword_rows)), "Top extracted terms"),
            (k2, "Top term", top_term, f"Score {top_score:.4f}"),
            (k3, "Bigrams", str(phrase_count), "Multi-word phrases"),
            (k4, "Method", "TF-IDF", "Unigrams + bigrams"),
        ]:
            with col:
                st.markdown(f"<div class='rm-kpi'><div class='rm-kpi-label'>{label}</div><div class='rm-kpi-value'>{value}</div><div class='rm-kpi-note'>{note}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='rm-keyword-section'>Top research vocabulary</div>", unsafe_allow_html=True)
        st.markdown("<div class='rm-keyword-note'>The cards below are ranked directly from the existing TF-IDF calculation. The bars are a visual normalization relative to the highest score in this result set; they are not probability values.</div>", unsafe_allow_html=True)

        cards = st.columns(3)
        for index, (term, score) in enumerate(keyword_rows[:12], 1):
            with cards[(index - 1) % 3]:
                pct = max(3, min(100, (score / max_score) * 100))
                st.markdown(
                    f"<div class='rm-keyword-card'><div class='rm-keyword-rank'>#{index:02d}</div>"
                    f"<div class='rm-keyword-name'>{term}</div>"
                    f"<div class='rm-keyword-score'><span>TF-IDF relevance</span><span>{score:.4f}</span></div>"
                    f"<div class='rm-keyword-bar'><span style='width:{pct:.1f}%'></span></div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='rm-keyword-section'>Full ranked list</div>", unsafe_allow_html=True)
        rows = [{"Rank": i, "Keyphrase": p, "TF-IDF score": round(s, 4)} for i, (p, s) in enumerate(keyword_rows, 1)]
        st.dataframe(rows, use_container_width=True, hide_index=True)

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Entities":
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">Research entity intelligence</div>
      <h2>Entities & Research Concepts</h2>
      <p>Surface important datasets, models, metrics, institutions and authors using the research-oriented extraction rules already implemented in ResearchMate.</p>
    </div>
    """, unsafe_allow_html=True)
    entity_data = extract_entities(chunks)
    total_entities = sum(len(v) for v in entity_data.values())
    a,b,c = st.columns(3)
    a.metric("Entity categories", len(entity_data)); b.metric("Extracted mentions", total_entities); c.metric("Source pages", len(chunks))
    st.markdown("<div class='rm-panel'><div class='rm-panel-title'>How to read these results</div><div class='rm-panel-sub'>These are pattern-based research entities, so they are useful for exploration but should be verified against the original paper.</div></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (label, values) in enumerate(entity_data.items()):
        with cols[idx % 2]:
            chips = ''.join(f"<span class='rm-chip'>{v}</span>" for v in values[:30]) if values else '<span style="color:#8293a3;font-size:.76rem">No matching entities found.</span>'
            st.markdown(f"<div class='rm-entity-card'><div class='rm-dashboard-kicker' style='color:#7890a5'>{label}</div><div class='rm-entity-count'>{len(values)}</div><div style='margin-top:.35rem'>{chips}</div></div>", unsafe_allow_html=True)

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Classification":
    st.subheader("Research-paper text classification")
    st.caption("A transparent TF-IDF keyword baseline estimates a broad research domain. "
               "The result is intended as a lightweight, interpretable classification aid.")
    results, evaluation = classify_papers(chunks)
    st.dataframe(results, use_container_width=True, hide_index=True)
    if evaluation:
        st.markdown("**Classifier evaluation**")
        st.json(evaluation)

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Topics":
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">NMF topic discovery</div>
      <h2>Discover the themes inside your papers</h2>
      <p>Non-negative Matrix Factorization groups recurring vocabulary into interpretable topic themes across the uploaded research collection.</p>
    </div>
    """, unsafe_allow_html=True)
    topics = discover_topics(chunks)
    if topics:
        t1,t2,t3 = st.columns(3)
        t1.metric("Topics discovered", len(topics)); t2.metric("Papers analyzed", len(uploads)); t3.metric("Pages indexed", len(chunks))
        st.markdown("<div class='rm-panel'><div class='rm-panel-title'>Topic map</div><div class='rm-panel-sub'>Each card shows the highest-ranked vocabulary terms returned by the existing NMF topic model. The topic labels are descriptive UI labels, not manually assigned categories.</div></div>", unsafe_allow_html=True)
        cards = st.columns(2)
        for i, topic in enumerate(topics, 1):
            with cards[(i-1)%2]:
                chips = ''.join(f"<span class='rm-chip'>{term}</span>" for term in topic)
                st.markdown(f"<div class='rm-topic-card'><div class='rm-topic-num'>Topic {i:02d}</div><div class='rm-topic-name'>Theme {i}</div><div>{chips}</div></div>", unsafe_allow_html=True)
    else:
        st.info("Upload at least two pages of text to discover topics.")

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Summary":
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">TextRank extractive summary</div>
      <h2>Research summary with source traceability</h2>
      <p>Important original sentences are ranked from the paper text. ResearchMate does not rewrite them into unsupported generated claims.</p>
    </div>
    """, unsafe_allow_html=True)
    summary_rows = textrank_summary(chunks)
    if summary_rows:
        s1,s2,s3 = st.columns(3)
        s1.metric("Summary sentences", len(summary_rows)); s2.metric("Papers", len(uploads)); s3.metric("Pages indexed", len(chunks))
        for i, (sentence, paper, page) in enumerate(summary_rows, 1):
            st.markdown(f"<div class='rm-summary-card'><span class='rm-summary-index'>{i:02d}</span><span class='rm-badge source'>📄 {paper} · Page {page}</span><div class='rm-summary-text'>{sentence}</div></div>", unsafe_allow_html=True)
    else:
        st.info("No summary sentences could be ranked from the uploaded text.")

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Compare Papers":
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">Multi-paper literature view</div>
      <h2>Compare papers side by side</h2>
      <p>Use the extracted research signals and structured comparison data to inspect similarities and differences across your uploaded papers.</p>
    </div>
    """, unsafe_allow_html=True)
    if len(uploads) < 2:
        st.info("Upload at least two research papers to compare them.")
    else:
        comparison = compare_papers(chunks)
        c1,c2,c3 = st.columns(3)
        c1.metric("Papers", len(uploads)); c2.metric("Pages", len(chunks)); c3.metric("Comparison rows", len(comparison))
        st.markdown("<div class='rm-panel'><div class='rm-panel-title'>Structured comparison</div><div class='rm-panel-sub'>The table below comes from the existing multi-paper comparison function. It is intended as a literature-review aid, not a ranking of paper quality.</div></div>", unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown("<div class='rm-panel-title' style='margin-top:1rem'>Extracted research signals</div>", unsafe_allow_html=True)
        signals = extract_research_signals(chunks)
        st.dataframe(signals, use_container_width=True, hide_index=True)

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Research Gaps":
    st.markdown("""
    <div class="rm-dashboard-hero">
      <div class="rm-dashboard-kicker">Research signal exploration</div>
      <h2>Potential research gaps & recurring signals</h2>
      <p>ResearchMate surfaces recurring limitation and future-work signals from the uploaded papers. These are prompts for human investigation, not proof of a novel scientific gap.</p>
    </div>
    """, unsafe_allow_html=True)
    if len(uploads) < 2:
        st.info("Upload at least two research papers for cross-paper gap analysis.")
    else:
        gaps = detect_research_gaps(chunks)
        if gaps:
            g1,g2,g3 = st.columns(3)
            g1.metric("Signals found", len(gaps)); g2.metric("Papers", len(uploads)); g3.metric("Pages", len(chunks))
            for i, gap in enumerate(gaps, 1):
                st.markdown(f"<div class='rm-gap-card'><div class='rm-gap-title'><span class='rm-badge'>Signal {i:02d}</span> {gap['title']}</div><div style='margin-top:.55rem;color:#42566b;font-size:.84rem;line-height:1.6'>{gap['gap']}</div><div class='rm-gap-evidence'><b>Evidence:</b> {gap['evidence']}</div></div>", unsafe_allow_html=True)
        else:
            st.info("No strong recurring gap signal was detected from the available limitations/future-work evidence.")

elif workspace_mode == "🧠 NLP Analysis" and active_section == "Ask the Papers":
    st.markdown("""
    <div class="rm-ask-hero">
      <div class="rm-ask-kicker">Evidence-first research assistant</div>
      <h2>Ask the Papers</h2>
      <p>Ask a focused question and ResearchMate retrieves the most relevant original passages from your uploaded papers, keeping the paper name and page attached to every result.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rm-ask-grid">
      <div class="rm-ask-tip"><b>📊 Methods & evaluation</b>“How were the results evaluated?”</div>
      <div class="rm-ask-tip"><b>🧪 Datasets</b>“Which datasets were used in the experiments?”</div>
      <div class="rm-ask-tip"><b>⚠️ Limitations</b>“What limitations are reported by the authors?”</div>
      <div class="rm-ask-tip"><b>🔬 Findings</b>“What are the main experimental findings?”</div>
    </div>
    """, unsafe_allow_html=True)

    if "rm_question" not in st.session_state:
        st.session_state.rm_question = ""
    if "rm_evidence" not in st.session_state:
        st.session_state.rm_evidence = None
    if "rm_error" not in st.session_state:
        st.session_state.rm_error = ""

    suggestion_cols = st.columns(4)
    suggestions = [
        "Which datasets were used?",
        "How were the results evaluated?",
        "What are the stated limitations?",
        "What are the main findings?",
    ]
    for col, suggestion in zip(suggestion_cols, suggestions):
        with col:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.rm_question = suggestion
                st.session_state.rm_evidence = None
                st.session_state.rm_error = ""

    question = st.text_area(
        "Research question",
        key="rm_question",
        height=105,
        placeholder="Ask something specific about the uploaded papers…",
        help="Ask about methods, datasets, results, limitations, models, metrics or other information present in the papers."
    )

    ask_col, clear_col = st.columns([4,1])
    with ask_col:
        ask = st.button("🔎 Retrieve evidence", type="primary", use_container_width=True)
    with clear_col:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.rm_question = ""
        st.session_state.rm_evidence = None
        st.session_state.rm_error = ""

    if ask:
        if not question.strip():
            st.session_state.rm_evidence = None
            st.session_state.rm_error = "Enter a research question first."
        else:
            try:
                st.session_state.rm_evidence = retrieve(question.strip(), chunks)
                st.session_state.rm_error = ""
            except Exception as error:
                st.session_state.rm_evidence = None
                st.session_state.rm_error = f"Evidence retrieval could not be completed: {error}"

    if st.session_state.rm_error:
        st.warning(st.session_state.rm_error)

    evidence = st.session_state.rm_evidence
    if evidence:
        method = evidence[0][2]
        st.markdown(
            f"<div class='rm-verify'><b>Retrieval method:</b> {method} · Results below are original paper excerpts. "
            "Always verify the cited page in the source paper before using a claim.</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"### {len(evidence)} relevant evidence passage(s)")
        for rank, (c, score, method) in enumerate(evidence, 1):
            normalized_score = max(0, min(100, float(score) * 100))
            st.markdown(
                f"<div class='rm-evidence'>"
                f"<div class='rm-evidence-meta'>"
                f"<span class='rm-badge'>Evidence {rank}</span>"
                f"<span class='rm-badge source'>📄 {c.paper} · Page {c.page}</span>"
                f"<span class='rm-badge'>Relevance {score:.2f}</span>"
                f"</div>"
                f"<div class='rm-evidence-text'>{c.text}</div>"
                f"<div class='rm-score'><span style='width:{normalized_score:.1f}%'></span></div>"
                f"</div>",
                unsafe_allow_html=True
            )
    elif ask and not st.session_state.rm_error:
        st.info("No relevant evidence was found. Try a more specific question using terms from the papers.")

elif workspace_mode == "🧠 NLP Analysis" and active_section == "NLP Pipeline":
    st.subheader("End-to-end NLP pipeline")
    pipeline = [
        ("1. Document", "Research-paper PDF"),
        ("2. Extraction", "Text + page metadata"),
        ("3. Preprocessing", "Sentence segmentation → tokenization → stopwords → stemming → lemmatization"),
        ("4. Linguistic analysis", "POS tagging → named entity extraction"),
        ("5. Feature extraction", "TF-IDF → keywords → embeddings"),
        ("6. NLP analysis", "Classification → NMF topics → semantic similarity"),
        ("7. Higher-level analysis", "TextRank → comparison → limitations/future work"),
        ("8. Research intelligence", "Research-gap detection → evidence-based Q&A"),
    ]
    for title, desc in pipeline:
        st.markdown(f"<div style='padding:.8rem 1rem;margin:.45rem 0;border-left:4px solid #e56f5b;"
                    f"background:rgba(255,255,255,.75);border-radius:0 12px 12px 0'>"
                    f"<b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)

st.markdown("<div class='rm-footer'>ResearchMate · Evidence-first NLP for research paper exploration</div>", unsafe_allow_html=True)
