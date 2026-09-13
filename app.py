import streamlit as st
import pandas as pd

from nlp_engine import (
    discover_topics, extract_entities, extract_pdf, keyword_scores, most_common_terms,
    retrieve, sentences, textrank_summary, tokenize, stem_tokens, pos_tag_text,
    classify_papers, compare_papers, detect_research_gaps, extract_research_signals,
)

st.set_page_config(
    page_title="ResearchMate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { --ink:#263e5a; --muted:#64748b; --cream:#fffaf1; --coral:#e56f5b; }
.stApp { background:linear-gradient(135deg,#fffaf1 0%,#f6fbff 48%,#eef9f5 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(255,250,241,.72); }
h1,h2,h3 { color:#28445e !important; font-family:Georgia,serif; }
.brand { text-align:center;color:#28445e;font-family:Georgia,serif;font-size:1.7rem;font-weight:700;
letter-spacing:.09em;margin:.85rem 0 1.4rem; }
.brand span { color:#e56f5b;font-size:.9rem;vertical-align:middle; }
[data-testid="stFileUploader"] { text-align:center;margin:2.5rem 0; }
[data-testid="stFileUploader"] section { background:#fff!important;border:1px dashed #e5c8ac!important;justify-content:center!important; }
[data-testid="stFileUploader"] section>div { justify-content:center!important; }
[data-testid="stFileUploader"] button { background:#fff!important;border:1px solid #e56f5b!important;
color:#b84d3f!important;font-size:0!important;margin-left:auto!important;margin-right:auto!important; }
[data-testid="stFileUploader"] button::after { content:"Upload files";color:#b84d3f;font-size:.95rem;font-weight:650; }
.nlp-pictures { display:flex;justify-content:center;align-items:center;gap:1.4rem;margin:1.3rem 0 -1.15rem; }
.nlp-picture { width:64px;height:64px;display:grid;place-items:center;border-radius:18px;background:#fff;
border:1px solid #eee1d2;font-size:1.8rem;box-shadow:0 7px 20px rgba(55,79,97,.05); }
.capability-title { text-align:center;color:#28445e;font-family:Georgia,serif;font-size:1.35rem;margin:2rem 0 .75rem; }
.capability { background:rgba(255,255,255,.72);border:1px solid #eee1d2;border-radius:14px;padding:.85rem .9rem;min-height:94px; }
.capability strong { color:#a84d40;font-size:.88rem;display:block;margin-bottom:.3rem; }
.capability span { color:#60758a;font-size:.78rem;line-height:1.35; }
[data-testid="stMetric"] { background:rgba(255,255,255,.72);border:1px solid #eadfce;border-radius:16px;padding:.85rem; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; }
.stTabs [data-baseweb="tab"] { background:#fffdf9;border:1px solid #eddfcb;border-radius:999px;padding:.4rem .85rem;color:#526b83; }
.stTabs [aria-selected="true"] { background:#ffead7;color:#9c4034; }
.stButton>button { background:#e56f5b;border:0;border-radius:999px;color:#fff;font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand'><span>✦</span> RESEARCHMATE</div>", unsafe_allow_html=True)
st.markdown("""
<div class="nlp-pictures">
  <div class="nlp-picture">📄</div><div class="nlp-picture">🧠</div><div class="nlp-picture">🔎</div>
</div>
""", unsafe_allow_html=True)

capabilities = st.columns(4)
for column, title, description in [
    (capabilities[0], "NLP preprocessing", "Sentence segmentation, tokenization, stopwords, stemming and lemmatization."),
    (capabilities[1], "Linguistic analysis", "POS tagging, named entities, TF-IDF keywords and research signals."),
    (capabilities[2], "Semantic analysis", "Embeddings, similarity search, NMF topic modelling and classification."),
    (capabilities[3], "Research intelligence", "Multi-paper comparison, gap detection and evidence-based retrieval."),
]:
    with column:
        st.markdown(f"<div class='capability'><strong>{title}</strong><span>{description}</span></div>",
                    unsafe_allow_html=True)

_, upload_center, _ = st.columns([1, 2.6, 1])
with upload_center:
    uploads = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True,
                               label_visibility="collapsed")

if not uploads:
    st.stop()

chunks = []
for upload in uploads:
    try:
        chunks.extend(extract_pdf(upload))
    except Exception as error:
        st.warning(f"Could not read {upload.name}: {error}")

if not chunks:
    st.error("No readable research-paper text was found in the uploaded files.")
    st.stop()

st.success(f"Analysis complete: {len(uploads)} paper(s) and {len(chunks)} pages indexed.")

tabs = st.tabs([
    "Preprocessing", "POS & Linguistic Analysis", "Keywords", "Entities",
    "Classification", "Topics", "Summary", "Compare Papers",
    "Research Gaps", "Ask the Papers", "NLP Pipeline"
])

with tabs[0]:
    st.subheader("Live NLP preprocessing")
    sentence_count = sum(len(sentences(c.text)) for c in chunks)
    token_count = sum(len(tokenize(c.text)) for c in chunks)
    stem_count = sum(len(stem_tokens(c.text)) for c in chunks)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Papers", len(uploads)); m2.metric("Pages", len(chunks))
    m3.metric("Sentences", sentence_count); m4.metric("Normalized tokens", token_count)

    options = [f"{c.paper} - page {c.page}" for c in chunks]
    selected = st.selectbox("Inspect page", options)
    chunk = chunks[options.index(selected)]
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

with tabs[1]:
    st.subheader("POS tagging and linguistic analysis")
    st.caption("POS tags identify grammatical roles such as nouns, verbs, adjectives and adverbs.")
    options = [f"{c.paper} - page {c.page}" for c in chunks]
    selected = st.selectbox("Select page for POS analysis", options, key="pos_page")
    chunk = chunks[options.index(selected)]
    tagged = pos_tag_text(chunk.text)
    st.dataframe(tagged, use_container_width=True, hide_index=True, height=350)
    if tagged:
        counts = pd.DataFrame(tagged).groupby("POS").size().reset_index(name="Count").sort_values("Count", ascending=False)
        st.markdown("**POS distribution**")
        st.dataframe(counts, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Keyword and keyphrase extraction")
    st.caption("TF-IDF gives more weight to terms that are distinctive in the uploaded papers.")
    rows = [{"Keyphrase":p,"TF-IDF score":round(s,4)} for p,s in keyword_scores(chunks)]
    st.dataframe(rows, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Named entity extraction")
    st.caption("Research-oriented entity patterns identify datasets, models, metrics and institutions.")
    for label, values in extract_entities(chunks).items():
        st.markdown(f"**{label}**")
        st.write(", ".join(values) if values else "No matching entities found.")

with tabs[4]:
    st.subheader("Research-paper text classification")
    st.caption("A transparent TF-IDF + Logistic Regression classifier estimates a broad research domain. "
               "If no trained model is available, ResearchMate uses a keyword-based baseline so the app remains runnable.")
    results, evaluation = classify_papers(chunks)
    st.dataframe(results, use_container_width=True, hide_index=True)
    if evaluation:
        st.markdown("**Classifier evaluation**")
        st.json(evaluation)

with tabs[5]:
    st.subheader("NMF topic modelling")
    topics = discover_topics(chunks)
    if topics:
        for i, topic in enumerate(topics, 1):
            st.write(f"**Theme {i}:** {' · '.join(topic)}")
    else:
        st.info("Upload at least two pages of text to discover topics.")

with tabs[6]:
    st.subheader("TextRank extractive summary")
    st.caption("High-ranked original sentences — no generated claims.")
    for sentence, paper, page in textrank_summary(chunks):
        st.write(sentence)
        st.caption(f"Source · {paper} · page {page}")

with tabs[7]:
    st.subheader("Multi-paper comparison")
    if len(uploads) < 2:
        st.info("Upload at least two research papers to compare them.")
    else:
        comparison = compare_papers(chunks)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown("**Extracted research signals**")
        signals = extract_research_signals(chunks)
        st.dataframe(signals, use_container_width=True, hide_index=True)

with tabs[8]:
    st.subheader("Research-gap detection")
    if len(uploads) < 2:
        st.info("Upload at least two research papers for cross-paper gap analysis.")
    else:
        gaps = detect_research_gaps(chunks)
        if gaps:
            for gap in gaps:
                st.markdown(f"### {gap['title']}")
                st.write(gap["gap"])
                st.caption(f"Evidence · {gap['evidence']}")
        else:
            st.info("No strong recurring gap signal was detected from the available limitations/future-work evidence.")

with tabs[9]:
    st.subheader("Evidence desk")
    st.caption("ResearchMate returns original excerpts with paper and page metadata rather than inventing evidence.")
    question = st.text_input("Your question", placeholder="What datasets were used and how were results evaluated?")
    if question:
        evidence = retrieve(question, chunks)
        if evidence:
            st.caption(f"Search method: {evidence[0][2]}. Verify the cited page before using a claim.")
            for rank,(c,score,method) in enumerate(evidence,1):
                with st.expander(f"Evidence {rank} · {c.paper}, page {c.page} · relevance {score:.2f}", expanded=rank==1):
                    st.write(c.text)
        else:
            st.info("No relevant evidence was found.")

with tabs[10]:
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
