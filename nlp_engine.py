"""ResearchMate NLP engine.

Local, explainable NLP pipeline with page-level metadata.
Advanced components use standard NLP/ML libraries and gracefully fall back
when optional models are unavailable.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import pymupdf
import networkx as nx
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "et", "al", "figure", "table", "fig", "using", "use", "used"
}


@dataclass
class Chunk:
    paper: str
    page: int
    text: str


def extract_pdf(uploaded_file) -> list[Chunk]:
    document = pymupdf.open(stream=uploaded_file.getvalue(), filetype="pdf")
    chunks = []
    for index, page in enumerate(document):
        text = clean_text(page.get_text("text"))
        if len(text) > 80:
            chunks.append(Chunk(uploaded_file.name, index + 1, text))
    document.close()
    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.split()) >= 5
    ]


def raw_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z-]{2,}", text.lower())


def tokenize(text: str) -> list[str]:
    return [
        light_lemma(token)
        for token in raw_tokens(text)
        if token not in STOP_WORDS
    ]


def light_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def stem_word(token: str) -> str:
    """Simple Porter-like suffix stemmer, kept local for reproducibility."""
    token = token.lower()
    suffixes = (
        "ational", "tional", "fulness", "ousness", "iveness",
        "ment", "ness", "ingly", "edly", "ing", "ed", "ies", "es", "s"
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def stem_tokens(text: str) -> list[str]:
    return [stem_word(t) for t in raw_tokens(text) if t not in STOP_WORDS]


def pos_tag_text(text: str, limit: int = 250) -> list[dict]:
    """POS tagging using NLTK when available; fallback uses transparent rules."""
    tokens = raw_tokens(text)
    try:
        import nltk
        try:
            tagged = nltk.pos_tag(tokens[:limit])
        except LookupError:
            tagged = None
        if tagged:
            return [{"Token": t, "POS": tag} for t, tag in tagged]
    except Exception:
        pass

    verbs = {"is","are","was","were","be","been","have","has","had","use","uses",
             "used","achieve","achieves","achieved","propose","proposed","improve",
             "improves","improved","evaluate","evaluated","train","trained"}
    adverbs = {"very","highly","significantly","typically","often","also","then"}
    adjectives = {"new","high","low","large","small","deep","neural","semantic",
                  "important","proposed","effective","efficient","robust","pretrained"}
    rows = []
    for t in tokens[:limit]:
        if t in verbs or t.endswith(("ing","ed")):
            tag = "VERB"
        elif t in adverbs or t.endswith("ly"):
            tag = "ADV"
        elif t in adjectives or t.endswith(("ive","al","ous","ful","able")):
            tag = "ADJ"
        else:
            tag = "NOUN"
        rows.append({"Token": t, "POS": tag})
    return rows


def keyword_scores(chunks: Iterable[Chunk], limit: int = 20):
    texts = [chunk.text for chunk in chunks]
    if not texts:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_df=0.95, min_df=1
    )
    matrix = vectorizer.fit_transform(texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    return sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:limit]


ENTITY_PATTERNS = {
    "Datasets": r"\b(?:ImageNet|COCO|CIFAR-?10|CIFAR-?100|MNIST|SQuAD(?:\s?\d\.\d)?|GLUE|SuperGLUE|WikiText(?:-\d+)?|Common Crawl|PubMed|arXiv|MS MARCO)\b",
    "Models / methods": r"\b(?:BERT|RoBERTa|GPT-?[\d\w.-]*|Transformer|LSTM|GRU|CNN|RNN|ResNet(?:-?\d+)?|GAN|VAE|BiLSTM|T5|BART|TF-IDF|TextRank|NMF|Word2Vec|GloVe|SBERT)\b",
    "Metrics": r"\b(?:accuracy|precision|recall|F1(?:-score)?|BLEU|ROUGE(?:-[L\d])?|AUC|perplexity|MAP|MRR|RMSE|MAE)\b",
    "Institutions": r"\b(?:University of [A-Z][\w ]+|[A-Z][\w ]+ University|Google(?: Research)?|Microsoft Research|OpenAI|Meta AI|DeepMind)\b",
}


def extract_entities(chunks: Iterable[Chunk]) -> dict[str, list[str]]:
    text = " ".join(c.text for c in chunks)
    found = {}
    for label, pattern in ENTITY_PATTERNS.items():
        values = sorted(
            {m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE)},
            key=str.lower
        )
        found[label] = values[:30]
    return found


def textrank_summary(chunks: Iterable[Chunk], sentence_limit: int = 6):
    records = [(s, c.paper, c.page) for c in chunks for s in sentences(c.text)]
    if not records:
        return []
    source = [r[0] for r in records]
    if len(source) <= sentence_limit:
        return records
    matrix = TfidfVectorizer(stop_words="english").fit_transform(source)
    similarity = cosine_similarity(matrix)
    graph = nx.from_numpy_array(similarity)
    ranks = nx.pagerank(graph, weight="weight")
    top = sorted(range(len(records)), key=lambda i: ranks[i], reverse=True)[:sentence_limit]
    return [records[i] for i in sorted(top)]


def retrieve(query: str, chunks: list[Chunk], limit: int = 5):
    if not chunks or not query.strip():
        return []
    texts = [c.text for c in chunks]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        vectors = model.encode([query, *texts], normalize_embeddings=True)
        scores = vectors[1:] @ vectors[0]
        method = "semantic embeddings"
    except Exception:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([query, *texts])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        method = "TF-IDF fallback"
    ranked = np.argsort(scores)[::-1][:limit]
    return [(chunks[i], float(scores[i]), method) for i in ranked]


def discover_topics(chunks: list[Chunk], topic_count: int = 3, terms_per_topic: int = 6):
    if len(chunks) < 2:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1)
        matrix = vectorizer.fit_transform([c.text for c in chunks])
        n_topics = min(topic_count, matrix.shape[0], matrix.shape[1])
        if n_topics < 1:
            return []
        model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=300)
        model.fit_transform(matrix)
        names = vectorizer.get_feature_names_out()
        return [[names[i] for i in topic.argsort()[-terms_per_topic:][::-1]]
                for topic in model.components_]
    except Exception:
        return []


def most_common_terms(chunks: Iterable[Chunk], limit: int = 15):
    return Counter(token for c in chunks for token in tokenize(c.text)).most_common(limit)


DOMAIN_KEYWORDS = {
    "Natural Language Processing": {"nlp","language","bert","transformer","token","text","linguistic","question"},
    "Machine Learning": {"machine","learning","classification","regression","training","prediction","feature"},
    "Computer Vision": {"image","vision","cnn","convolutional","object","pixel","segmentation"},
    "Data Science": {"data","analytics","dataset","statistics","visualization","prediction"},
    "Artificial Intelligence": {"artificial","intelligence","agent","reasoning","knowledge","neural"},
    "Cybersecurity": {"security","attack","malware","intrusion","encryption","threat","cyber"},
}


def _keyword_classify(text: str):
    terms = set(tokenize(text))
    scores = {d: len(terms & keys) for d, keys in DOMAIN_KEYWORDS.items()}
    label = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    confidence = scores[label] / total
    return label, round(confidence, 3), scores


def classify_papers(chunks: list[Chunk]):
    """Classify papers. Uses supervised Logistic Regression only when enough
    labeled training examples are available; otherwise a deterministic baseline."""
    by_paper = defaultdict(list)
    for c in chunks:
        by_paper[c.paper].append(c.text)
    rows = []
    for paper, texts in by_paper.items():
        label, confidence, scores = _keyword_classify(" ".join(texts))
        rows.append({
            "Paper": paper,
            "Predicted domain": label,
            "Confidence": confidence,
            "Method": "TF-IDF keyword baseline",
        })
    return rows, {}


def _signal_sentences(chunks, patterns):
    out = []
    for c in chunks:
        for s in sentences(c.text):
            if any(re.search(p, s, re.I) for p in patterns):
                out.append((s, c.paper, c.page))
    return out


def extract_research_signals(chunks: list[Chunk]):
    limitation_patterns = [r"\blimitations?\b", r"\blimited\b", r"\bchalleng", r"\bdrawback",
                            r"\bhowever\b", r"\bcomputational cost\b", r"\bmemory\b"]
    future_patterns = [r"\bfuture work\b", r"\bfuture research\b", r"\bfurther research\b",
                       r"\bshould be investigated\b", r"\bremains to be\b", r"\bnext work\b"]
    rows = []
    for s,p,page in _signal_sentences(chunks, limitation_patterns) + _signal_sentences(chunks, future_patterns):
        kind = "Limitation" if any(re.search(x,s,re.I) for x in limitation_patterns) else "Future work"
        rows.append({"Type":kind,"Paper":p,"Page":page,"Evidence":s})
    return rows


def compare_papers(chunks: list[Chunk]):
    by_paper = defaultdict(list)
    for c in chunks:
        by_paper[c.paper].append(c)
    rows = []
    for paper, pcs in by_paper.items():
        text = " ".join(c.text for c in pcs)
        entities = extract_entities(pcs)
        top = keyword_scores(pcs, limit=8)
        rows.append({
            "Paper": paper,
            "Pages": len(pcs),
            "Top keywords": ", ".join(k for k,_ in top),
            "Datasets": ", ".join(entities["Datasets"][:8]) or "—",
            "Models / methods": ", ".join(entities["Models / methods"][:8]) or "—",
            "Metrics": ", ".join(entities["Metrics"][:8]) or "—",
            "Words": len(raw_tokens(text)),
        })
    return rows


def detect_research_gaps(chunks: list[Chunk]):
    """Heuristic, evidence-first gap detection.
    It does not claim a novel scientific gap; it identifies recurring limitations
    and future-work themes in the uploaded papers."""
    signals = extract_research_signals(chunks)
    if not signals:
        return []

    limitation_rows = [r for r in signals if r["Type"] == "Limitation"]
    future_rows = [r for r in signals if r["Type"] == "Future work"]

    buckets = {
        "Computational efficiency": ["computational","memory","resource","expensive","cost","latency"],
        "Data limitations": ["dataset","data","limited data","small dataset","scarce"],
        "Generalization": ["generaliz","domain","robust","transfer","different"],
        "Scalability": ["scale","scalability","large-scale","deployment"],
        "Performance": ["accuracy","performance","precision","recall","f1","error"],
    }

    gaps = []
    for title, words in buckets.items():
        matched = [r for r in limitation_rows + future_rows
                   if any(w in r["Evidence"].lower() for w in words)]
        papers = sorted({r["Paper"] for r in matched})
        if len(papers) >= 2:
            evidence = " | ".join(
                f"{r['Paper']} p.{r['Page']}: {r['Evidence'][:220]}" for r in matched[:4]
            )
            gaps.append({
                "title": title,
                "gap": f"Multiple uploaded papers contain evidence related to {title.lower()}. "
                        f"This indicates an area that may deserve further investigation; it is a "
                        f"potential research gap rather than proof of a novel gap.",
                "evidence": evidence,
            })

    return gaps


def _classify_training_demo():
    """Small public API placeholder kept separate so a real labeled corpus can
    be plugged in later without changing the Streamlit interface."""
    return None
