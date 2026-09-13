# ResearchMate — NLP Research Paper Assistant

ResearchMate is a shareable Streamlit web application for analyzing uploaded research PDFs. It deliberately uses classical and modern NLP techniques rather than treating the paper as a generic chat prompt.

## Features

- PDF text extraction with page-level source metadata
- NLP preprocessing: sentence splitting, tokenization, stopword filtering and light lemmatization
- TF-IDF keyword extraction
- Rule-based extraction of datasets, models, metrics, institutions and author names
- TextRank extractive summaries
- Topic discovery using NMF topic modelling across uploaded papers
- Semantic retrieval using `all-MiniLM-L6-v2`, with a TF-IDF fallback when unavailable
- Evidence-based question answering that returns the most relevant paper excerpts and page citations

## Run locally

```bash
cd outputs/researchmate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first semantic-search run downloads the sentence-transformer model. If that is not available, ResearchMate automatically uses TF-IDF retrieval.

## Publish for use on any device

The simplest free option is **Streamlit Community Cloud**. Once deployed, it provides a public HTTPS URL that works on phones, tablets, and computers.

1. Create a GitHub repository and upload the contents of this `researchmate` folder (including `.streamlit/config.toml`).
2. Open [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose **Create app**.
3. Select your repository, branch, and `app.py` as the main file.
4. Click **Deploy**, then share the generated URL.

For private or production deployments, deploy the same repository to Render, Railway, Azure, or AWS. Do not upload confidential papers to a public deployment unless you add user authentication and suitable storage/privacy controls.

### Share on your local Wi-Fi (temporary)

For testing from another phone or laptop on the same network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Open `http://YOUR_COMPUTER_IP:8501` on the other device. This is only for your local network; use a cloud deployment for internet access.

## Suggested demo questions

- Which datasets were used in these papers?
- What are the stated limitations of the proposed method?
- How is BERT evaluated?
- Find sections about the experimental results.

## NLP pipeline

```text
PDF → page-aware extraction → cleaning / sentence segmentation → chunks
    ├→ TF-IDF keywords
    ├→ entity patterns
    ├→ TextRank summary
    ├→ NMF topics
    └→ semantic retrieval → cited evidence answer
```

The app does not generate claims that are absent from the paper: its answer area returns retrieved evidence and asks the user to verify it in the cited page.
