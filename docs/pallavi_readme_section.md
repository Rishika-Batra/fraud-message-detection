# Grounded Explanation Pipeline (CA-2, Track B)
**Author:** Pallavi  
**Teammate:** Rishika (DistilBERT Classifier)

This module adds retrieval-augmented grounded explanations to our Fraud Message Detection project. When Rishika's fine-tuned DistilBERT classifier flags a message as fraudulent, this pipeline retrieves verified fraud advisories from trusted authorities (RBI, NPCI, Cybercrime Portal, CERT-In, SBI, HDFC) and uses Google Gemini to generate factual, cited explanations.

---

## Architecture Overview

```
Incoming SMS
    │
    ▼
[DistilBERT Classifier] (model/predict.py)
    │
    ├─── If "legitimate" ────────► Return: "Message looks legitimate; no explanation needed."
    │
    └─── If "fraud"
            │
            ▼
     [FAISS Semantic Retriever] (rag/retriever.py)
            │  Embeds query with all-MiniLM-L6-v2
            │  Searches rag/index/kb.index (cosine threshold: 0.35)
            │
            ▼
     [Grounded Gemini Explainer] (rag/explain.py)
            │  Strict prompt grounded strictly in retrieved notes
            │  Cites sources inline like [1], [2]
            │  Honesty rule: Flags false positives if message is a genuine OTP/alert
            │
            ▼
     [Gradio Web Interface] (app/app.py)
```

---

## Quickstart: How to Run the Demo

### 1. Prerequisites & Environment Setup
Open your terminal in the repository root directory:

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the example environment file and add your Gemini API key:

```bash
# On Windows:
copy .env.example .env
# On Linux / macOS:
cp .env.example .env
```

Open `.env` in any text editor and paste your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(You can obtain a free key from [Google AI Studio](https://aistudio.google.com/)).*

### 3. Verify Model Weights
Ensure the trained DistilBERT weights are located in:
```
model/saved_model/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer_config.json
  └── vocab.txt
```

### 4. Build or Rebuild the Semantic Index
Generate the FAISS vector index from our curated fraud knowledge base (`rag/knowledge_base.json`):

```bash
python rag/build_index.py
```
*This downloads the lightweight `all-MiniLM-L6-v2` embedding model locally and saves the index to `rag/index/kb.index` and `rag/index/metadata.json`.*

### 5. Launch the Interactive Gradio Web Demo
Start the web interface:

```bash
python app/app.py
```
Open your browser and visit: `http://127.0.0.1:7860`

The UI provides:
- A text box to enter/paste any message
- Clickable preset examples (clear fraud, utility threats, job scams, legitimate chat)
- Instant classification label and confidence percentage
- Grounded explanation with inline bracket citations
- Clickable links to verified government and banking advisories

### 6. Run the Benchmark Test Suite
Run the 5 standard test cases (evaluating clear fraud, borderline cases, legitimate chat, and genuine OTP false-positive handling):

```bash
python samples/run_samples.py
```
Results are saved to:
- `samples/sample_outputs.json`
- `samples/sample_outputs.md`

---

## File Structure

```
├── rag/
│   ├── knowledge_base.json    # 14 curated advisories from RBI, NPCI, Cybercrime Portal, etc.
│   ├── build_index.py         # Builds FAISS index from knowledge_base.json
│   ├── retriever.py           # Top-k similarity search with 0.35 threshold
│   ├── explain.py             # Grounded LLM explainer using google-genai SDK
│   └── index/                 # Generated FAISS index & metadata (git-ignored)
├── app/
│   ├── pipeline.py            # End-to-end analyze() pipeline
│   └── app.py                 # Gradio interactive web application
├── samples/
│   ├── run_samples.py         # 5-sample benchmark runner script
│   ├── sample_outputs.json    # Raw evaluation data
│   └── sample_outputs.md      # Formatted evaluation report
├── docs/
│   └── pallavi_readme_section.md # This documentation file
├── .env.example               # Template for API keys
├── .gitignore                 # Configured to ignore secrets, venv, and cache
└── requirements.txt           # Unified dependency specification
```

---

## Key Viva Talking Points

1. **Why RAG instead of asking an LLM directly?**  
   Zero hallucination. The LLM is strictly constrained to the retrieved advisory notes and must cite official sources (RBI, NPCI, CERT-In). It cannot invent false security advisories or make up rules.

2. **How is the False-Positive problem addressed?**  
   The classifier often flags genuine OTPs as "fraud" due to high-frequency keywords like "OTP", "bank", "INR". In our pipeline:
   - The knowledge base contains explicit notes on authentic OTP/transaction alert characteristics (`kb13`, `kb14`).
   - The prompt enforces a critical honesty rule: If the retrieved notes do not demonstrate a fraud pattern, the LLM explicitly reports that the message does not match fraud advisories and that the classifier likely produced a false positive.

3. **Why FAISS + `all-MiniLM-L6-v2`?**  
   MiniLM produces compact 384-dimensional embeddings that run fast entirely on CPU without requiring an external API key or network latency for retrieval. Normalizing embeddings allows using `IndexFlatIP` for exact cosine similarity search.

