# ForensicText - AI Detection & Humanizer (100+ Page Local Tool)

ForensicText is a professional-grade, privacy-focused tool designed to analyze, humanize, and verify text for AI-generated patterns. Unlike most tools, ForensicText runs **entirely locally** on your hardware, making it free and secure for documents of any length (even 100+ pages).

## 🚀 Key Features

- **Local AI Detection**: Uses a local RoBERTa model to classify text with high Precision.
- **Humanizer with Word-level Diff**: A T5-based paraphrasing engine that rewrites text. It features **sentence-by-sentence processing** and **word-level highlighting** (Red for removals, Green for additions).
- **Word to LaTeX Converter**: Seamlessly export `.docx` or `.txt` files directly to `.tex` LaTeX format. Supports a robust Pandoc integration or a native Python fallback.
- **AI Second Opinion (Reasoning)**: Integrated with **Ollama (Llama-3)** to provide a detailed, human-readable reasoning report on why a text was flagged.
- **Large Document Support**: Intelligent chunking allows you to upload and process 100+ page PDFs, DOCX, and TXT files.
- **Node-Free Architecture**: A "Python-Only" setup designed to run without Node.js or admin rights—the FastAPI server hosts the React frontend automatically.

## 🛠 Setup & Installation

### 1. Requirements
Ensure you have Python 3.9+ installed and, optionally:
- [Ollama](https://ollama.com/) for the Reasoning feature.
- [Pandoc](https://pandoc.org/) for high-fidelity LaTeX conversion.

### 2. Install Dependencies
```powershell
pip install fastapi uvicorn python-multipart PyMuPDF python-docx transformers torch requests sentencepiece pypandoc
```

### 3. Setup Ollama (Optional for Reasoning)
To use the "Second Opinion" verification:
```powershell
ollama pull llama3
```

## 🏃 How to Run

1. **Start the Server**:
   ```powershell
   python -m uvicorn AI_detection:app --reload
   ```

2. **Open the Tool**:
   Open your browser and navigate to:
   **[http://localhost:8000](http://localhost:8000)**

## 🧬 Architecture Overview

- **Backend**: FastAPI (Python) handles document parsing, local model inference (Torch/Transformers), and streams humanized text via SSE.
- **Frontend**: A standalone React application (served via `index.html`) using CDNs for a lightweight, zero-node-dependency experience.
- **Models & Libraries**:
    - Detector: `roberta-base-openai-detector`
    - Humanizer: `Vamsi/T5_Paraphrase_Paws` (tuned for sentence-level context)
    - Reasoning: `Llama-3-8B` (via local Ollama API)
    - Conversion: `pypandoc` (with custom fallback)

## 📄 License
MIT License - Free for local and personal use.
