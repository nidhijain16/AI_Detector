import os
import re
import difflib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List
import json
import asyncio
import requests

# Attempt to load pyMuPDF and docx for document parsing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

# Attempt to load Transformers and Torch for local AI Models
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    torch = None
    pipeline = None

app = FastAPI(title="AI Detection & Humanization Local API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# INITIALIZE LOCAL AI MODELS
# ---------------------------------------------------------------------
device = 0 if torch and torch.cuda.is_available() else -1
print(f"Loading local models on: {'GPU' if device == 0 else 'CPU'}...")

try:
    if pipeline:
        # standard base roberta detector. High probability = AI.
        print("Initializing RoBERTa detector...")
        detector_model = pipeline("text-classification", model="roberta-base-openai-detector", device=device)
        
        # popular t5 paraphrase model for humanizing
        print("Initializing T5 humanizer (manual load for stability)...")
        model_name = "Vamsi/T5_Paraphrase_Paws"
        
        # Manually load tokenizer and model to avoid 'Fast' conversion issues
        t5_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        paraphrase_model = pipeline(
            "text2text-generation", 
            model=t5_model, 
            tokenizer=t5_tokenizer, 
            device=device
        )
        print("Models loaded successfully!")
    else:
        detector_model = None
        paraphrase_model = None
        print("Transformers library not installed. Starting without local LLMs.")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    detector_model = None
    paraphrase_model = None

# ---------------------------------------------------------------------
# CHUNKING & TEXT PROCESSING ENGINE
# ---------------------------------------------------------------------
def split_into_chunks(text: str, max_words=350) -> List[str]:
    """
    Intelligently chunks 100+ pages of text into blocks of max_words.
    Prefers splitting at paragraphs, then sentences.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_words = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        
        words_in_p = len(p.split())
        
        # If adding this paragraph exceeds limit...
        if current_words + words_in_p > max_words:
            # If current chunk isn't empty, save it.
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0
                
            # If the single paragraph is still too big, we must split by sentence
            if words_in_p > max_words:
                sentences = re.split(r'(?<=[.!?]) +', p)
                for s in sentences:
                    words_in_s = len(s.split())
                    if current_words + words_in_s > max_words and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_words = 0
                    current_chunk.append(s)
                    current_words += words_in_s
            else:
                current_chunk.append(p)
                current_words += words_in_p
        else:
            current_chunk.append(p)
            current_words += words_in_p
            
    # Add any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ---------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------
@app.post("/api/upload_and_analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Accepts PDF, DOCX, or TXT. Extracts text, chunks it, and runs local AI detection.
    """
    filename = file.filename.lower()
    text = ""
    
    content = await file.read()
    
    if filename.endswith(".pdf"):
        if not fitz:
            raise HTTPException(status_code=500, detail="PyMuPDF not installed. Cannot parse PDF.")
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
            
    elif filename.endswith(".docx"):
        if not docx:
            raise HTTPException(status_code=500, detail="python-docx not installed. Cannot parse DOCX.")
        import io
        doc = docx.Document(io.BytesIO(content))
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    elif filename.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    text = text.strip()
    if not text:
        return {"status": "error", "message": "Document empty or unreadable."}
        
    # Chunking
    chunks = split_into_chunks(text)
    
    # Run AI Detection
    chunk_scores = []
    overall_score = 0.0
    
    if detector_model:
        # Process in batches or one-by-one
        for i, chunk in enumerate(chunks):
            # Model output: [{'label': 'Fake', 'score': 0.999}] or 'Real'
            res = detector_model(chunk[:2000], truncation=True)[0]
            # Convert 'Fake' / 'Real' logic to AI Percentage
            if res['label'] == 'Fake':
                score = res['score'] * 100
            else:
                score = (1 - res['score']) * 100
                
            chunk_scores.append({
                "chunk_id": i,
                "text": chunk,
                "ai_score_percent": round(score, 2)
            })
            
        # Overall simple average
        if chunk_scores:
            overall_score = sum(c['ai_score_percent'] for c in chunk_scores) / len(chunk_scores)
    else:
        # Fallback if no models loaded
        chunk_scores = [{"chunk_id": i, "text": c, "ai_score_percent": 0.0} for i, c in enumerate(chunks)]
        overall_score = 0.0

    return {
        "status": "success",
        "total_chunks": len(chunks),
        "overall_ai_score": round(overall_score, 2),
        "chunk_details": chunk_scores,
        "full_text": text
    }

class HumanizeRequest(BaseModel):
    chunks: List[str]

@app.post("/api/humanize")
async def humanize_stream(req: HumanizeRequest):
    """
    Since humanizing 100 pages is slow, we stream the output back chunk-by-chunk.
    """
    if not paraphrase_model:
        raise HTTPException(status_code=500, detail="Paraphrase model not loaded.")
        
    async def event_generator():
        for i, chunk in enumerate(req.chunks):
            # The T5_Paraphrase_Paws model expects input in this format
            prompt = "paraphrase: " + chunk + " </s>"
            
            # This is a blocking call. For real production with 100 concurrent users, 
            # you would offload to threadpool or Celery.
            out = paraphrase_model(prompt, max_length=512, num_return_sequences=1, truncation=True)
            rewritten_text = out[0]['generated_text']
            
            # Generate a word-level diff for highlighting
            s1 = chunk.split()
            s2 = rewritten_text.split()
            matcher = difflib.SequenceMatcher(None, s1, s2)
            diff_ops = matcher.get_opcodes()
            
            # Send standard Server-Sent Event (SSE)
            yield f"data: {json.dumps({'chunk_id': i, 'rewritten_text': rewritten_text, 'diff': diff_ops, 'original_words': s1, 'rewritten_words': s2})}\n\n"
            
            # Small async sleep to prevent loop monopolization
            await asyncio.sleep(0.01)
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")


class PlagiarismRequest(BaseModel):
    text: str
    database_texts: List[str] 

@app.post("/api/check_plagiarism")
async def check_plagiarism(req: PlagiarismRequest):
    """
    Basic diff comparison kept from original code for fallback.
    For local large-scale, a vector DB like Chroma is recommended.
    """
    if not req.text or not req.database_texts:
        return {"max_similarity_score": 0.0, "status": "No text provided"}
    
    highest_sim = 0.0
    best_match = ""
    
    # Simple chunking comparison
    req_chunks = split_into_chunks(req.text, 200)
    
    for db_text in req.database_texts:
        # Optimization: only compare if db text isn't empty
        if not db_text.strip(): continue
        
        # Compare highest match chunk against entire text diff
        seq = difflib.SequenceMatcher(None, req.text.lower(), db_text.lower())
        sim = seq.ratio() * 100
        
        if sim > highest_sim:
            highest_sim = sim
            best_match = db_text
            
    return {
        "max_similarity_score": round(highest_sim, 2),
        "verdict": "Likely Plagiarized" if highest_sim > 40 else "Original"
    }

class VerifyRequest(BaseModel):
    text: str
    metrics: dict

@app.post("/api/verify_with_llm")
async def verify_with_llm(req: VerifyRequest):
    """
    Uses local Ollama (Llama-3) to provide a "Second Opinion" on AI detection.
    """
    prompt = f"""
    Analyze the following text and determine if it is likely AI-generated or Human-written.
    I will provide some metrics to help you:
    - AI Classifier Score: {req.metrics.get('ai_score')}%
    - Sentence Length Variance (Burstiness): {req.metrics.get('burstiness')}
    - Vocab Diversity (TTR): {req.metrics.get('ttr')}

    TEXT:
    \"\"\"{req.text[:2000]}\"\"\"

    Provide a concise reasoning and a final verdict.
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return {"reasoning": response.json().get("response"), "status": "success"}
        else:
            return {"reasoning": "Ollama server error or model 'llama3' not found.", "status": "error"}
    except Exception as e:
        return {"reasoning": f"Could not connect to Ollama: {str(e)}", "status": "error"}

# To run:
# pip install fastapi uvicorn transformers torch PyMuPDF python-docx
# uvicorn AI_detection:app --reload

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """
    Serves the standalone React-based frontend if index.html exists.
    """
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ForensicText Backend is Running</h1><p>Please ensure index.html is in the same directory.</p>"
