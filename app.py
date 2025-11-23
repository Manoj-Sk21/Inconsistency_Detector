# app.py
"""
Inconsistency Detector — Groq-first MVP
- Tries Groq API when GROQ_API_KEY + GROQ_API_URL are set.
- Falls back to Hugging Face Inference API if HF_API_KEY is set.
- Otherwise uses rule-based explanations.
- Keep your templates/index.html unchanged.
"""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from flask import send_file
from datetime import datetime

import os
import re
import json
import requests
from flask import Flask, request, render_template, jsonify
import spacy
import dateparser
import dateparser.search
from sentence_transformers import SentenceTransformer, util

# --- NLP models (local)
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

# ---- Patterns / lists ----
TIME_PATTERNS = [r"\b(\d{1,2}:\d{2})\b", r"\b(\d{1,2}\s?(AM|PM|am|pm))\b"]
VEHICLE_REGEX = r"\b([A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{1,4})\b"
COLOR_WORDS = ["red","black","white","blue","green","yellow","grey","gray","brown","orange","pink","purple"]
KEYWORDS = ["knife","gun","vehicle","bike","plate","helmet","weapon","two-wheeler","scarf"]


# ---- Helper extractors ----
def extract_entities(text):
    doc = nlp(text or "")
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC","FAC")]
    times = [ent.text for ent in doc.ents if ent.label_ in ("DATE","TIME")]
    return {"persons": persons, "places": places, "times": times}


def extract_times(text):
    if not text:
        return []
    found = []
    for p in TIME_PATTERNS:
        found += re.findall(p, text)
    try:
        dp = dateparser.search.search_dates(text)
        if dp:
            for t in dp:
                # dp returns tuples like ("21 Nov 2025", datetime)
                found.append(str(t[1]))
    except Exception:
        pass
    cleaned = []
    for t in found:
        if isinstance(t, tuple):
            cleaned.append(t[0])
        else:
            cleaned.append(t)
    return sorted(list(set(cleaned)))


def extract_colors(text):
    t = (text or "").lower()
    return [c for c in COLOR_WORDS if c in t]


def extract_plates(text):
    txt = text or ""
    matches = re.findall(VEHICLE_REGEX, txt, flags=re.IGNORECASE)
    return [m.replace(" ", "").upper() for m in matches]


def extract_keywords(text):
    t = (text or "").lower()
    return [k for k in KEYWORDS if k in t]


def embed_sim(a, b):
    if not a or not b:
        return 0.0
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(e1, e2).item())


# ---- Conflict detection ----
def detect_conflicts(docs):
    conflicts = []

    # Person/entity mismatches
    persons = {name: [p.lower() for p in extract_entities(txt)["persons"]] for name, txt in docs.items()}
    names = list(docs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            set_a = set(persons[a])
            set_b = set(persons[b])
            if set_a != set_b:
                conflicts.append({
                    "type": "PERSON_MISMATCH",
                    "between": [a, b],
                    "only_in_a": sorted(list(set_a - set_b)),
                    "only_in_b": sorted(list(set_b - set_a))
                })

    # Time mismatches
    times = {name: extract_times(txt) for name, txt in docs.items()}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            if times[a] and times[b] and set(times[a]) != set(times[b]):
                conflicts.append({
                    "type": "TIME_MISMATCH",
                    "between": [a, b],
                    "times_a": times[a],
                    "times_b": times[b]
                })

    # Color mismatches
    colors = {name: extract_colors(txt) for name, txt in docs.items()}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            if colors[a] and colors[b] and set(colors[a]).isdisjoint(colors[b]):
                conflicts.append({
                    "type": "COLOR_MISMATCH",
                    "between": [a, b],
                    "colors_a": colors[a],
                    "colors_b": colors[b]
                })

    # Vehicle plate mismatch
    plates = {name: extract_plates(txt) for name, txt in docs.items()}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            if plates[a] and plates[b] and set(plates[a]) != set(plates[b]):
                conflicts.append({
                    "type": "VEHICLE_MISMATCH",
                    "between": [a, b],
                    "plate_a": plates[a],
                    "plate_b": plates[b]
                })

    # Missing details
    for name, txt in docs.items():
        kws = extract_keywords(txt)
        for k in kws:
            present_in = [n for n, t in docs.items() if k in (t or "").lower()]
            if len(present_in) != len(docs):
                conflicts.append({
                    "type": "MISSING_DETAIL",
                    "keyword": k,
                    "present_in": present_in
                })

    return conflicts


# ---- LLM integration: Groq first, then Hugging Face fallback ----

import os, requests, json

def call_groq(prompt_text):
    """
    Uses Groq's OpenAI-compatible chat/completions endpoint.
    Requires these env vars:
      - GROQ_API_URL  (e.g. https://api.groq.com/openai/v1/chat/completions)
      - GROQ_API_KEY
      - GROQ_MODEL    (e.g. meta-llama/llama-4-scout-17b-16e-instruct)
    Returns generated text (string) or raises RuntimeError with details.
    """
    api_url = os.environ.get("GROQ_API_URL")
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL")

    if not api_url or not api_key or not model:
        raise RuntimeError("Missing GROQ_API_URL, GROQ_API_KEY or GROQ_MODEL environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Try chat-completions shape first (OpenAI-compatible)
    payload_chat = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": 400
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload_chat, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Network/error calling Groq: {e}")

    # If success, try to parse common OpenAI-like response shapes
    if resp.status_code == 200:
        data = resp.json()
        # 1) chat-completions: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        # 2) completions style: choices[0].text
        try:
            return data["choices"][0]["text"]
        except Exception:
            pass
        # 3) some Groq responses place the text in other fields
        # return a compact JSON string as a fallback
        return json.dumps(data)

    # If chat-style failed, try a prompt/completions fallback payload
    # (some endpoints accept "prompt")
    payload_completion = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": 400,
        "temperature": 0.0
    }
    try:
        resp2 = requests.post(api_url, headers=headers, json=payload_completion, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Network/error calling Groq (fallback): {e}")

    if resp2.status_code == 200:
        data2 = resp2.json()
        try:
            return data2["choices"][0]["text"]
        except Exception:
            return json.dumps(data2)

    # Both attempts failed — raise a detailed error so you can paste it here if needed
    raise RuntimeError(f"Groq API errors:\nPrimary ({resp.status_code}): {resp.text}\nFallback ({resp2.status_code}): {resp2.text}")



def call_huggingface(prompt_text):
    """
    Fallback to Hugging Face Inference API if HF_API_KEY is configured.
    """
    hf_key = os.environ.get("HF_API_KEY")
    hf_model = os.environ.get("HF_MODEL", "gpt2")  # default small model; replace if you set HF_MODEL
    if not hf_key:
        raise RuntimeError("Hugging Face key not configured")

    headers = {"Authorization": f"Bearer {hf_key}"}
    api_url = f"https://api-inference.huggingface.co/models/{hf_model}"
    payload = {"inputs": prompt_text, "parameters": {"max_new_tokens": 300, "temperature": 0.0}}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")
    data = resp.json()
    # HF often returns a list of dicts with 'generated_text'
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Hugging Face response error: {data['error']}")
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # fallback to string
    return json.dumps(data)


def generate_explanation_with_providers(conflicts):
    """
    Try Groq, then Hugging Face. On any provider error, fall back to rule-based explanation.
    """
    prompt = (
        "You are a forensic AI assistant for police investigations. "
        "Given the following conflict list (JSON), explain each conflict in 1-2 short sentences, "
        "why it matters, and give a confidence score between 0 and 1.\n\n"
        f"{json.dumps(conflicts, indent=2)}\n\n"
        "Output: Plain text, bullet points or numbered lines."
    )

    # Try Groq first
    try:
        if os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_URL"):
            return call_groq(prompt)
    except Exception as e:
        # keep the error message but continue to fallback
        groq_err = f"(Groq error: {str(e)})\n"
    else:
        groq_err = ""

    # Try Hugging Face
    try:
        if os.environ.get("HF_API_KEY"):
            return groq_err + call_huggingface(prompt)
    except Exception as e:
        hf_err = f"(HuggingFace error: {str(e)})\n"
    else:
        hf_err = groq_err

    # If both providers failed / not configured, use rule-based
    return (groq_err + hf_err + "\n") + rule_based(conflicts)


# ---- Rule-based fallback explanation ----
def rule_based(conflicts):
    lines = []
    for c in conflicts:
        t = c.get("type")
        if t == "TIME_MISMATCH":
            lines.append(f"Timeline mismatch between {c['between']}: {c.get('times_a')} vs {c.get('times_b')} (Confidence ~0.7).")
        elif t == "COLOR_MISMATCH":
            lines.append(f"Appearance mismatch between {c['between']}: {c.get('colors_a')} vs {c.get('colors_b')} (Confidence ~0.6).")
        elif t == "VEHICLE_MISMATCH":
            lines.append(f"Vehicle plate mismatch between {c['between']}: {c.get('plate_a')} vs {c.get('plate_b')} (Confidence ~0.8).")
        elif t == "PERSON_MISMATCH":
            lines.append(f"Person/entity mismatch between {c['between']}: only_in_a={c.get('only_in_a')} only_in_b={c.get('only_in_b')} (Confidence ~0.75).")
        elif t == "MISSING_DETAIL":
            lines.append(f"Missing detail '{c.get('keyword')}' present in {c.get('present_in')}. Consider checking other sources (Confidence ~0.5).")
        else:
            lines.append(str(c))
    if not lines:
        return "No contradictions detected with current heuristics. (Confidence ~0.9)"
    return "\n".join(lines)


# ---- Flask routes ----
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    docs = {
        "witness1": data.get("witness1", "") or data.get("w1", ""),
        "witness2": data.get("witness2", "") or data.get("w2", ""),
        "fir": data.get("fir", ""),
        "cctv": data.get("cctv", "")
    }

    conflicts = detect_conflicts(docs)

    # similarity checks
    names = list(docs.keys())
    sims = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = embed_sim(docs[names[i]] or "", docs[names[j]] or "")
            sims.append({"pair": [names[i], names[j]], "similarity": round(sim, 4)})

    explanation = generate_explanation_with_providers(conflicts)

    return jsonify({"conflicts": conflicts, "similarities": sims, "explanation": explanation})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
