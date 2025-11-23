# app.py
"""
Inconsistency Detector — FULLY UPDATED VERSION (2025)
- Uses the new OpenAI Python SDK (OpenAI() client)
- Removes deprecated ChatCompletion calls
- Clean, stable, hackathon-ready
"""

import os
import re
import json
from flask import Flask, request, render_template, jsonify
import spacy
import dateparser
import dateparser.search
from sentence_transformers import SentenceTransformer, util

# NEW OpenAI client (correct modern version)
from openai import OpenAI

# Load NLP models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

# ---- Patterns ----
TIME_PATTERNS = [r"\b(\d{1,2}:\d{2})\b", r"\b(\d{1,2}\s?(AM|PM|am|pm))\b"]
VEHICLE_REGEX = r"\b([A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{1,4})\b"
COLOR_WORDS = ["red","black","white","blue","green","yellow","grey","gray","brown","orange","pink","purple"]
KEYWORDS = ["knife","gun","vehicle","bike","plate","helmet","weapon","two-wheeler"]


# ---- Extractors ----
def extract_entities(text):
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    places = [ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC","FAC")]
    times = [ent.text for ent in doc.ents if ent.label_ in ("DATE","TIME")]
    return {"persons": persons, "places": places, "times": times}


def extract_times(text):
    found = []
    for p in TIME_PATTERNS:
        found += re.findall(p, text)

    # Use dateparser for additional time detection
    dp = dateparser.search.search_dates(text)
    if dp:
        for t in dp:
            found.append(str(t[1]))

    cleaned = [t[0] if isinstance(t, tuple) else t for t in found]
    return list(set(cleaned))


def extract_colors(text):
    t = text.lower()
    return [c for c in COLOR_WORDS if c in t]


def extract_plates(text):
    matches = re.findall(VEHICLE_REGEX, text, flags=re.IGNORECASE)
    return [m.replace(" ", "").upper() for m in matches]


def extract_keywords(text):
    t = text.lower()
    return [k for k in KEYWORDS if k in t]


def embed_sim(a, b):
    e1 = embedder.encode(a, convert_to_tensor=True)
    e2 = embedder.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(e1, e2).item())


# ---- Conflict detection ----
def detect_conflicts(docs):
    conflicts = []

    # Entity mismatch (people)
    ents = {name: extract_entities(txt)['persons'] for name, txt in docs.items()}
    for a in docs:
        for b in docs:
            if a >= b: continue
            set_a = set([x.lower() for x in ents[a]])
            set_b = set([x.lower() for x in ents[b]])
            if set_a != set_b:
                conflicts.append({
                    "type": "PERSON_MISMATCH",
                    "between": [a, b],
                    "only_in_a": list(set_a - set_b),
                    "only_in_b": list(set_b - set_a)
                })

    # Time conflicts
    times = {name: extract_times(txt) for name, txt in docs.items()}
    for a in docs:
        for b in docs:
            if a >= b: continue
            if times[a] and times[b] and set(times[a]) != set(times[b]):
                conflicts.append({
                    "type": "TIME_MISMATCH",
                    "between": [a, b],
                    "times_a": times[a],
                    "times_b": times[b]
                })

    # Color mismatch
    colors = {name: extract_colors(txt) for name, txt in docs.items()}
    for a in docs:
        for b in docs:
            if a >= b: continue
            if colors[a] and colors[b] and set(colors[a]).isdisjoint(colors[b]):
                conflicts.append({
                    "type": "COLOR_MISMATCH",
                    "between": [a, b],
                    "colors_a": colors[a],
                    "colors_b": colors[b]
                })

    # Vehicle plate mismatch
    plates = {name: extract_plates(txt) for name, txt in docs.items()}
    for a in docs:
        for b in docs:
            if a >= b: continue
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
            present_in = [n for n, t in docs.items() if k in t.lower()]
            if len(present_in) != len(docs):
                conflicts.append({
                    "type": "MISSING_DETAIL",
                    "keyword": k,
                    "present_in": present_in
                })

    return conflicts


# ---- Modern OpenAI explanation ----
def generate_explanation(conflicts):
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return "(No API key found) " + rule_based(conflicts)

    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are an expert forensic AI system used by Indian Police.\n"
            "Explain these contradictions clearly and professionally.\n"
            "For each conflict, explain its meaning and give a confidence score (0–1).\n\n"
            f"{json.dumps(conflicts, indent=2)}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"(LLM error: {str(e)})\n\n" + rule_based(conflicts)


# ---- Fallback explanation ----
def rule_based(conflicts):
    lines = []
    for c in conflicts:
        if c["type"] == "TIME_MISMATCH":
            lines.append(
                f"Timeline mismatch between {c['between']}: {c['times_a']} vs {c['times_b']}."
            )
        elif c["type"] == "COLOR_MISMATCH":
            lines.append(
                f"Color mismatch between {c['between']}: {c['colors_a']} vs {c['colors_b']}."
            )
        elif c["type"] == "VEHICLE_MISMATCH":
            lines.append(
                f"Vehicle mismatch between {c['between']}: {c['plate_a']} vs {c['plate_b']}."
            )
        else:
            lines.append(str(c))
    return "\n".join(lines) if lines else "No contradictions detected."


# ---- Flask routes ----
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    docs = {
        "witness1": data.get("w1") or data.get("witness1", ""),
        "witness2": data.get("w2") or data.get("witness2", ""),
        "fir": data.get("fir", ""),
        "cctv": data.get("cctv", "")
    }

    conflicts = detect_conflicts(docs)

    # compute similarities
    names = list(docs.keys())
    sims = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            sim = embed_sim(docs[names[i]], docs[names[j]])
            sims.append({"pair": [names[i], names[j]], "similarity": round(sim, 4)})

    explanation = generate_explanation(conflicts)

    return jsonify({
        "conflicts": conflicts,
        "similarities": sims,
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(debug=True)

#test