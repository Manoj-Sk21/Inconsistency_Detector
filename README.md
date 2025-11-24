AI-Powered Inconsistency Detector for Police Case Analysis

A Gen-AI–driven system that detects inconsistencies across witness statements, FIRs, and CCTV summaries to help investigators quickly spot contradictions and missing information.

🚀 Overview

The Inconsistency Detector is an AI-powered tool designed to help law enforcement officers and investigators analyze multiple case-related text inputs. It automatically identifies mismatches in:

🕒 Timelines

🎽 Suspect appearance

🛵 Vehicle details (ID/plate)

🧍 Person/entity references

❗ Missing or incomplete information

The system provides two versions of analysis:

Version A (Simple View): Concise summary for quick field decisions

Version B (Detailed View): Full analysis with structured conflicts, similarity scores, and explanations

A PDF report can be exported for documentation.

🎯 Problem Statement

Investigators spend significant time manually comparing witness statements, FIR entries, and CCTV summaries.
This leads to:

High chances of human error

Delays in identifying contradictions

Difficulty in handling unstructured text

No automated verification support

💡 Solution

A Gen-AI–supported system that:

Reads and understands case-related unstructured text

Extracts timestamps, entities, colors, vehicle numbers & more

Compares statements against each other

Highlights contradictions instantly

Generates simple and detailed reports

Allows exporting a PDF summary

This significantly reduces manual effort and improves case accuracy.

🧠 Use of Gen-AI

This project uses LLaMA via Groq API for:

Natural language understanding

Reasoning about contradictions

Generating explanations

Producing summaries for Version A & Version B

Traditional rule-based systems can’t interpret narrative text—Gen AI gives near-human reasoning capability.

🛠️ Tech Stack
Frontend

HTML, CSS, JavaScript

Responsive UI

Version toggle (A/B)

Dynamic rendering of results

Backend

Flask (Python)

REST APIs: /analyze, /export_pdf

PDF generation using ReportLab

AI/NLP

Groq LLaMA API (main reasoning model)

spaCy (entity extraction)

sentence-transformers (semantic similarities)

dateparser (timestamp extraction)

📁 Project Structure
Inconsistency_Detector/
│── app.py                # Flask backend + AI logic
│── templates/
│     └── index.html      # Frontend UI
│── static/               # CSS/JS (if added)
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── venv/                 # Virtual environment (ignored in GitHub)

🔧 Setup & Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Inconsistency_Detector.git
cd Inconsistency_Detector

2️⃣ Create Virtual Environment
python -m venv venv

3️⃣ Activate Virtual Environment
Windows:
.\venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Add Groq API Key

Create a .env file:

GROQ_API_KEY=your_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

▶️ Run the Application
python app.py


Open browser:

http://127.0.0.1:5000

🧪 Example Inputs
Witness 1: I saw a man wearing a red shirt at 8:30 PM riding a vehicle KA 01 AB 1234.
Witness 2: Person wore a black shirt and passed around 8:52 PM on a different bike.
FIR: Suspect wore a red shirt around 8:30 PM. Vehicle KA01AB1234 reported.
CCTV: Footage at 20:52 shows a person walking fast. Shirt unclear. No visible plate.

📤 PDF Export

Click Export PDF

Generates report based on Version A or Version B

Includes conflicts + summaries + case insights

🎛️ Features

✔ Time mismatch detection
✔ Appearance mismatch detection
✔ Vehicle/plate mismatch detection
✔ Missing detail detection
✔ Similarity scoring
✔ Simple vs Detailed view
✔ PDF Export
✔ Clean UI for investigators

📌 Limitations

Works on text only (no images/videos)

Accuracy depends on input quality

Not a replacement for human investigation

🌟 Future Enhancements

Multi-language support

Voice-to-text for witness recording

Integration with police case management systems

CCTV frame extraction and image-to-text analysis

ML fine-tuning using real-world datasets

🤝 Contributors

You (Project Lead, AIML Developer)

Manoj (Tester / Collaborator)
