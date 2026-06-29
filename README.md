# Hate Speech Detection (NNDL Project)

A complete from-scratch Deep Learning approach to detect textual boundaries between Hate Speech, Offensive Language, and Neutral Content.

## 🧠 Deep Learning Architecture: BiLSTM + Attention

This project implements a **Bidirectional Long Short-Term Memory (BiLSTM)** network supplemented by a custom **Attention Mechanism**.
We process textual artifacts recursively (forwards & backwards) extracting temporal context before a learned Attention Matrix assigns weights evaluating each state's relevance towards hateful sentiment analysis. 

---

## 📁 Project Structure

```
hate_speech_detection/
├── app/               
│   └── app.py         # Flask Web Server & API Root
├── templates/         
│   └── index.html     # Frontend UI Template
├── static/            
│   ├── style.css      # Custom UI Styling
│   └── script.js      # Frontend AJAX & Dynamic UI handling
├── config/            # Hyperparameters and path configurations
├── data/              # Stores raw `.csv` and `processed` tokenizers
├── models/            # PyTorch `nn.Module` classes and training loops
├── utils/             # DataLoaders, Tokenizer, Metrics, Text cleanup
├── predict.py         # Local CLI interactive Inference wrapper
├── evaluate.py        # Model evaluation and metrics module
├── error_analysis.py  # Error analysis module showing misclassifications
├── requirements.txt   # Python prerequisites
└── README.md          # Project Docs
```

---

## ⚙️ Setup & Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Web App (Full-Stack)

We have built a sleek, dark-themed frontend directly connected to the deep learning model.

1. Start the Flask application natively:
```bash
python app/app.py
```
2. Open your web browser to the provided local URL:
```text
http://127.0.0.1:5000
```
3. Type out explicit sentences or use the example buttons provided in the UI! 
   The frontend uses `fetch` API via AJAX to asynchronously contact `/predict` running the PyTorch tensor evaluations.

---

## 🧪 Localhost Inference (CLI Mode)

Don't want to use the browser? Run the CLI script to evaluate textual inputs directly inside terminal.

```bash
python predict.py
```

**Example Output:**
```text
Enter text: I hate you
Prediction: Hate Speech
Confidence: 0.9234
Class Probabilities:
  "Hate Speech": 0.9234
  "Offensive Language": 0.0521
  "Neutral": 0.0245
```

---

## 📊 Evaluation & Diagnostics

Run internal checks against the test datasets natively:

1. **Evaluate Model Overall Performance:**
```bash
python evaluate.py
```

2. **Analyze Errors / False Positives:**
```bash
python error_analysis.py
```

