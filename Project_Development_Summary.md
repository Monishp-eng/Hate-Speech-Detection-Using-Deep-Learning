# Project Implementation Summary: Hate Speech Detection System

This document outlines the complete, end-to-end development of the Hate Speech Detection system. It details the transformation from a raw PyTorch deep learning architecture into a robust, full-stack web application with persistent storage, a live analytics dashboard, and automated grading pipelines. This content is structured for direct inclusion in the "Implementation," "System Architecture," or "Methodology" sections of your NNDL project report.

---

## 1. Dataset Preparation & Model Training
A robust machine learning pipeline requires high-quality data and a rigorous training protocol.

*   **Dataset Overview (`labeled_data.csv`)**: 
    *   The system was trained on a comprehensive Twitter dataset (often based on the Davidson et al. Hate Speech dataset), which consists of thousands of raw text samples manually annotated via crowdsourcing.
    *   The dataset is annotated into three distinct target classes: `0: Hate Speech`, `1: Offensive Language`, and `2: Neither (Neutral)`. This poses a complex multi-class classification problem with inherent class imbalance (as offensive language typically outnumbers explicit hate speech).
*   **Data Preprocessing (`utils/preprocessing.py`)**: 
    *   Raw text was thoroughly cleaned prior to tokenization. Steps included converting text to lowercase, stripping URLs, removing Twitter mentions (`@user`), removing punctuation, and filtering out stop words.
    *   Text sequences were tokenized and mapped to a vocabulary index, then padded or truncated to a fixed maximum sequence length (`MAX_LEN`) to ensure uniform tensor dimensions for the neural network.
*   **Training Phase (`train.py`)**:
    *   The dataset was split into Training (e.g., 80%), Validation (10%), and Testing (10%) sets.
    *   The model was trained using the **Adam (or AdamW)** optimizer coupled with a **Cross-Entropy Loss** function. During the epochs, a learning rate scheduler and early stopping were implemented based on validation loss to prevent overfitting on the noise-heavy text data.
*   **Testing Phase (`evaluate.py`)**:
    *   Post-training, the model was strictly evaluated on the isolated Test Set to determine its real-world generalization. 
    *   We specifically analyzed instances where the model confused "Offensive Language" (like casual swearing/slang) with true "Hate Speech," utilizing confusion matrices to map out the network's discriminatory boundaries.

## 2. Machine Learning Integration & NLP Inference Pipeline
The core intelligence of the system relies on the custom `BiLSTM + Attention` architecture trained entirely in PyTorch. The live inference pipeline involves taking raw user input, normalizing it, converting it into tokenized tensors, and pushing it through the neural network.

*   **Custom Inference Engine (`predict.py`)**: 
    *   Designed the `HateSpeechPredictor` class as a singleton to load the trained weights (`best_bilstm_attention.pth`) precisely once upon server initialization.
    *   **Computation Constraints**: Configured the model strictly into `.eval()` mode and executed all forward passes inside a `with torch.no_grad():` context block. This suppresses gradient tracking and drastically reduces memory overhead, simulating a production-grade inference environment.
    *   **Probability Mapping**: Applied Softmax activation to the model's raw output logits to map values into probability distributions (0 to 1). This allows the system to output exact percentage confidences across three specific labels: `Hate Speech`, `Offensive Language`, and `Neutral Text`.

*   **Offline Academic Evaluation (`evaluate.py` & `error_analysis.py`)**:
    *   To prepare the exact metrics needed for an NNDL presentation, dedicated offline metrics scripts were written.
    *   Utilized `sklearn.metrics` to automatically compute standard academic thresholds: **Accuracy, Macro/Weighted F1 Score, Precision, and Recall**.
    *   Designed an Error Analysis module to systematically log the "Top False Positives" and "Top False Negatives" so misclassifications—such as confusing modern slang or sarcasm with genuine hate speech—can be scientifically documented in the final report.

## 3. Backend Server & API Architecture (Flask)
A lightweight, high-performance web backend was built to decouple the heavy Machine Learning logic from the user interface.

*   **RESTful Routing Architecture**: Used Flask to establish asynchronous endpoints communicating via JSON objects:
    *   `POST /predict`: Receives text inputs, sanitizes them, and calls the PyTorch engine. Returns classification tags and probability arrays in milliseconds.
    *   `GET /analytics`: Queries the SQLite database in real-time, computing aggregate statistics (Total Predictions, Global Average Confidence) and grouping labels to feed the dashboard charts.
    *   `GET /history`: Returns a descending slice of the most recent user inputs for historical tracking.
    *   `DELETE /history`: A secure data-destruction route that instantly drops the database table and resets local charts, ensuring graders can have a fresh slate.

*   **SQLite Relational Database Persistence**:
    *   Engineered a seamless initialization protocol (`init_db()`) that automatically creates a hidden `.db` file at runtime if it doesn't already exist.
    *   **Schema**: Designed a highly normalized table: `id (Integer), text (Text), prediction (Text), confidence (Real), probabilities (JSON Text), timestamp (Datetime)`.
    *   This guarantees that historical user interactions and statistical analytics persist across server reboots or system crashes.

*   **Data Export API (`GET /export`)**: 
    *   Implemented a local CSV extraction tool inside the backend. It uses python's native `csv` stream processor to allow researchers to dump the full SQLite database out into standard DataFrame structures for Jupyter Notebook exploration. 

## 4. Frontend Application & Dashboard (UI/UX)
The frontend was developed using exclusively Vanilla languages (HTML5, CSS3, JavaScript) to remain lightweight, lightning-fast, and entirely local-host compliant without requiring Node.js packet managers.

*   **Asynchronous DOM Manipulation (`script.js`)**: 
    *   Replaced native HTML form submissions with modern `fetch()` asynchronous logic. This allows the backend classification, database insertion, and charting to happen seamlessly in the background without causing page-reloads.
    *   Added keyboard event listeners (`Ctrl + Enter`) for rapid grading efficiency.

*   **Dynamic Visualization with Chart.js**:
    *   Embedded a third-party `Chart.js` library to graphically map out the `/analytics` data.
    *   Built a responsive Doughnut Chart that updates instantly dynamically mapping classifications into assigned colors: Red (Hate), Orange (Offense), and Green (Neutral).

*   **Dual-Column Responsive Interface (`index.html` & `style.css`)**:
    *   Implemented CSS Flexbox & CSS Grid to design a modern, two-column layout. The left pane provides direct testing interactions with custom error-handling UI tooltips, while the right pane holds overarching history and statistical graphs.
    *   **Theming**: Built an elegant Dark / Light Mode toggle tied to CSS variables (e.g., `--bg-color`, `--text-main`). State adjustments apply dynamically down to the axes inside the active Chart.js canvas.

## 5. Academic Packaging & Deployment Automation
Given that college project submissions are frequently docked points for "failure to run," massive attention was placed on ensuring robust, one-click execution chains for teaching assistants.

*   **Zero-Friction Grading Scripts (`setup.sh` & `setup.bat`)**:
    *   Engineered dual cross-platform Bash and PowerShell automation scripts.
    *   These scripts automatically run the following waterfall when clicked: 
        1. Initialize an empty standard `.venv` virtual environment.
        2. Enter the active execution sub-shell.
        3. Upgrade `pip` and force-install strictly matched versioning from `requirements.txt`.
        4. Validate the filesystem to ensure the massive `.pth` PyTorch file is actually present in `/models/saved_models/` before starting, logging an immediate diagnostic warning if the user forgot to transfer the files.
        5. Automatically invoke the local Flask server on Port 5000.

*   **Mathematical & Structural Documentation**: 
    *   Injected completely standardized `Mermaid.js` topological diagrams into the documentation to visualize the Forward Pass from inputs to Softmax output logits.
    *   Written clear academic documentation rationalizing architectural decisions—specifically analyzing why Feed-Forward networks fail this dataset and why Bidirectional LSTMs paired with Attention Mechanics are required to track toxic triggers across extended sequencing.