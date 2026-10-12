# 🇵🇰 Urdu Sentiment Analysis using Multilingual BERT

### Natural Language Processing • Urdu Text Analysis • Sentiment Classification 🚀

A Python-based project exploring sentiment analysis for Urdu, Roman Urdu, and code-mixed text. The repository includes a sentiment-prediction script, an Urdu sentiment corpus, and a spreadsheet version of the dataset.

The project demonstrates lexicon-based sentiment prediction, text-type detection, terminal-based reporting, and visual summaries. It also contains illustrative training and model-comparison outputs that should be distinguished from independently verified machine-learning experiments.

---

## 🎯 About the Project

Sentiment analysis is a Natural Language Processing (NLP) task that identifies the emotional polarity expressed in text. For Urdu, this task presents additional challenges because users communicate in Urdu script, Roman Urdu, English, and mixed-language forms.

This project explores a workflow for analyzing sentiment in Urdu-language text and provides a foundation for further research using multilingual transformer models.

### 🎯 Project Objectives

- 🇵🇰 Explore sentiment analysis for Urdu-language text.
- 🔤 Examine Urdu, Roman Urdu, and code-mixed text.
- 🧠 Investigate multilingual NLP and transformer-based approaches.
- 📊 Display sentiment predictions and class distributions.
- 📈 Visualize illustrative model-training and comparison information.
- 🔬 Establish a foundation for reproducible Urdu NLP experiments.

> **Implementation note:** The current Python script uses a lexicon-based predictor for interactive sentiment classification. Its mBERT and XLM-RoBERTa training logs and comparison figures are hard-coded demonstrations, not evidence of actual model fine-tuning or independently measured performance.

---

## ✨ Key Features

### 💬 Urdu Sentiment Prediction

The `predict_sentiment()` function uses predefined positive, negative, and neutral word lists to estimate a sentiment category and return normalized scores.

### 🔤 Urdu and Roman Urdu Text Analysis

The script includes a heuristic for categorizing text as Pure Urdu, Roman Urdu, or code-mixed based on the proportion of English-alphabet tokens.

### 🧾 Dataset Loading

The project reads a tab-separated corpus containing text and class labels. If the configured dataset path cannot be found, the script falls back to a small synthetic demonstration dataset.

### 🖥️ Interactive Terminal Interface

The program uses Rich to display formatted tables, panels, progress indicators, and interactive sentiment predictions.

### 📊 Visualization Dashboard

The script generates a dashboard containing class distributions, illustrative training curves, model-comparison charts, a confusion matrix, and other demonstration statistics.

**Important:** Several displayed metrics and visualizations are hard-coded in the current script. They should not be interpreted as verified experimental results.

---

## 🧠 System Architecture

The following diagram summarizes the workflow represented by the current script.

```text
              📄 Urdu Sentiment Corpus
                         │
                         ▼
                  📥 Data Loading
                         │
                         ▼
                🧹 Data Inspection
                         │
                         ▼
               🔤 Text-Type Detection
                ┌────────┼────────┐
                ▼        ▼        ▼
              Urdu    Roman Urdu  Code-Mixed
                └────────┼────────┘
                         ▼
               💬 Sentiment Prediction
                  (Lexicon-Based)
                         │
                         ▼
                📊 Prediction Scores
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       🖥️ Terminal Output      📈 Dashboard
```

The script also displays a separate illustrative pipeline for augmentation, mBERT training, model comparison, error analysis, and API output. Those displayed stages should not be assumed to perform actual training or host a working API in the current implementation.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Dataset loading and tabular data handling |
| NumPy | Numerical operations and visualization data |
| Matplotlib | Charts and dashboard generation |
| Seaborn | Heatmap visualization |
| Rich | Terminal interface and formatted output |
| Regular expressions | Text matching and heuristic analysis |
| TSV / XLS | Dataset storage formats |
| Git and GitHub | Version control and project sharing |

The current script does not import the Hugging Face Transformers library or implement an actual mBERT/XLM-RoBERTa training loop. Those are potential future extensions.

---

## 📁 Project Structure

```text
Urdu-Sentiment-Analysis-using-Multilingual-BERT/
│
├── README.md
│
├── urdu_sentiment.py
│   ├── Lexicon-based sentiment predictor
│   ├── Urdu / Roman Urdu text-type heuristic
│   ├── Dataset loader and fallback demo data
│   ├── Terminal reporting
│   └── Demonstration dashboard
│
├── urdu-sentiment-corpus-v1.tsv
│   └── Tab-separated text and class labels
│
└── urdu-sentiment-corpus-v1.xls
    └── Spreadsheet version of the corpus
```

The dashboard image is generated when the script runs successfully. Other generated files may depend on the execution environment.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/basil531-hub/Urdu-Sentiment-Analysis-using-Multilingual-BERT.git
```

Navigate to the project directory:

```bash
cd Urdu-Sentiment-Analysis-using-Multilingual-BERT
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

The current script imports pandas, NumPy, Matplotlib, Seaborn, and Rich.

Install these packages with:

```bash
python -m pip install pandas numpy matplotlib seaborn rich
```

The repository does not currently provide a pinned dependency file. The command above installs the direct dependencies visible in the script; package versions have not been independently tested or pinned.

### 4. Configure the Dataset Path

The script currently defines a dataset path pointing to a specific Windows drive:

```python
DATASET = r"U:\4th Semester\Prog for AI Lab\Project\urdu-sentiment-corpus-v1.tsv"
```

This path will not work on most other computers.

Update it to point to the TSV file in your cloned repository. For example:

```python
from pathlib import Path

DATASET = str(
    Path(__file__).resolve().parent
    / "urdu-sentiment-corpus-v1.tsv"
)
```

This change allows the script to locate the corpus relative to the project directory.

### 5. Run the Program

```bash
python urdu_sentiment.py
```

The program displays its terminal-based pipeline, generates a visualization dashboard, and then opens an interactive prompt for sentiment prediction.

To exit the interactive prompt, enter:

```text
quit
```

**Note:** The script's displayed training logs and model-comparison results are illustrative. Running the script does not independently train multilingual BERT or reproduce those figures through model evaluation.

---

## 📚 Dataset Documentation

### Urdu Sentiment Corpus

The repository includes two versions of the corpus:

| File | Format | Intended use |
|---|---|---|
| `urdu-sentiment-corpus-v1.tsv` | Tab-separated text | Programmatic data loading |
| `urdu-sentiment-corpus-v1.xls` | Excel-compatible spreadsheet | Manual inspection |

The script expects two columns when loading the TSV file:

- `Tweet` — the text sample
- `Class` — the sentiment label

The current code uses the following label mapping:

| Label | Meaning in the script |
|---|---|
| `P` | Positive |
| `N` | Negative |
| `O` | Neutral |

These meanings are taken from the script's class-display logic. The corpus annotation guidelines and the correctness of individual labels should be verified separately.

### Dataset Preparation

Before using the corpus for a research experiment, document:

- Number of samples and unique records
- Class distribution and imbalance
- Source and collection method
- Annotation procedure and label definitions
- Duplicate and missing-value handling
- Language distribution: Urdu script, Roman Urdu, and code-mixed text
- Dataset license and permitted use
- Training, validation, and test split strategy

Do not report corpus statistics until they have been calculated directly from the dataset.

---

## 🔍 How Sentiment Prediction Works

The current `predict_sentiment()` function uses manually defined word lists.

### 1. Lexicon Matching

The script contains positive, negative, and neutral word sets with examples in Urdu script, Roman Urdu, and English.

### 2. Score Calculation

The predictor counts matching terms and calculates normalized scores for three categories:

- `POSITIVE`
- `NEGATIVE`
- `NEUTRAL`

### 3. Prediction

The class with the largest calculated score is returned as the prediction.

### 4. Text-Type Heuristic

The `text_type()` function estimates whether the text is Pure Urdu, Roman Urdu, or code-mixed using English-letter token proportions.

### Limitations of This Approach

A lexicon-based predictor is not equivalent to a fine-tuned transformer. It can struggle with sarcasm, context, negation, spelling variations, domain-specific language, and words whose sentiment changes with context.

The returned scores are heuristic scores, not calibrated probabilities of correctness.

---

## 📊 Evaluation and Results

The script displays example training logs, model-comparison figures, error rates, a confusion matrix, and a sample prediction confidence chart.

However, those values are embedded in the source code rather than computed from an actual model evaluation pipeline. They must not be presented as measured research results.

### Recommended Evaluation Protocol

A reproducible multilingual sentiment-classification experiment should include:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Correctness of positive predictions for each class |
| Recall | Proportion of examples correctly identified for each class |
| Macro F1-score | Balanced summary across sentiment classes |
| Confusion matrix | Breakdown of correct and incorrect classifications |
| Per-language performance | Comparison across Urdu, Roman Urdu, and code-mixed samples |

### Recommended Experimental Setup

1. Inspect and validate the corpus.
2. Define a documented training, validation, and test split.
3. Prevent duplicate or augmented versions of the same sample from leaking across splits.
4. Establish a lexicon-based baseline.
5. Fine-tune an actual multilingual transformer model, such as `bert-base-multilingual-cased`, using a reproducible training pipeline.
6. Evaluate all models on the same held-out test set.
7. Record the configuration, package versions, random seeds, and actual metric outputs.

No verified model accuracy, F1-score, or benchmark result is claimed in this README.

---

## 🔐 Responsible Use and Data Considerations

Sentiment analysis can misinterpret opinions, dialects, informal spelling, sarcasm, political expressions, and culturally specific language.

For responsible experimentation:

- Check the corpus license and usage restrictions.
- Remove personal information where appropriate.
- Document known dataset biases.
- Avoid treating heuristic scores as certainty.
- Evaluate performance across Urdu and Roman Urdu text.
- Do not use unvalidated predictions as the sole basis for consequential decisions.

---

## ⚠️ Current Limitations

The current implementation has several limitations that should be understood before using it for research:

- The interactive predictor is lexicon-based rather than transformer-based.
- The mBERT and XLM-RoBERTa training logs and comparison tables are hard-coded demonstrations.
- The displayed confusion matrix and error rates are illustrative, not computed from verified predictions.
- The dataset path is machine-specific and needs to be configured.
- The fallback dataset contains repeated synthetic examples and is not a substitute for the original corpus.
- No pinned dependency file or reproducible transformer-training configuration is included.
- The heuristic text-type detector may misclassify short or heavily code-mixed samples.
- Prediction scores are not calibrated probabilities.

These limitations should be addressed before describing the project as a validated multilingual BERT model.

---

## 🔮 Future Improvements

- [ ] Replace the hard-coded training demonstrations with a real training pipeline.
- [ ] Integrate Hugging Face Transformers for mBERT fine-tuning.
- [ ] Add an XLM-RoBERTa baseline with a documented configuration.
- [ ] Implement configurable dataset paths and input validation.
- [ ] Add a pinned `requirements.txt` or equivalent dependency file.
- [ ] Calculate metrics from actual held-out predictions.
- [ ] Add reproducible training and evaluation scripts.
- [ ] Document corpus provenance, licensing, and annotation guidelines.
- [ ] Improve Urdu normalization and Roman Urdu handling.
- [ ] Evaluate sarcasm, negation, and code-mixed expressions.
- [ ] Add automated tests for data loading and prediction behavior.
- [ ] Implement and test a real API only if deployment is required.

---

## 🎓 Research Relevance

This project connects to several areas of Natural Language Processing research:

- Urdu Natural Language Processing
- Sentiment Analysis
- Multilingual Text Classification
- Roman Urdu Processing
- Code-Mixed Language Analysis
- Transformer-Based Language Models
- Dataset Quality and Evaluation
- Reproducible Machine Learning

It provides a starting point for developing and evaluating a genuine multilingual transformer-based sentiment-analysis system.

---

## 👨‍💻 Developer

**Muhammad Basel**

BS Computer Science  
The Islamia University of Bahawalpur, Pakistan

### Areas of Interest

- Artificial Intelligence and Machine Learning
- Natural Language Processing
- Computer Vision
- Multilingual AI
- Research and Software Development

### Connect

- **GitHub:** [basil531-hub](https://github.com/basil531-hub)
- **Project Repository:** [Urdu Sentiment Analysis using Multilingual BERT](https://github.com/basil531-hub/Urdu-Sentiment-Analysis-using-Multilingual-BERT)

---

## 📄 License

Review the repository's license information before redistributing or reusing the code and dataset. If no license is provided, do not assume that reuse or redistribution is permitted.

---

### 🚀 Understanding Language. Exploring Sentiment. Advancing Urdu NLP.

*An exploratory project focused on Urdu sentiment analysis and future multilingual NLP research.*

**Developed by Muhammad Basel**
