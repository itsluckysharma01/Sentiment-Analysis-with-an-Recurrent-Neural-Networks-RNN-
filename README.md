# Sentiment Analysis with a Recurrent Neural Network (RNN)

An end-to-end NLP project that classifies Swiggy customer reviews as Positive or Negative using a TensorFlow SimpleRNN model.

## Quick Links

- [Project Snapshot](#project-snapshot)
- [Demo Workflow](#demo-workflow)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Run This Project](#run-this-project)
- [Try Your Own Review](#try-your-own-review)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Project Snapshot

| Item        | Details                                                            |
| ----------- | ------------------------------------------------------------------ |
| Goal        | Binary sentiment classification (Positive/Negative)                |
| Model       | `Embedding -> SimpleRNN -> Dense(sigmoid)`                         |
| Framework   | TensorFlow / Keras                                                 |
| Data Source | Swiggy customer reviews (`swiggy.csv`)                             |
| Notebook    | `Sentiment Analysis with an Recurrent Neural Networks (RNN).ipynb` |

<details>
<summary><strong>What this project does</strong></summary>

- Loads review data directly from a public CSV URL.
- Cleans and normalizes text.
- Creates sentiment labels from ratings.
- Tokenizes and pads text sequences.
- Trains an RNN-based binary classifier.
- Predicts sentiment for custom review text.

</details>

## Demo Workflow

```text
Review Text
	 -> Text Cleaning
	 -> Tokenization
	 -> Padding
	 -> Embedding + SimpleRNN
	 -> Sigmoid Output
	 -> Positive / Negative
```

<details>
<summary><strong>Cell-by-cell notebook flow</strong></summary>

1. Import libraries (`pandas`, `numpy`, `sklearn`, `tensorflow.keras`).
2. Load dataset from GitHub raw URL.
3. Clean review text and generate sentiment label.
4. Tokenize text and apply sequence padding.
5. Split data for training, validation, and test.
6. Build and compile the SimpleRNN model.
7. Train and evaluate the model.
8. Run prediction on custom sample reviews.

</details>

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Jupyter Notebook

## Dataset

- Source: `https://raw.githubusercontent.com/itsluckysharma01/Datasets/refs/heads/main/swiggy.csv`
- Used fields:
  - `Review`
  - `Avg Rating`
- Label rule:
  - `Avg Rating >= 3.5 -> Positive (1)`
  - `Avg Rating < 3.5 -> Negative (0)`

<details>
<summary><strong>Preprocessing used in notebook</strong></summary>

- Lowercasing review text.
- Removing non-alphanumeric characters.
- Dropping missing values.
- Tokenizer vocabulary size: `5000`.
- Sequence length: `200`.

</details>

## Model Pipeline

```python
model = Sequential([
		Embedding(input_dim=5000, output_dim=16, input_length=200),
		SimpleRNN(32, activation="tanh", return_sequences=False),
		Dense(1, activation="sigmoid")
])
```

- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`
- Training setup: `epochs=15`, `batch_size=32`

## Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/itsluckysharma01/Sentiment-Analysis-with-an-Recurrent-Neural-Networks-RNN-.git
cd "Sentiment-Analysis-with-an-Recurrent-Neural-Networks-RNN-"
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow notebook
```

### 4. Launch notebook

```bash
jupyter notebook
```

Open:

- `Sentiment Analysis with an Recurrent Neural Networks (RNN).ipynb`

## Try Your Own Review

After training, call the prediction function with any text:

```python
sample_review = "The food was great"
print(predict_sentiment(sample_review))
```

<details>
<summary><strong>More sample inputs</strong></summary>

- `"Delivery was too late and food was cold"`
- `"Amazing taste and quick service"`
- `"Not worth the money"`

## Roadmap

- [ ] Add confusion matrix and classification report.
- [ ] Compare `SimpleRNN` with `LSTM` and `GRU`.
- [ ] Save and reload trained model for inference.
- [ ] Build a small web app for live sentiment prediction.

## Contributing

Contributions are welcome. If you want to improve preprocessing, model quality, or deployment, feel free to open an issue or a pull request.

## Author

- GitHub: [itsluckysharma01](https://github.com/itsluckysharma01)
