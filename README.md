# Sentiment analysis of Amazon reviews

ISY503 Intelligent Systems, Assessment 3 (NLP project).

A neural network that reads a product review and answers **"Positive review"** or **"Negative review"**.

## Results

| Model | Validation accuracy | Test accuracy |
|---|---|---|
| TF-IDF + logistic regression (baseline) | 86.6% | 89.1% |
| ELECTRA-small + our classifier | 94.3% | 92.9% |

The test set has 786 reviews, so one run can differ from another by about 2 points.

On 68 short phrases we wrote ourselves, the network is right in 90% of cases and the baseline in 72%.
Sarcasm is the weak spot: 6 of 12 phrases.

## Presentation

https://laustu-my.sharepoint.com/:p:/g/personal/irina_pugaeva_student_torrens_edu_au/cQrShj-n-j--SJz8vJ6c-ppJEgUDqupiOz5QX54UnYdhcoviFg


## What is in the repository

| File | What it is |
|---|---|
| `sentiment_bert.ipynb` | The whole project: data, baseline, training, testing, fairness checks |
| `model/` | The trained model (`sentiment_classifier.pt`) and its tokenizer |
| `requirements.txt` | Python packages |

Web page: [to be added]

## How to run

**Google Colab.** Upload `sentiment_bert.ipynb`, choose a GPU runtime and run all cells.
The notebook downloads the dataset by itself.

**On your own computer.**

```bash
pip install -r requirements.txt
```

Then open `sentiment_bert.ipynb` in Jupyter or VS Code and run all cells.
Training takes about an hour on an Apple Silicon laptop.

## How it works

1. **Data.** 8,000 labelled reviews from four categories (books, DVDs, electronics, kitchen). After removing duplicates and very short reviews: 7,851, split 80 / 10 / 10 into train, validation and test.
2. **More training data.** 8,000 extra reviews from the `unlabeled.review` files (labelled by their star rating) and 10,921 review titles as short examples. Validation and test sets are not touched.
3. **Model.** The pre-trained [`google/electra-small-discriminator`](https://huggingface.co/google/electra-small-discriminator) encoder with our own classification layer on top (class `SentimentClassifier`).
4. **Training.** 4 epochs with AdamW; the epoch with the best validation accuracy is saved.
5. **Checks.** Error analysis, our own short phrases, fairness tests (names, brands, typos) and word highlighting.

To use the saved model outside the notebook you need the `SentimentClassifier` class, `clean_text` and `predict` from the notebook.

## Limitations

- The model has only two answers, so a neutral sentence still gets a confident "positive" or "negative".
- Labels come from star ratings, not from reading the text.
- In our fairness test the score changed with a person's name and with a brand.

## Data

Multi-Domain Sentiment Dataset (Blitzer, Dredze & Pereira, 2007):
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

## Team

- Irina Pugaeva
- Sovannara Hourn
- Aidana Makash
