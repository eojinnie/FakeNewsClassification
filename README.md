# FakeNewsClassification

This project is focused on building a robust fake news classification model using a pre-trained DistilBERT transformer model, implemented with TensorFlow and fine-tuned on a labeled dataset. The classifier identifies whether a given text is fake or real news, handling imbalanced data and optimizing the model's performance.

## Project Overview

- **Goal**: To classify news articles as either *fake* or *real*.
- **Model**: Fine-tuned DistilBERT model.
- **Dataset**: [WELFake Dataset]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)) on Kaggle.
- **Challenges Addressed**:
  - Imbalanced data handling using SMOTE and class weighting.
  - Hyperparameter tuning for improved precision and recall.
  - Jupyter Notebook processing for seamless integration with GitHub.

## Files
- `notebook.ipynb` : Main notebook file containing data preprocessing, model training, and evaluation.
- `README.md` : Project overview, usage instructions, and requirements.

## Installation

Clone the repository and install dependencies in a virtual environment.

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
