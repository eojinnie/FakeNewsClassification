# 📰 FakeNewsClassification

This project is focused on building a robust fake news classification model using a pre-trained DistilBERT transformer model, implemented with TensorFlow and fine-tuned on a labeled dataset. The classifier identifies whether a given text is fake or real news, handling imbalanced data and optimizing the model's performance.

## 🎯 Project Overview

- **Goal**: To classify news articles as either *fake* or *real*.
- **Model**: Fine-tuned DistilBERT model.
- **Dataset**: [WELFake Dataset]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)) on Kaggle.
- **Challenges Addressed**:
  - Imbalanced data handling using SMOTE and class weighting.
  - Hyperparameter tuning for improved precision and recall.
  - Jupyter Notebook processing for seamless integration with GitHub.

## 📂 Files
- `notebook.ipynb` : Main notebook file containing data preprocessing, model training, and evaluation.
- `README.md` : Project overview, usage instructions, and requirements.

## 🛠️ Installation

Clone the repository and install dependencies in a virtual environment.

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🏗️ Project Setup and Requirements

1. **Dependencies**: Required libraries include tensorflow, transformers, scikit-learn, imbalanced-learn, matplotlib, and seaborn. Install all with:

`bash
pip install tensorflow transformers scikit-learn imbalanced-learn matplotlib seaborn
`
3. **Dataset**: Download the dataset from Kaggle and place it in the root directory, or specify the correct path in notebook.ipynb.

## 🔍 Model Training and Evaluation

1. **Data Preprocessing**: The text data is tokenized with the DistilBERT tokenizer, truncated to a maximum length of 100 tokens.
2. **SMOTE and Class Weighting**: Applied to manage class imbalance, improving recall and precision.
3. **Model Architecture**: Fine-tuned DistilBERT model with dropout layers for regularization.
4. **Training**: The model is trained with balanced cross-entropy loss and an Adam optimizer.

## 🚀 Running the Notebook

Open the Jupyter Notebook to explore, train, and evaluate the model:

`bash
jupyter notebook notebook.ipynb
`

## 💻 Example Usage

After training, load the saved model and make predictions as shown below:

```python
import pickle
from transformers import DistilBertTokenizer
import tensorflow as tf

# Load model and tokenizer info
with open('./model/info.pkl', 'rb') as f:
    MODEL_NAME, MAX_LEN = pickle.load(f)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = tf.keras.models.load_model('./model/clf.keras')

# Sample prediction
text = "Sample news article text to classify."
inputs = tokenizer(text, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors="tf")
logits = model.predict([inputs["input_ids"], inputs["attention_mask"]])
print("Predicted class:", tf.argmax(logits, axis=1).numpy()[0])
```

## 📊 Results

The model achieved a balanced accuracy of approximately 83% on the validation set. Performance can be improved with further tuning.

**Evaluation Metrics**

| Metric    | Score    | Score    | Score    | Score    |
|-----------|----------|----------|----------|----------|
| Precision | 0.79     | 0.87     | 0.83     | 0.83     | 
| Recall    | 0.88     | 0.78     | 0.83     | 0.83     |
| F1 Score  | 0.83     | 0.82     | 0.83     | 0.83     | 

## 🔮 Future Improvements

- Experiment with other oversampling techniques.
- Test additional transformer architectures for improved accuracy.
- Further optimize batch size and learning rate to reduce overfitting.



























