# Emotion Sentiment Analysis with LSTM

A Streamlit-based app powered by a Bi-directional LSTM model that performs **multi-class emotion detection** on text data. It supports emotions such as **happiness**, **sadness**, **anger**, **fear**, **love**, and **surprise** and provides an interactive UI for prediction and visualization.

---

##  Features

- **Clean and tokenize** raw text inputs.
- **Pad sequences** to a fixed length with your tokenizer.
- Perform **emotion classification** using a pre-trained BiLSTM model.
- Visualize **confidence scores** via Plotly bar charts.
- Display **detailed probabilities** with progress bars.
- Offer **emotion-specific tips**:
  - e.g., if sadness is detected → suggest coping strategies;
  - if happiness → encourage savoring the moment.
- Provide a set of **clickable example phrases** to test the model quickly.

---

##  Getting Started

###  Prerequisites

Make sure you have:
- Python 3.7+
- Installed dependencies:
  ```bash
  pip install streamlit numpy pandas plotly tensorflow scikit-learn
````

### Repository Structure

```
emotion_sentiment_analysis_lstm/
├── app.py
├── model.h5            # or model.keras / saved_model/
├── tokenizer.pkl
├── label_encoder.pkl
├── Emotion_final.csv
└── asd.ipynb
```

### Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/furqank73/emotion_sentiment_analysis_lstm.git
   cd emotion_sentiment_analysis_lstm
   ```
2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Input your text, click **Predict Emotion**, and view results and suggestions!

---

## Tips & Suggestions (Emotion-Specific)

As showcased in the app, each detected emotion triggers tailored guidance:

* **Sadness**: "💙 Practice deep breathing", "📖 Journal your thoughts", "👥 Talk with someone."
* **Happiness**: "🎉 Savor the moment", "📝 Reflect on why you're happy", "🌟 Spread joy around you."
* (Can be extended to **anger**, **fear**, **surprise**, **love**, **neutral**, etc.)

---

## Screenshots

*(Add screenshots here to give users a visual overview)*
For example:

![App Screenshot](path/to/screenshot.png)

---


## Acknowledgements

* Built with **Streamlit**, **TensorFlow/Keras**, **Plotly**, **scikit-learn**.
* **furqank73** — for creating the initial model and repository. ([GitHub][1])

```

