# app.py — enhanced Streamlit app for emotion sentiment analysis with personalized tips
import streamlit as st
import os, pickle, traceback
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Choose caching decorator safely (compat with older/newer Streamlit)
try:
    cache_resource = st.cache_resource
except Exception:
    cache_resource = lambda f: st.cache(allow_output_mutation=True)(f)

st.set_page_config(
    page_title="Emotion Sentiment Analysis", 
    page_icon="😊", 
    layout="centered"
)

# ---------------------------
# Custom CSS for styling
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .tips-box {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        border-radius: 5px;
        padding: 15px;
        margin: 15px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def clean_text(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_emotion_tips(emotion, confidence):
    """Return personalized tips based on the predicted emotion"""
    tips = {
        "joy": [
            "🌟 Celebrate this positive moment!",
            "📝 Consider journaling about what's making you happy",
            "🤗 Share your joy with someone you care about",
            "🎉 Do something fun to extend this positive feeling",
            "🙏 Practice gratitude for these moments of happiness"
        ],
        "happiness": [
            "😊 Savor this moment of happiness",
            "📸 Capture this feeling by taking a mental picture",
            "🌞 Spend time outdoors to enhance your mood",
            "💝 Do something kind for someone else to spread the joy",
            "🎵 Listen to your favorite uplifting music"
        ],
        "sadness": [
            "🤗 It's okay to feel sad - acknowledge your emotions",
            "📞 Reach out to a friend or loved one for support",
            "🎨 Express your feelings through creative activities",
            "🌳 Take a walk in nature to help clear your mind",
            "📓 Journal about what's causing these feelings"
        ],
        "anger": [
            "🌬️ Take deep breaths to calm your nervous system",
            "🏃 Go for a walk or run to release pent-up energy",
            "🎯 Identify the source of your anger and address it constructively",
            "💧 Drink a glass of water and take a moment to pause",
            "🎵 Listen to calming music to help regulate emotions"
        ],
        "fear": [
            "🔍 Break down what's frightening you into manageable pieces",
            "🧘 Practice mindfulness to stay grounded in the present",
            "📝 Write down your fears to gain perspective on them",
            "💪 Remember past challenges you've overcome successfully",
            "🗣️ Talk to someone about what's making you anxious"
        ],
        "surprise": [
            "🔄 Take a moment to process this unexpected event",
            "🤔 Consider whether this surprise is positive or needs response",
            "📱 Share the surprising news with someone you trust",
            "⏸️ Pause before reacting to ensure thoughtful response",
            "🌅 View surprises as opportunities for new experiences"
        ],
        "love": [
            "💖 Express your feelings to the person you care about",
            "📞 Reach out to loved ones to strengthen your connections",
            "🎁 Do something special for someone you appreciate",
            "📸 Reminisce about positive memories with loved ones",
            "❤️ Practice self-love and appreciation too"
        ],
        "disgust": [
            "🚫 Distance yourself from the source of discomfort",
            "🌿 Focus on things that bring you peace and comfort",
            "🔄 Try to understand what triggered this reaction",
            "🍃 Clean or organize your space to create freshness",
            "🎨 Engage with beautiful art or nature to counterbalance"
        ]
    }
    
    # Default tips if emotion not found
    default_tips = [
        "📝 Journaling about your feelings can provide clarity",
        "🌳 Spending time in nature often helps regulate emotions",
        "💤 Ensure you're getting enough rest and self-care",
        "🧘 Try a brief mindfulness or breathing exercise",
        "📞 Connect with someone you trust about how you're feeling"
    ]
    
    # Get tips for the specific emotion or use default
    emotion_tips = tips.get(emotion.lower(), default_tips)
    
    # Select 2-3 tips based on confidence level
    num_tips = min(3, max(2, int(confidence * 5)))
    selected_tips = emotion_tips[:num_tips]
    
    return selected_tips

@cache_resource
def load_resources(model_fname="model.h5", tokenizer_fname="tokenizer.pkl", le_fname="label_encoder.pkl"):
    # Try to resolve paths relative to app file or current working dir
    base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    model_path = os.path.join(base, model_fname)
    tok_path = os.path.join(base, tokenizer_fname)
    le_path = os.path.join(base, le_fname)

    # Try alternative extensions for model files
    if not os.path.exists(model_path):
        model_path_keras = os.path.join(base, "model.keras")
        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        else:
            model_path_savedmodel = os.path.join(base, "saved_model")
            if os.path.exists(model_path_savedmodel):
                model_path = model_path_savedmodel
    
    missing = [p for p in (model_path, tok_path, le_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}. Run streamlit from the folder containing them.")

    # Load model (use compile=False for inference)
    try:
        model = load_model(model_path, compile=False)
    except:
        # Try loading with custom objects if needed
        model = load_model(model_path, compile=False)
    
    # Load tokenizer and label encoder
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # sanity checks
    if not hasattr(tokenizer, "texts_to_sequences"):
        raise TypeError("Loaded tokenizer does not look like a Keras Tokenizer object.")
    if not hasattr(le, "inverse_transform") or not hasattr(le, "classes_"):
        raise TypeError("Loaded label encoder does not look like sklearn.preprocessing.LabelEncoder.")

    return model, tokenizer, le

# ---------------------------
# App start
# ---------------------------
st.markdown('<h1 class="main-header">🎭 Emotion Sentiment Analysis</h1>', unsafe_allow_html=True)

# Try to load resources
try:
    model, tokenizer, le = load_resources()
except Exception as e:
    st.error("❌ Failed to load model/tokenizer/encoder.")
    st.error(f"Error: {str(e)}")
    
    # Provide troubleshooting tips
    with st.expander("Troubleshooting tips"):
        st.write("""
        1. Make sure you have these files in your app directory:
           - model.h5 (or model.keras)
           - tokenizer.pkl
           - label_encoder.pkl
        2. If using a different filename, modify the load_resources() function
        3. Check that the files are not corrupted
        4. Ensure all dependencies are installed
        """)
    st.stop()

classes = list(le.classes_)
maxlen = 100  # Fixed value as requested

# Initialize session state for text input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Main content area
st.subheader("Analyze Text Emotion")
user_input = st.text_area("✍️ Enter text to analyze emotion:", 
                         value=st.session_state.user_input,
                         height=120, 
                         placeholder="Type your text here...",
                         key="text_input")

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔮 Predict Emotion", type="primary")
with col2:
    if st.button("🔄 Clear", type="secondary"):
        st.session_state.user_input = ""
        st.rerun()

if predict_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Analyzing..."):
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
                preds = model.predict(padded, verbose=0)  # shape (1, n_classes)
                probs = preds[0]
                top_idx = int(np.argmax(probs))
                top_prob = probs[top_idx]
                top_label = le.inverse_transform([top_idx])[0]
                
                # Display results
                st.markdown(f'<div class="prediction-box"><h3>🎉 Predicted Emotion: {top_label}</h3>'
                           f'<p>Confidence: {top_prob:.3f}</p></div>', unsafe_allow_html=True)
                
                # Show personalized tips based on emotion
                tips = get_emotion_tips(top_label, top_prob)
                st.markdown("#### 💡 Personalized Suggestions")
                st.markdown('<div class="tips-box">', unsafe_allow_html=True)
                for tip in tips:
                    st.write(f"• {tip}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a nice visualization with Plotly
                df = pd.DataFrame({"Emotion": classes, "Confidence": probs})
                df = df.sort_values("Confidence", ascending=False)
                
                fig = px.bar(df, x="Emotion", y="Confidence", 
                            color="Confidence",
                            color_continuous_scale="Bluered",
                            title="Emotion Confidence Scores")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed probabilities
                st.subheader("Detailed Probabilities")
                for emotion, prob in zip(classes, probs):
                    col_a, col_b = st.columns([2, 4])
                    with col_a:
                        st.write(f"{emotion}: {prob:.3f}")
                    with col_b:
                        st.progress(float(prob))
                
        except Exception as e:
            st.error("Prediction failed. See error details below:")
            st.exception(e)

# Add some examples for users to try
st.subheader("Try These Examples")
examples = [
    "I'm so happy today! Everything is going well.",
    "This is terrible. I can't believe this happened.",
    "I'm feeling anxious about the future.",
    "i was feeling he wasnt shocked at all by what i was telling him.",
    "im now on day two of the plan and im feeling positive",
    "i feel like being all stubborn and stingy"
]

example_cols = st.columns(2)
for i, example in enumerate(examples):
    with example_cols[i % 2]:
        if st.button(f"Try: {example}", key=example):
            # Update the session state with the example text
            st.session_state.user_input = example
            # Rerun the app to update the text area
            st.rerun()