from flask import Flask, render_template, request, jsonify
import joblib
import pickle
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load('sentiment_analysis_model(lr).pkl')

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

def analyze_sentiment(text):
    sw_list = stopwords.words('english')
    new_text_processed = ' '.join([word for word in text.split() if word.lower() not in sw_list])

    new_text_bow = cv.transform([new_text_processed]).toarray()
    prediction_proba = model.predict_proba(new_text_bow)


    if prediction_proba[0, 1] >= 0.75:
        return {
            "image": "<img src='/static/images/very_positive.png' alt='Very Positive' />",
            "type": "very_positive",
            "text": "Very  Positive"
        }
    elif prediction_proba[0, 1] >= 0.55 and prediction_proba[0, 1] < 0.75:
        return {
            "image": "<img src='/static/images/positive.png' alt='Positive' />",
            "type": "positive",
            "text": "POSITIVE"
        }
    elif prediction_proba[0, 1] >= 0.45 and prediction_proba[0, 1] < 0.55:
        return {
            "image": "<img src='/static/images/neutral.png' alt='Neutral' />",
            "type": "neutral",
            "text": "Neutral"
        }
    elif prediction_proba[0, 1] >= 0.20 and prediction_proba[0, 1] < 0.45:
        return {
            "image": "<img src='/static/images/negative.png' alt='Negative' />",
            "type": "negative",
            "text": "Negative"
        }
    elif prediction_proba[0, 1] < 0.20:
        return {
            "image": "<img src='/static/images/very_negative.png' alt='Very Negative' />",
            "type": "very_negative",
            "text": "Very Negative"
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text', '')
        sentiment = analyze_sentiment(text)
        return jsonify({'text': text, 'sentiment': sentiment})


if __name__ == '__main__':
    app.run(debug=True)

 