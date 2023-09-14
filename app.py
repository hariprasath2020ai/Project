
from flask import Flask, request, jsonify
from tensorflow import keras
from Feature_Extractor import extract_features

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Get the URL from the query parameters
    url = request.args.get('url')

    # # Load the Keras model
    model_path=r"C:\Users\Barathkumar\Desktop\BETA1 (API)\model.h5"
    model = keras.models.load_model(model_path)

    # # Extract features from the URL
    url_features = extract_features(url)


    # # Make a prediction with the model
    prediction = model.predict([url_features])
    score = prediction[0][0] * 100
    score = round(score, 3)
    print(score)


    # # Return the prediction as a JSON response
    return {'url': url, 'score': score}

if __name__ == '__main__':
    app.run()
