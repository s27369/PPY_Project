from flask import Flask, request, jsonify, render_template
import joblib
import os
import pickle
import webbrowser
from source.KNN import KNN

app = Flask(__name__)
models = {}

@app.route('/')
def home():
    return render_template(r'index.html', models=models.keys())


@app.route('/result', methods=['GET'])
def result():
    model_name = request.args.get('model')
    sepal_length = request.args.get('SepalLength').replace(",", ".").strip()
    sepal_width = request.args.get('SepalWidth').replace(",", ".").strip()
    petal_length = request.args.get('PetalLength').replace(",", ".").strip()
    petal_width = request.args.get('PetalWidth').replace(",", ".").strip()
    k = request.args.get('k').replace(",", ".").strip()

    if model_name not in models:
        return render_template('result.html', prediction="Model not found")
    model = models[model_name]

    try:
        features = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width), "xdd"]
        if model_name=="knn_model":
            if k != "":
                try:
                    k = int(k)
                except:
                    return render_template('result.html', prediction=f"invalid k parameter")
            prediction = model.predict(features, 3 if k == "" else k)
        else:
            prediction = "Iris-Setosa" if model.predict(features)==1 else "Non-setosa"
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    model_dir = 'models'
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            model_name = filename[:-4]  # bez .pkl
            models[model_name] = load_model(os.path.join(model_dir, filename))

    app.run(debug=False)