from flask import Flask, render_template, request, jsonify, send_file
from base64 import b64encode
from conexions import *
import json

app = Flask(__name__)

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/howto')
def howto():
    return render_template('howto.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_train_files/<file_type>', methods=['GET', 'POST'])
def get_train_file(file_type):
    if request.method == 'POST':
        json_array = request.get_json()
        on_file_upload(json_array, file_type)
        return render_template('_dropdowns.html', file_names = json_array, file_type = file_type)

    file_attri = file_type.split('-', 1)
    return jsonify(get_file_content(file_attri[1], file_attri[0]))

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        json_array = request.get_json()
        
        results = model_train(model_name=json_array['model-select'], vector_transform=json_array['transform-select'], prune=int(json_array['prune-range']))
        cm_image   = b64encode(results[0].getvalue()).decode()
        plot_image = b64encode(results[2].getvalue()).decode()

        return render_template('_results-section.html', results = results, cm_image = cm_image, plot_image = plot_image)

@app.route('/trained_model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        model_name = on_trained_upload(request.files['model_file'].read())
        results = [0, 0, 0, model_name]
        return render_template('_results-section.html', results=results)

@app.route('/classify')
def classify():
    print(test_model())
    return render_template('_classification.html', results_json = test_model())
    
@app.route('/download_model')
def download_trained_model():
    save_model_cv('download.sav')
    return send_file('download.sav', as_attachment=True)

@app.route('/download_results')
def download_results():
    with open('results.json', 'w') as outfile:
        json.dump(test_model(), outfile)
    return send_file('results.json', as_attachment=True)

@app.route('/')
def index():
    return render_template('classifier.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')