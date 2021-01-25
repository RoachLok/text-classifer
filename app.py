from flask import Flask, render_template, request, jsonify
from conexions import *

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
        return render_template('_results-section.html', train_model(json_array))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')