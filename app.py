from flask import Flask, render_template, request
from conexions import *

app = Flask(__name__)

@app.route('/Classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/get_train_files/<file_type>', methods=['GET', 'POST'])
def get_train_file(file_type):
    if request.method == 'POST':
        json_array = request.get_json()
        return render_template('_dropdowns.html', file_names = json_array, file_type = file_type)

    return render_template('_dropdowns.html', file_type = file_type)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)