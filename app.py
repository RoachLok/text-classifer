from flask import Flask, json, render_template, request, jsonify

app = Flask(__name__)

@app.route('/Classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/About')
def about():
    return render_template('about.html')

@app.route('/get_train_files', methods=['GET', 'POST'])
def get_train_file():
    if request.method == 'POST':
        pass
    return jsonify("name: test", "name2: test2"), 201

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)