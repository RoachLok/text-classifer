import os
import pandas as pd
from pathlib import Path
from tkinter import filedialog
from tkinter import *
from modelo_noticias_despoblacion import ModeloDesp


despo_samples = []
nodespo_samples = []
model = ModeloDesp()


def on_file_upload(json_array, file_type):
    global despo_samples
    global nodespo_samples
    print(despo_samples)
    if file_type == 'despo':
        despo_samples = json_array
    else:
        nodespo_samples = json_array


def get_file_content(file_name, file_type):

    if file_type == 'despo':
        for file in despo_samples:
            if file['name'] == file_name:
                return file['content']
    else:
        for file in nodespo_samples:
            if file['name'] == file_name:
                return file['content']


# Filename required to save the file
def train_model(stopwords, prune):
    prune = prune / 100
    model.model_training(model.dataset, prune, stopwords)
    return model.save_model(model.count_vectorizer, model.selectedModel)


# Check if model exists, if not dont apply model
def apply_model(optimizations):
    if optimizations == True and not model.dataset.empty:
        model.model_self_tuning()
    else:
        return "Error the model name is null ore empty"


def load_model(model, model_selected):
    if model_selected is not None:
        model.load_model(model_selected)
        return True
    else:
        return False


'''
def loadTest(model):
    model.test_set = ModeloDesp.cargarTextosTest(model.test_set, chooserDir())

def testModel(model):
    ModeloDesp.model_testing(
        model.test_set, model.count_vectorizer, model.selectedModel)
'''


def getNames(dir):
    texts = Path(dir).rglob('*.txt')
    names = [str(path) for path in texts]
    name_list = []
    for name in names:
        new_title = os.path.basename(name)
        name_list.append(new_title)
    return name_list
