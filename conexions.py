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


'''

def train_model(stopwords, prune):
    prune = prune / 100
    return model.save_model()

def loadTrain(model):
    model.dataset = ModeloDesp.cargarTextosTraining(
        model.dataset, chooserDir())

def loadTest(model):
    model.test_set = ModeloDesp.cargarTextosTest(model.test_set, chooserDir())


def trainModel(model):
    model.count_vectorizer, model.selectedModel = ModeloDesp.model_training(
        model.dataset, "AUTO", 0.06)


def saveModel(model):
    ModeloDesp.save_model(
        chooserSave(), model.count_vectorizer, model.selectedModel)


def loadModel(model):
    model.count_vectorizer, model.selectedModel = ModeloDesp.load_model(
        chooserLoad())


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
