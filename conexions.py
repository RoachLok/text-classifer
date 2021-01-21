import os
import numpy as np
import pandas as pd
from pathlib import Path
from tkinter import filedialog
from tkinter import *
from modelo_noticias_despoblacion import ModeloDesp


despo_samples = []
nodespo_samples = []
model = ModeloDesp()
corpus_desp = []
corpus_nodespo = []
corpus_dataset = []


def on_file_upload(json_array, file_type):
    global despo_samples
    global nodespo_samples
    global corpus_desp
    global corpus_nodespo

    if file_type == 'despo':
        despo_set = None
        despo_samples = json_array
        despo_set = model.cargarTextosTraining(json_array, "Despoblacion")
        corpus_desp = model.preprocesarTextos(despo_set)
        # return corpus_desp

    else:
        nodespo_set = None
        nodespo_samples = json_array
        nodespo_set = model.cargarTextosTraining(json_array, "No Despoblacion")
        corpus_nodespo = model.preprocesarTextos(nodespo_set)
        # return corpus_nodespo


def get_full_corpus():
    global corpus_dataset
    corpus_dataset = corpus_desp + corpus_nodespo
    # Esto hacerlo de cualquiera de las maneras, pero luego siempre juntarlo para hacer el count_vectorizer (que siempre tiene que ser de todos los textos)


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


def apply_preprocessing(file_type, *settings):
    pd.DataFrame(columns=["noticia", "categoria"])

    if file_type == 'despo':
        for sample in despo_samples:
            model.dataset.append(
                {'noticia': sample['content'], 'categoria': file_type}, ignore_index=True)
    else:
        for sample in nodespo_samples:
            model.dataset.append(
                {'noticia': sample['content'], 'categoria': file_type}, ignore_index=True)

    model.preprocesarTextos({'noticia'})

    return True


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
