import os
import numpy as np
import pandas as pd
from pathlib import Path
from modelo_noticias_despoblacion import ModeloDesp

despo_samples = []
nodespo_samples = []
model = ModeloDesp()
despo_set = None
nodespo_set = None
dataset = None
corpus_desp = []
corpus_nodespo = []
corpus_dataset = []


def on_file_upload(json_array, file_type, stopwords=True):
    global despo_samples
    global nodespo_samples
    global despo_set
    global nodespo_set
    global corpus_desp
    global corpus_nodespo

    if file_type == 'despo':
        despo_set = None
        despo_samples = json_array
        despo_set = model.cargarTextosTraining(json_array, "Despoblacion")
        corpus_desp = model.preprocesarTextos(despo_set, stopwords)

    else:
        nodespo_set = None
        nodespo_samples = json_array
        nodespo_set = model.cargarTextosTraining(json_array, "No Despoblacion")
        corpus_nodespo = model.preprocesarTextos(nodespo_set, stopwords)


def get_full_dataset():
    global dataset
    try:
        dataset = pd.concat([despo_set, nodespo_set])
    except:
        print("Uno de los datasets no esta inicializado")


def get_full_corpus():
    global corpus_dataset
    try:
        corpus_dataset = corpus_desp + corpus_nodespo
    except:
        print("Uno de los corpus no esta inicializado")


def get_file_content(file_name, file_type):
    if file_type == 'despo':
        for index, file in enumerate(despo_samples):
            if file['name'] == file_name:
                return [file['content'], corpus_desp[index]]
    else:
        for index, file in enumerate(nodespo_samples):
            if file['name'] == file_name:
                return [file['content'], corpus_nodespo[index]]


def train_model(modelName="AUTO", vector_transform="cv", prune=10):
    get_full_dataset()
    get_full_corpus()
    prune = prune / 100
    if modelName == "AUTO":
        cm, accuracy, plt_img = model.model_training(
            modelName, dataset, corpus_dataset, vector_transform, prune)
        return cm, accuracy, plt_img
    else:
        cm, accuracy = model.model_training(
            modelName, dataset, corpus_dataset, vector_transform, prune)
        return cm, accuracy
