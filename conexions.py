import pandas as pd
from modelo_noticias_despoblacion import ModeloDesp

despo_samples = []
nodespo_samples = []
unlabeled_samples = []
model = ModeloDesp()
despo_set = None
nodespo_set = None
dataset = None
unlabeled_set = None
corpus_desp = []
corpus_nodespo = []
corpus_unlabeled = []
corpus_dataset = []

modelNameAutoSelected = None
metrics = []
confusion_matrix = None
plt_img = None

def on_file_upload(json_array, file_type, stopwords=True):
    global despo_samples
    global nodespo_samples
    global unlabeled_samples
    global despo_set
    global nodespo_set
    global unlabeled_set
    global corpus_desp
    global corpus_nodespo
    global corpus_unlabeled

    if file_type == 'despo':
        despo_samples = json_array
        despo_set = model.cargarTextosTraining(json_array, "Despoblacion")
        corpus_desp = model.preprocesarTextos(despo_set, stopwords)
    elif file_type == 'nodespo':
        nodespo_samples = json_array
        nodespo_set = model.cargarTextosTraining(json_array, "No Despoblacion")
        corpus_nodespo = model.preprocesarTextos(nodespo_set, stopwords)
    else:
        unlabeled_samples = json_array
        unlabeled_set = model.cargarTextosTest(json_array)
        corpus_unlabeled = model.preprocesarTextos(unlabeled_set, stopwords)


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
    elif file_type == 'nodespo':
        for index, file in enumerate(nodespo_samples):
            if file['name'] == file_name:
                return [file['content'], corpus_nodespo[index]]
    else:
        for index, file in enumerate(unlabeled_samples):
            if file['name'] == file_name:
                return [file['content'], corpus_unlabeled[index]]

def model_train(model_name="AUTO", vector_transform="cv", prune=10):
    global modelNameAutoSelected
    global confusion_matrix
    global metrics
    global plt_img
    get_full_dataset()
    get_full_corpus()
    if prune == 0:
        prune = None
    else:
        prune = prune / 100
    if model_name == "AUTO":
        modelNameAutoSelected, confusion_matrix, metrics, plt_img = model.model_training(
            model_name, dataset, corpus_dataset, vector_transform, prune)
        return confusion_matrix, metrics, plt_img, modelNameAutoSelected
    else:
        modelNameAutoSelected, confusion_matrix, metrics, plt_img = model.model_training(
            model_name, dataset, corpus_dataset, vector_transform, prune)
        return confusion_matrix, metrics, plt_img, modelNameAutoSelected


def save_model_cv(savepath):
    model.save_model(savepath, modelNameAutoSelected, confusion_matrix, metrics, plt_img)


def on_trained_upload(serialized_file):
    modelNameAutoSelected, confusion_matrix, metrics, plt_img = model.load_model(serialized_file)
    return confusion_matrix, metrics, plt_img, modelNameAutoSelected

def tune_model():
    modelNameAutoSelected, confusion_matrix, metrics, plt_img = model.model_self_tuning()
    return confusion_matrix, metrics, plt_img, modelNameAutoSelected


def test_model():
    file_test_names = []
    for index, file in enumerate(unlabeled_samples):
        file_test_names.append(file['name'])
    dict_results_test = model.model_testing(unlabeled_set, corpus_unlabeled, file_test_names)

    return dict_results_test
