import os
import pandas as pd
from pathlib import Path
from tkinter import filedialog
from tkinter import *
from modelo_noticias_despoblacion import ModeloDesp


class ModeloDesp:
    def __init__(self):
        self.dataset = pd.DataFrame(columns=["noticia", "categoria"])
        self.test_set = pd.DataFrame(columns=["noticia"])
        self.count_vectorizer = None
        self.selectedModel = None
        self.test = None


def chooserDir():  # dir
    root = Tk()
    root.withdraw()  # Starts interactive file input
    folder_selected = filedialog.askdirectory()
    return folder_selected


def chooserLoad():  # load
    load = Tk()
    load.withdraw()
    load_selected = filedialog.askopenfilename()
    return load_selected


def chooserSave():  # save
    save = Tk()
    save.withdraw()
    save_selected = filedialog.asksaveasfilename()
    return save_selected


def loadTrain(model):
    model.dataset = ModeloDesp.cargarTextosTraining(model.dataset, chooserDir())


def loadTest(model):
    model.test_set = ModeloDesp.cargarTextosTest(model.test_set, chooserDir())


def trainModel(model):
    model.count_vectorizer, model.selectedModel = ModeloDesp.model_training(
        model.dataset, "AUTO", 0.06)


def saveModel(model):
    ModeloDesp.save_model(chooserSave(), model.count_vectorizer, model.selectedModel)


def loadModel(model):
    model.count_vectorizer, model.selectedModel = ModeloDesp.load_model(chooserLoad())


def testModel(model):
    ModeloDesp.model_testing(model.test_set, model.count_vectorizer, model.selectedModel)


def getNames(dir):
    texts = Path(dir).rglob('*.txt')
    names = [str(path) for path in texts]
    name_list = []
    for name in names:
        new_title = os.path.basename(name)
        name_list.append(new_title)
    return name_list
