import glob
import chardet
import os
import pandas as pd
from pathlib import Path
from charset_normalizer import detect
from chardet.universaldetector import UniversalDetector
from IPython.display import display
from tkinter import filedialog
from tkinter import *
from modelo_noticias_despoblacion import cargarTextosTest, cargarTextosTraining, load_model, model_testing, model_training, save_model


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
    model.dataset = cargarTextosTraining(model.dataset, chooserDir())


def loadTest(model):
    model.test_set = cargarTextosTest(model.test_set, chooserDir())


def trainModel(model):
    model.count_vectorizer, model.selectedModel = model_training(
        model.dataset, "AUTO", 0.06)


def saveModel(model):
    save_model(chooserSave(), model.count_vectorizer, model.selectedModel)


def loadModel(model):
    model.count_vectorizer, model.selectedModel = load_model(chooserLoad())


def testModel(model):
    model_testing(model.test_set, model.count_vectorizer, model.selectedModel)


def getNames(dir):
    texts = Path(dir).rglob('*.txt')
    names = [str(path) for path in texts]
    name_list = []
    for name in names:
        new_title = os.path.basename(name)
        name_list.append(new_title)
    return name_list
