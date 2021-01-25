# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
import re
import time
import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Importacion de librerias para serializacion y fechaa/hora actual
from pickle import dump
from pickle import load
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


class ModeloDesp:
    '''
    Constructor para usar como variables de clase:
    - vectorizer = transformador de terminos a matriz,
    - selectedModel = objeto del modelo seleccionado,
    - dateTime = fecha y hora actuales,
    - X = variable independiente que contendra la matriz de terminos para entrenar
    - y = variable dependiente a predecir que contendra las categorias
    '''
    def __init__(self):
        self.vectorizer = None
        self.selectedModel = None
        self.dateTime = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        self.X = None
        self.y = None

    '''
    Metodo para cargar los textos de entrenamiento, pasando de json a un pandas dataframe con los textos y su categoria
    '''
    def cargarTextosTraining(self, json_files_train, categoria):
        train_set = pd.json_normalize(json_files_train)
        train_set = train_set.filter(items=['content'])
        train_set = train_set.rename(columns={"content": "noticia"})
        train_set['categoria'] = categoria
        return train_set

    '''
    Metodo para cargar los textos de test, pasando de json a un pandas dataframe con los textos no etiquetados
    '''
    def cargarTextosTest(self, json_files_test):
        test_set = pd.json_normalize(json_files_test)
        test_set = test_set.filter(items=['content'])
        test_set = test_set.rename(columns={"content": "noticia"})
        return test_set

    '''
    Metodo para preprocesar los textos incluidos en el dataframe de entrenamiento
    '''
    def preprocesarTextos(self, dataset, stopwords_setting=True):
        # Limpieza de textos, en los que se incluye tokenizacion y stemming.
        corpus = []
        all_stopwords = stopwords.words('spanish')
        for i in range(len(dataset)):
            # Primera limpieza = solo conserva las letras. Todo lo que no sea una letra se remplaza por un espacio ''.
            text = re.sub('[^a-zA-Z]', ' ', dataset['noticia'][i])
            # Segunda limpieza = transformar todas las letras mayúsculas en minúsculas.
            text = text.lower()
            text = text.split()  # Tercera limpieza = dividir los textos(tokenizacion) en sus diferentes palabras para que podamos aplicar stemming a cada palabra
            # Cuarta limpieza, aplicamos stemming (SnowballStemmer es la unica tecnica que permite hacer stemmizacion en castellano)
            sst = SnowballStemmer('spanish')
            # Stemmizacion aplicando stopwords
            if stopwords_setting:
                # For en una línea, aplicamos stemming a cada palabra con la condición de no tratar y deshacerse de las stopwords.
                text = [sst.stem(word) for word in text if not word in set(all_stopwords)]
            # Stemmizacion sin aplicar stopwords
            else:
                text = [sst.stem(word) for word in text]
            # Volvemos a unir las palabras de los textos separados por espacio para obtener el formato original de estos
            text = ' '.join(text)
            corpus.append(text)
        return corpus

    '''
    Metodo que nos permite devolver la clase del modelo seleccionado
    '''
    def model_name_selection(self, modelName):
        if modelName == "LR":
            return LogisticRegression(n_jobs=-1)
        elif modelName == "LDA":
            return LinearDiscriminantAnalysis()
        elif modelName == "KNN":
            return KNeighborsClassifier(n_jobs=-1)
        elif modelName == "NB":
            return GaussianNB()
        elif modelName == "CART":
            return DecisionTreeClassifier()
        elif modelName == "RF":
            return RandomForestClassifier(n_jobs=-1)
        elif modelName == "AB":
            return AdaBoostClassifier()
        elif modelName == "ANN":
            return MLPClassifier()

    '''
    Metodo que nos permite entrenar un modelo en base a:
    - modelName: Este parametro tendra el tipo de modelo a entrenar. Si el modelName = "AUTO" se hara una comparacion automatica
      de modelos mediante validacion cruzada y se seleccionara aquel modelo que mejor accuracy obtenga en su entrenamiento.
      De lo contrario, si el modelName es un nombre de modelo (LR, LDA, etc...), sera ese modelo seleccionado el que se entrenara.

    - dataset: Este parametro contendra el dataset completo, el cual nos servira en esta ocasion para obtener la variable dependiente y,
      es decir, la categoria perteneciente a cada noticias.

    - corpus: Este parametro contendra el corpus completo de las noticias, es decir, las noticias ya preprocesadas. Lo cual nos servirá para
      poder añadirlas al transformador de terminos y así generar la matriz de terminos de las noticias (X).

    - vector_transform: Este parametro tendra el tipo de transformador de terminos a aplicar. Si el vector_transform = "cv" se aplicara CountVectorizer
      generando una matriz de terminos en base a la frecuencia de terminos. Si el vector_transform = "tfidf" se aplicara TfidfVectorizer generando una matriz
      de terminos en base a la frecuencia de terminos pero en este caso expresando numericamente la relevancia de una palabra para un documento.

    - min_dif: Este parametro contendra el porcentaje de poda que se aplicara. Por ejemplo si min_df=0.06 se aplicara una poda de tal manera que aquellos terminos
      que no aparezcan en al menos 6% del total de la coleccion de textos, se eliminaran; es decir, se podaran y no se tendran en cuenta a la hora de generar
      la matriz de terminos. 
    '''
    def model_training(self, modelName, dataset, corpus, vector_transform, min_dif=None):
        if (vector_transform == "cv"):
            self.vectorizer = CountVectorizer(min_df=min_dif)
        elif(vector_transform == "tfidf"):
            self.vectorizer = TfidfVectorizer(min_df=min_dif)
        self.X = self.vectorizer.fit_transform(corpus).toarray() # Aplicamos la transformacion de terminos al corpus para generar la matriz de terminos (X)
        self.y = dataset['categoria'].values # Variable dependiente y (categorias)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) # Split para entrenamiento-test
        # Si elegimos hacer una seleccion automatica de modelo
        if modelName == "AUTO":
            models = []
            models.append(('LR', LogisticRegression(n_jobs=-1)))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
            models.append(('NB', GaussianNB()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('RF', RandomForestClassifier(n_jobs=-1)))
            models.append(('AB', AdaBoostClassifier()))
            models.append(('ANN', MLPClassifier()))

            results = []
            names = []
            for name, model in models:
                cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
                results.append(cv_results)
                names.append(name)

            best_model = dict(zip(names, [np.average(result) for result in results]))
            self.selectedModel = [model for name, model in models if name == max(best_model, key=best_model.get)][0]
            self.selectedModel.fit(X_train, y_train)
            y_pred = self.selectedModel.predict(X_test)

            # Plot de comparacion de algoritmos en base a su validacion cruzada
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            img = io.BytesIO()
            plt.savefig(img)

            return type(self.selectedModel).__name__, confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred), img

        # Si elegimos nosotros el modelo a entrenar
        else:
            self.selectedModel = self.model_name_selection(modelName)
            cv_results = cross_val_score(self.selectedModel, X_train, y_train, cv=10, scoring="accuracy")
            self.selectedModel.fit(X_train, y_train)
            y_pred = self.selectedModel.predict(X_test)

            # Plot de resultado para la validacion cruzada de un modelo
            fig = plt.figure()
            fig.suptitle('Algorithm cross validation results')
            ax = fig.add_subplot(111)
            plt.boxplot(cv_results)
            ax.set_xlabel(type(self.selectedModel).__name__)
            img = io.BytesIO()
            plt.savefig(img)

            return confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred), img

    '''
    Metodo para hacer un autoajuste de los parametros internos de tu modelo seleccionado.
    De esta manera conseguiremos obtener un modelo optimizado.
    '''
    def model_self_tuning(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        parameters = []
        if type(self.selectedModel).__name__ == 'LogisticRegression':
            parameters = [{'C': np.arange(0.1, 1.1, 0.1).tolist()}, {'penalty': [
                'elasticnet'], 'C':np.arange(0.1, 1.1, 0.1).tolist()}]
        elif type(self.selectedModel).__name__ == 'LinearDiscriminantAnalysis':
            parameters = [{'solver': ['svd', 'lsqr', 'eigen'],
                           'shrinkage':['None', 'auto']}]
        elif type(self.selectedModel).__name__ == 'KNeighborsClassifier':
            parameters = [{'n_neighbors': np.arange(1, 31, 1), 'weights': [
                'uniform', 'distance']}]
        elif type(self.selectedModel).__name__ == 'GaussianNB': # Este modelo NO contiene parametros internos ajustables, por lo que no se puede autoajustar.
            self.selectedModel.fit(X_train, y_train)
            y_pred = self.selectedModel.predict(X_test)
            return confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)
        elif type(self.selectedModel).__name__ == 'DecisionTreeClassifier':
            parameters = [{'criterion': [
                'gini', 'entropy'], 'max_features':['auto', 'sqrt', 'log2']}]
        elif type(self.selectedModel).__name__ == 'RandomForestClassifier':
            parameters = [{'n_estimators': np.arange(100, 320, 20), 'criterion': [
                'gini', 'entropy']}]
        elif type(self.selectedModel).__name__ == 'AdaBoostClassifier':
            parameters = [{'n_estimators': np.arange(25, 50, 100, 200)}]
        elif type(self.selectedModel).__name__ == 'MLPClassifier':
            parameters = [{'hidden_layer_sizes': [(50, 50), (100, 100, 100), (400, 100, 50, 10)], 'batch_size':[
                16, 32, 64], 'learning_rate':['adaptive'], 'max_iter':[50, 100, 300]}]

        # GridSearch para seleccionar aquellos parametros que mejor se ajustan al modelo seleccionado.
        grid_search = GridSearchCV(estimator=self.selectedModel,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.selectedModel = grid_search.best_estimator_ # Obtenemos nuestro modelo seleccionado con los parametros internos que mejor se ajustan (modelo seleccionado optimizado)
        self.selectedModel.fit(X_train, y_train)
        y_pred = self.selectedModel.predict(X_test)
        return confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

    '''
    Metodo que nos permite predecir la categoria a la que pertenecen noticias sin etiquetar
    Los resultados de la prediccion vendran dado en formato: categoria predicha y la probabilidad de pertenecer a dicha categoria
    '''
    def model_testing(self, corpus_test):
        test = self.vectorizer.transform(corpus_test).toarray()
        y_pred = self.selectedModel.predict(test)
        y_pred_proba = self.selectedModel.predict_proba(test)
        y_pred_proba = np.matrix.round(y_pred_proba, 3)
        return y_pred, y_pred_proba

    '''
    Metodo que nos permite serializar y guardar:
     - El transformador de terminos a matriz
     - El modelo seleccionado
     - La fecha y hora en que se hizo el entrenamiento
    '''
    def save_model(self, filename):
        # save the model to disk
        # filename = 'finalized_model.sav'
        bow_model_save = (self.vectorizer, self.selectedModel, self.dateTime)
        dump(bow_model_save, open(filename, 'wb'))

    '''
    Metodo que nos permite deserializar y cargar:
     - El transformador de terminos a matriz
     - El modelo seleccionado
     - La fecha y hora en que se hizo el entrenamiento del archivo cargado
    '''
    def load_model(self, filename):
        # load the model from disk
        bow_model_save = load(open(filename, 'rb'))
        self.vectorizer, self.selectedModel, self.dateTime = bow_model_save
