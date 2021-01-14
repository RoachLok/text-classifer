
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import time
import nltk
from nltk.stem import SnowballStemmer
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from pickle import dump
from pickle import load

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from tkinter import filedialog
from tkinter import *


class ModeloDesp:
    def __init__(self):
        self.dataset = pd.DataFrame(columns = ["noticia", "categoria"])
        self.test_set = pd.DataFrame(columns = ["noticia"])
        self.count_vectorizer = None
        self.selectedModel = None
        self.X = None
        self.y = None
        self.test = None
    
    def cargarTextosTraining(self, dirTrain):
        files_name = []
        for f in os.listdir(dirTrain):
            path = os.path.join(dirTrain, f)
            with open(path, 'r', encoding='utf-8', errors = 'ignore') as file:
                self.dataset = self.dataset.append({'noticia': file.read(), 'categoria':  os.path.basename(dirTrain[:-1])}, ignore_index=True)
        
    def cargarTextosTest(self, dirTest):
        for f in os.listdir(dirTest):
            path = os.path.join(dirTest, f)
            with open(path, 'r', encoding='utf-8', errors = 'ignore') as file:
                self.test_set = self.test_set.append({'noticia': file.read()}, ignore_index=True)
              
    
    def preprocesarTextos(self, dataset):
        # Limpieza de textos, en los que se incluye tokenizacion y stemming.
        corpus = []
        for i in range(len(dataset)):
            text = re.sub('[^a-zA-Z]', ' ', dataset['noticia'][i]) # Primera limpieza = solo conserva las letras. Todo lo que no sea una letra se remplaza por un espacio ''.
            text = text.lower() # Segunda limpieza = transformar todas las letras mayúsculas en minúsculas.
            text = text.split() # Tercera limpieza = dividir los textos en sus diferentes palabras para que podamos aplicar lematizacion/stemming a cada palabra
            # Cuarta limpieza
            #ps = PorterStemmer()
            all_stopwords = stopwords.words('spanish')
            sst = SnowballStemmer('spanish')
            text = [sst.stem(word) for word in text if not word in set(all_stopwords)] # For en una línea, aplicamos stemming a cada palabra con la condición de no tratar y deshacerse de las stopwords.
            text = ' '.join(text) # Volvemos a unir las palabras de los textos separados por espacio para obtener el formato original de los textos
            corpus.append(text)  
        return corpus

    def model_name_selection(self, modelName):
        if modelName == "LR":
            return LogisticRegression()
        elif modelName == "LDA":
            return LinearDiscriminantAnalysis()
        elif modelName == "KNN":
            return KNeighborsClassifier()
        elif modelName == "NB":
            return GaussianNB()
        elif modelName == "SVC":
            return SVC()
        elif modelName == "RF":
            return RandomForestClassifier()
        elif modelName == "ANN":
            return MLPClassifier()
    
    def model_training(self, modelName, min_dif = None):
        self.count_vectorizer = CountVectorizer(min_df = min_dif)
        self.X = self.count_vectorizer.fit_transform(self.preprocesarTextos(self.dataset)).toarray()
        self.y = self.dataset['categoria'].values # Dependent variable
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
        if modelName == "AUTO":
            #print("Auto model selection")
            models = []
            models.append(('LR', LogisticRegression()))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))
            models.append(('RF', RandomForestClassifier()))
            models.append(('ANN', MLPClassifier()))
            
            results = []
            names = []
            for name, model in models:
                cv_results = cross_val_score(model, X_train, y_train, cv = 10, scoring = "accuracy")
                results.append(cv_results)
                names.append(name)
                msg = f"{name}: {cv_results.mean()} ({cv_results.std()})"
                #print(msg)
                
            best_model = dict(zip(names, [np.average(result) for result in results]))
            #print(max(best_model, key=best_model.get))
            self.selectedModel = [y for x, y in models if x == max(best_model, key=best_model.get)][0]
            self.selectedModel = self.selectedModel.fit(X_train, y_train)
            y_pred = self.selectedModel.predict(X_test)
            
            # b) Compare Algorithms
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(results)
            ax.set_xticklabels(names)
            plt.show()
            
        else:
            self.selectedModel = self.model_name_selection(modelName)
            cv_results = cross_val_score(self.selectedModel, X_train, y_train, cv = 10, scoring = "accuracy")
            #print(f"{modelName}: {cv_results.mean()} ({cv_results.std()})")
            self.selectedModel.fit(X_train, y_train)
            y_pred = self.selectedModel.predict(X_test)
        '''
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        '''
    def model_self_tuning(self):
        #print("Parameter tuning")
        parameters = []
        if type(self.selectedModel).__name__ == 'LogisticRegression':
            parameters = [{'C':np.arange(0.1, 1.1, 0.1).tolist()}, {'penalty': ['elasticnet'] ,'C':np.arange(0.1, 1.1, 0.1).tolist()}]
        elif type(self.selectedModel).__name__ == 'LinearDiscriminantAnalysis':
            parameters =  [{'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage':['None', 'auto']}]
        elif type(self.selectedModel).__name__ == 'KNeighborsClassifier':
            parameters = [{'n_neighbors' : np.arange(1, 31, 1), 'weights':['uniform', 'distance']}]
        elif type(self.selectedModel).__name__ == 'GaussianNB':
            # Eliminar o cambiar
            #parameters =
            pass
        elif type(self.selectedModel).__name__ == 'SVC':
            parameters = [{'C':np.arange(0.1, 1.1, 0.1).tolist(), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['scale', 'auto']}]
        elif type(self.selectedModel).__name__ == 'RandomForestClassifier':
            parameters = [{'n_estimators': np.arange(100, 320, 20), 'criterion': ['gini', 'entropy']}]
        elif type(self.selectedModel).__name__ == 'MLPClassifier':
            parameters = [{'hidden_layer_sizes':[(50, 50), (100, 100, 100), (400, 100, 50, 10)], 'batch_size':[16, 32, 64], 'learning_rate':['adaptive'], 'max_iter':[50,100,300]}]
        
        #print(parameters)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
        grid_search = GridSearchCV(estimator = self.selectedModel,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
        grid_search.fit(X_train, y_train)
        #print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        self.selectedModel = grid_search.best_estimator_
        self.selectedModel.fit(X_train, y_train)
        y_pred = self.selectedModel.predict(X_test)
        '''
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        '''
    def model_testing(self):
        self.test = self.count_vectorizer.transform(self.preprocesarTextos(self.test_set)).toarray()
        y_pred = self.selectedModel.predict(self.test)
        y_pred_proba = self.selectedModel.predict_proba(self.test)
        y_pred_proba = np.matrix.round(y_pred_proba, 3)
        '''
        print(y_pred)
        print(y_pred_proba)
        '''
    def save_model(self, filename):
        # save the model to disk
        #filename = 'finalized_model.sav'
        bow_model_save = (self.count_vectorizer, self.selectedModel)
        dump(bow_model_save, open(filename, 'wb'))
        
    
    def load_model(self, filename):
        # load the model from disk
        bow_model_save = load(open(filename, 'rb'))
        self.count_vectorizer, self.selectedModel = bow_model_save
        
      
'''
La clase esta para pruebas. O mantenerla y crear un objeto externo;
o generar una clase externa que llame a los metodos y pueda mantener las variables en memoria.
'''
'''
prueba = ModeloDesp()

prueba.cargarTextosTraining("data/Noticias/Despoblacion/")
prueba.cargarTextosTraining("data/Noticias/No Despoblacion/")
prueba.model_training(modelName = "LDA", min_dif = 0.06)
prueba.model_self_tuning()
prueba.save_model("finalized_model.sav")
'''
'''
prueba.load_model("finalized_model.sav")
prueba.cargarTextosTest("data/unlabel/unlabel-1/")
prueba.cargarTextosTest("data/unlabel/unlabel-2/")
prueba.model_testing()
'''



