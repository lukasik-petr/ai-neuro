#!/usr/bin/python3

#------------------------------------------------------------------------------
# ai-neuro
# (C) GNU General Public License,
# pro potreby projektu TM-AI vyrobil Petr Lukasik , 2022 ");
#------------------------------------------------------------------------------
# program pro projekt TM-AI v TAJMAC-ZPS,a.s.
# predikce vlivu teploty na presnost obrabeni
# Prerekvizity: linux Debian-11 nebo Ubuntu-20.04,);
#               miniconda3,
#               python 3.9,
#               tensorflow 2.8,
#               mathplotlib,
#               scikit-learn-intelex,
#               pandas,
#               numpy,
#               keras,
#
# Windows se pokud mozno vyhnete. Tak je doporuceno v manualech TensorFlow
# i kdyz si myslim ze by to take fungovalo. Ale proc si delat zbytecne
# starosti...
#
# Pozor pro instalaci je nutno udelat nekolik veci
#  1. instalace prostredi miniconda 
#       a. stahnout z webu miniconda3 v nejnovejsi verzi
#       b. chmod +x Miniconda3-py39_4.11.0-Linux-x86_64.sh
#       c. ./Miniconda3-py39_4.11.0-Linux-x86_64.sh
#
#  2. update miniconda
#       conda update conda
#
#  3. vyrobit behove prostredi 'tf' v miniconda
#       conda create -n tf python=3.8
#
#  4. aktivovat behove prostredi tf (preX tim je nutne zevrit a znovu
#     otevrit terminal aby se conda aktivovala.
#       conda activate  tf
#
#  5. instalovat tyto moduly (pomoci conda)
#       conda install tensorflow=2.8 
#       conda install mathplotlib
#       conda install scikit-learn-intelex
#       conda install pandas
#       conda install numpy
#       conda install keras
#
#  6. v prostredi tf jeste upgrade tensorflow 
#       pip3 install --upgrade tensorflow
#------------------------------------------------------------------------------

import os;
import logging;
import sys, getopt, traceback;
#import IPython
#import IPython.display
import glob as glob;
import pandas as pd;
import seaborn as sns;
import tensorflow as tf;
import math;
import numpy as np;
import shutil;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import pickle;

from dateutil import parser
from sklearn.preprocessing import MinMaxScaler;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import max_error;
from sklearn.utils import shuffle
from numpy import asarray;
#from matplotlib import pyplot;
from dataclasses import dataclass;
from datetime import datetime
from tabulate import tabulate
from pathlib import Path

#from tensorflow.keras.datasets import imdb;
from tensorflow.keras import models;
from tensorflow.keras import layers;
from tensorflow.keras import optimizers;
from tensorflow.keras import losses;
from tensorflow.keras import metrics;
from tensorflow.keras import callbacks;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import InputLayer;
from tensorflow.keras.layers import Input;
from tensorflow.keras.layers import Dense;
from tensorflow.keras.layers import LSTM;
from tensorflow.keras.layers import GRU;
from tensorflow.keras.layers import Conv1D;
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
#from keras.utils.vis_utils import plot_model
from matplotlib import cm;
from datetime import datetime
from _cffi_backend import string
from pandas.core.frame import DataFrame

from scipy.signal import butter, lfilter, freqz


#---------------------------------------------------------------------------
# DataFactory
#---------------------------------------------------------------------------

class DataFactory():
    

    df_parmX_predict = [];
    
    @dataclass
    class DataTrain:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z


    @dataclass
    class DataTrainDim:
          DataTrain: object;
          
    @dataclass
    class DataResult:
        # return self.DataResult(x_test, y_test, y_result, mse, mae)
        x_test:    object           #testovaci mnozina v ose x
        y_test:    object           #testovaci mnozina v ose y
        y_result:  object           #vysledna mnozina
        axis:      string           #osa stroje [X,Y, nebo Z]

    @dataclass
    class DataResultDim:
          DataResultX: object;

    def __init__(self, path_to_result, window):
        
    #Vystupni list parametru - co budeme chtit po siti predikovat
        self.df_parmx = ['temp_S1','temp_pr01',
                         'temp_pr02',
                         'temp_pr03',
                         'temp_vr01',
                         'temp_vr02',
                         'temp_vr03',
                         'temp_vr04',
                         'temp_vr05',
                         'temp_vr06',
                         'temp_vr07'];
        
    #Tenzor predlozeny k uceni site
        self.df_parmX = ['dev_x4',
                         'dev_y4',
                         'dev_z4', 
                         'temp_S1',
                         'temp_pr01',
                         'temp_pr02',
                         'temp_pr03',
                         'temp_vr01',
                         'temp_vr02',
                         'temp_vr03',
                         'temp_vr04',
                         'temp_vr05',
                         'temp_vr06',
                         'temp_vr07'];
        
        self.path_to_result = path_to_result;
        self.getParmsFromFile();
        self.window = window;
        self.df_multiplier = 1;


    #---------------------------------------------------------------------------
    # myformat         
    #---------------------------------------------------------------------------
    def myformat(x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------

    def buttFilter(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------
    def buttLowpassFilter(self, data, cutoff, fs, order=5):
        b, a = self.buttFilter(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------
    def lowpassFilter(self, df, parmx):
        
        order = 6
        fs = 70.0       
        cutoff = 3.667
        
        cols = df.columns;
        for col in cols:
            if col in parmx:
                df[col] = self.buttLowpassFilter(df[col], cutoff, fs, order);
        
        return (df);
        

    #---------------------------------------------------------------------------
    # multDF
    #---------------------------------------------------------------------------
    def multDF(self, df):

        if self.df_multiplier == 1:
            return df;

        df = df.assign(Index=range(len(df))).set_index('Index')

        df_mult = pd.DataFrame();
        
        rows_list = [];
        for ind in df.index:
            for i in range(self.df_multiplier):
                rows_list.append(df.iloc[ind]);
        
        df_mult = pd.DataFrame(rows_list);
        return df_mult;

    #---------------------------------------------------------------------------
    # divDF
    #---------------------------------------------------------------------------
    def divDF(self, df):

        if self.df_multiplier == 1:
            return df;


        #index datetime 
        df['Index'] = df["datetime"];
        df.set_index('Index', inplace=True);

        #mean ve sloupcich "predict"        
        for col in df.columns:
            if "predict" in col:
                df[col] = df[col].groupby(['Index']).mean();
        
        df.reset_index(drop=True);
        #index integer        
        df = df.assign(Index=range(len(df))).set_index('Index')
        df_div = pd.DataFrame();
        
        rows_list = [];
        i = 0;
        for ind in df.index:
                if i < self.df_multiplier:
                    i += 1;
                    
                    if i == self.df_multiplier:
                        rows_list.append(df.iloc[ind]);
                        i = 0;
        
        df_div = pd.DataFrame(rows_list);
        return df_div;
    


    
    #---------------------------------------------------------------------------
    # getData
    #---------------------------------------------------------------------------
    def getData(self, shuffling=False, timestamp_start='2022-06-29 05:00:00', timestamp_stop='2022-07-01 23:59:59'):
        
        txdt_b = False;
        
        if((timestamp_start and timestamp_start.strip()) and (timestamp_stop and timestamp_stop.strip())):
            txdt_b = True;
        
        try:        
            self.DataTrainDim.DataTrain = None;
            files = os.path.join("./br_data", "tm-ai_2022*.csv");
            
            # list souboru pro join
            joined_list = glob.glob(files);
            
            # sort souboru pro join
            joined_list.sort(key=None, reverse=False);
            
            df = pd.concat([pd.read_csv(csv_file,
                                         sep=",|;", 
                                         engine='python',  
                                         header=0, encoding="utf-8",
                                       )
                                    for csv_file in joined_list],
                                    axis=0, 
                                    ignore_index=True
                    );
        # bordel pri domluve nazvoslovi...            
            df.columns = df.columns.str.lower();
        # merge csv   
        #   df.to_csv("./br_data/tm-ai_2022-00-01.csv", sep=";",  index=False, mode="w", encoding="utf-8");
        #   sys.exit();

            
        # vyber dat dle timestampu
            df["timestamp"] = pd.to_datetime(df["datetime"].str.slice(0, 18));

        # predikcni mnozina - pokus zvetsi testovaci mnozinu self.df_multiplier krat...      
            df_test = self.multDF(df[df["timestamp"].between(timestamp_start, timestamp_stop)]);
        # pokus - lowpass filter - nevyhodne zhorsuje presnost
        #   df_test = self.lowpassFilter(df_test, self.df_parmx);

        # treninkova a validacni mnozina    
            df = df[(df["timestamp"] < timestamp_start) | (df["timestamp"] > timestamp_stop)];
        # pokus - zvetsi treninkovou a validacni mnozinu self.df_multiplier krat...
        #   df = self.multDF(df);    
        # pokus - lowpass filter - nevyhodne zhorsuje presnost
        #   df = self.lowpassFilter(df, self.df_parmX);

            if self.window >= len(df_test) and txdt_b:
                print("Prilis maly vzorek dat pro predikci - exit(1)");
                sys.exit(1);
                
            df["index"] = pd.Index(range(0, len(df), 1));
            df.set_index("index", inplace=True);

            size = len(df.index)
            size_train = math.floor(size * 8 / 12)
            size_valid = math.floor(size * 4 / 12)
            size_test  = math.floor(size * 0 / 12)  

            if self.df_parmx == None or self.df_parmX == None:
                print("");
            else:
                self.DataTrainDim.DataTrain = self.setDataX(df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b,
                                                             shuffling=shuffling 
                                                        );
            

            return self.DataTrainDim(self.DataTrainDim.DataTrain);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());



        
    #-----------------------------------------------------------------------
    # saveDataResult  - result
    #-----------------------------------------------------------------------
    def saveDataResult(self, timestamp_start, model, typ, saveresult=True):
        
        filename = "./result/predicted_"+model+".csv"
        
        if "train" in typ:
            saveresult=True;
            
        
        if not saveresult:
            print("Vystupni soubor " + filename + " nevznikne !!!, saveresult = " +str(saveresult));
            return;
        else:
            print("Vystupni soubor " + filename + " vznikne.");
        
        try:
            col_names_y = list(self.DataTrain.df_parm_y);
            col_names_x = list(self.DataTrain.df_parm_x);
            
            col_names_predict = list("");
            col_names_drop    = list("");
            col_names_drop2   = list("");
            col_names_train   = list({"datetime"});
            col_names_dev     = list("");
            
            for col in col_names_y:
                col_names_train.append(col);
            
            for i in range(len(col_names_y)):
                if "dev" in col_names_y[i]:
                    col_names_predict.append(col_names_y[i]+"_predict");
                    col_names_dev.append(col_names_y[i]);
                else:    
                    col_names_drop.append(col_names_y[i]);

            for col in self.DataTrain.test.columns:
                if col in col_names_train:
                    a = 0;
                else:    
                    col_names_drop2.append(col);
                    
            
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

            
        try:
            self.DataTrain.test.reset_index(drop=True, inplace=True)
            
            df_result = pd.DataFrame();
            df_result  = pd.DataFrame(self.DataResultDim.DataResultX.y_result, columns = col_names_y);
            df_result.drop(col_names_drop, inplace=True, axis=1);
            df_result  = pd.DataFrame(np.array(df_result), columns =col_names_predict);
            
            df_result2 = pd.DataFrame();
            df_result2 = pd.DataFrame(self.DataTrain.test);
            df_result2.drop(col_names_drop2, inplace=True, axis=1);

            df_result  = pd.concat([df_result2, df_result], axis=1);
            df_result["txdat_group"] = timestamp_start;
            
            df_result = self.divDF(df_result);

            
            
            
            #df_result = df_result.assign(Index=range(len(df_result))).set_index('Index')
            
            # Absolute Error
            for col in col_names_dev:
                ae = (df_result[col] - df_result[col+"_predict"]);
                df_result[col+"_ae"] = ae;
            # Mean Squared Error
            for col in col_names_dev:
                mse = mean_squared_error(df_result[col],df_result[col+"_predict"]);
                df_result[col+"_mse"] = mse;
            
            
            
            path = Path(filename)

            if path.is_file():
                append = True;
            else:
                append = False;
        
            if append:             
                print(f'Soubor {filename} existuje - append', len(df_result));
                df_result.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
            else:
                print(f'Soubor {filename} neexistuje - create', len(df_result));
                df_result.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
        
        return;    
        
    #-----------------------------------------------------------------------
    # getParmsFromFile - nacte parametry z ./parms/parms.txt
    #-----------------------------------------------------------------------
    def getParmsFromFile(self):
        # otevreno v read mode
        
        parmfile = "ai-parms.txt"
        try:
            file = open(parmfile, "r")
            lines = file.readlines();
            
            count = 0
            for line in lines:
                line = line.replace("\n","").replace("'","").replace(" ","");
                line = line.strip();
                #if not line:
                x = line.startswith("df_parmx=")
                if x:
                    line = line.replace("df_parmx=", "").lower();
                    self.df_parmx = line.split(",");
                    if "null" in line:
                        self.df_parmx = None;
                        
                X = line.startswith("df_parmX=");
                if X:
                    line = line.replace("df_parmX=", "").lower();
                    self.df_parmX = line.split(",");
                    if "null" in line:
                        self.df_parmX = None;
            
                
            file.close();
            print("parametry nacteny z ", parmfile);       
            logging.info("parametry nacteny z "+ parmfile);                 
            
                
        except:
            print("Soubor parametru ", parmfile, " nenalezen!");                
            print("Parametry pro trenink site budou nastaveny implicitne v programu");                 
            logging.info("Soubor parametru " + parmfile + " nenalezen!");
        
        return();  
  
      
    #-----------------------------------------------------------------------
    # toTensorLSTM(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Pracujeme - li s rekurentnimi sitemi (LSTM GRU...), pak 
    # musíme vygenerovat dataset ve specifickém formátu.
    # Vystupem je 3D tenzor ve forme 'window' casovych kroku.
    #  
    # Jakmile jsou data vytvořena ve formě 'window' časových kroků, 
    # jsou nasledne prevedena do pole NumPy a reshapovana na 
    # pole 3D X_dataset.
    #
    # Funkce take vyrobi pole y_dataset, ktere muze byt pouzito pro 
    # simulaci modelu vstupnich dat, pokud tato data nejsou k dispozici.  
    # y_dataset predstavuje "window" časových rámců krat prvni prvek casoveho 
    # ramce pole X_dataset
    #
    # funkce vraci: X_dataset - 3D tenzor dat pro uceni site
    #               y_dataset - vektor vstupnich dat (model)
    #               dataset_cols - pocet sloupcu v datove sade. 
    #
    # poznamka: na konec tenzoru se pripoji libovolne 'okno' aby se velikost
    #           o toto okno zvetsila - vyresi se tim chybejici okno pri predikci
    #           
    #-----------------------------------------------------------------------
    
    def toTensorLSTM(dataset, window):
        
        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                            #jen v pripade ze vst. data nejsou definovana
                        
        values = dataset[0 : window, ];
        dataset = np.append(dataset, values, axis=0) #pridej delku okna
        dataset_rows, dataset_cols = dataset.shape;

        
        if window >= dataset_rows:
            print("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");
            logging.info("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");
        
        for i in range(window, dataset_rows):
            X_dataset.append(dataset[i - window : i, ]);
            y_dataset.append(dataset[i, ]);
        
        #doplnek pro append chybejicich window vzorku pri predikci
        X_dataset.append(dataset[0 : window, ]);
            
        X_dataset = np.array(X_dataset);
        y_dataset = np.array(y_dataset);
        
        X_dataset = np.reshape(X_dataset, (X_dataset.shape[0], X_dataset.shape[1], dataset_cols));
        
        return NeuronLayerLSTM.DataSet(X_dataset, y_dataset, dataset_cols);

    #-----------------------------------------------------------------------
    # fromTensorLSTM(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Poskladej vysledek vzdy z posledniho behu treninkove sady
    # a vrat vysledek o rozmeru [0: (dataset.shape[0] - 1)] krat [0 : dataset.shape[2]]
    # priklad: ma li tenzor rozmer 100 x 64 x 16, pak vrat vysledek [0:100-1], 64, [0,16-1]
    # funkce vraci: y_result - 2D array vysledku predikce
    #-----------------------------------------------------------------------
    def fromTensorLSTM(dataset):
        return(dataset[0 : (dataset.shape[0]),  (dataset.shape[1] - 1) , 0 : dataset.shape[2]]);
        
    #-----------------------------------------------------------------------
    # toTensorGRU(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Pracujeme - li s rekurentnimi sitemi (LSTM GRU...), pak 
    # musíme vygenerovat dataset ve specifickém formátu.
    # Vystupem je 3D tenzor ve forme 'window' casovych kroku.
    #  
    # Jakmile jsou data vytvořena ve formě 'window' časových kroků, 
    # jsou nasledne prevedena do pole NumPy a reshapovana na 
    # pole 3D X_dataset.
    #
    # Funkce take vyrobi pole y_dataset, ktere muze byt pouzito pro 
    # simulaci modelu vstupnich dat, pokud tato data nejsou k dispozici.  
    # y_dataset predstavuje "window" časových rámců krat prvni prvek casoveho 
    # ramce pole X_dataset
    #
    # funkce vraci: X_dataset - 3D tenzor dat pro uceni site
    #               y_dataset - vektor vstupnich dat (model)
    #               dataset_cols - pocet sloupcu v datove sade.
    # 
    # poznamka: na konec tenzoru se pripoji libovolne 'okno' aby se velikost
    #           o toto okno zvetsila - vyresi se tim chybejici okno pri predikci
    #           
    #-----------------------------------------------------------------------
    def toTensorGRU(dataset, window):
        
        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                        #jen v pripade ze vst. data nejsou definovana
                        

        values = dataset[0 : window, ];
        dataset = np.append(dataset, values, axis=0) #pridej delku okna
        dataset_rows, dataset_cols = dataset.shape;
        
        if window >= dataset_rows:
            print("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");
            logging.info("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");


        for i in range(window, dataset_rows):
            X_dataset.append(dataset[i - window : i, ]);
            y_dataset.append(dataset[i, ]);

        
        X_dataset = np.array(X_dataset);
        y_dataset = np.array(y_dataset);
        
        X_dataset = np.reshape(X_dataset, (X_dataset.shape[0], X_dataset.shape[1], dataset_cols));
        
        return NeuronLayerGRU.DataSet(X_dataset, y_dataset, dataset_cols);
    

    #-----------------------------------------------------------------------
    # fromTensorGRU(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Poskladej vysledek vzdy z posledniho behu treninkove sady
    # a vrat vysledek o rozmeru [0: (dataset.shape[0] - 1)] krat [0 : dataset.shape[2]]
    # priklad: ma li tenzor rozmer 100 x 64 x 16, pak vrat vysledek [0:100-1], 64, [0,16-1]
    # funkce vraci: y_result - 2D array vysledku predikce
    #-----------------------------------------------------------------------
    def fromTensorGRU(dataset):
        
        ds = dataset[0 : (dataset.shape[0]),  (dataset.shape[1] - 1) , 0 : dataset.shape[2]]
        return(ds);
        

    #-----------------------------------------------------------------------
    # toTensorBIDI(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Pracujeme - li s rekurentnimi sitemi (LSTM GRU, BIDI...), pak 
    # musíme vygenerovat dataset ve specifickém formátu.
    # Vystupem je 3D tenzor ve forme 'window' casovych kroku.
    #  
    # Jakmile jsou data vytvořena ve formě 'window' časových kroků, 
    # jsou nasledne prevedena do pole NumPy a reshapovana na 
    # pole 3D X_dataset.
    #
    # Funkce take vyrobi pole y_dataset, ktere muze byt pouzito pro 
    # simulaci modelu vstupnich dat, pokud tato data nejsou k dispozici.  
    # y_dataset predstavuje "window" časových rámců krat prvni prvek casoveho 
    # ramce pole X_dataset
    #
    # funkce vraci: X_dataset - 3D tenzor dat pro uceni site
    #               y_dataset - vektor vstupnich dat (model)
    #               dataset_cols - pocet sloupcu v datove sade.
    # 
    # poznamka: na konec tenzoru se pripoji libovolne 'okno' aby se velikost
    #           o toto okno zvetsila - vyresi se tim chybejici okno pri predikci
    #           
    #-----------------------------------------------------------------------
    def toTensorBIDI(dataset, window):
        
        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                        #jen v pripade ze vst. data nejsou definovana
                        
        
        values = dataset[0 : window, ];
        dataset = np.append(dataset, values, axis=0) #pridej delku okna
        dataset_rows, dataset_cols = dataset.shape;
        
        sh = dataset.shape;
        # inverze mnoziny treninkovych a validacnich dat
        for i in range(sh[1]):
            dataset[i:] = [x[::-1] for x in dataset[i:]];                
        
        if window >= dataset_rows:
            print("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");
            logging.info("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru");


        for i in range(window, dataset_rows):
            X_dataset.append(dataset[i - window : i, ]);
            y_dataset.append(dataset[i, ]);

        
        X_dataset = np.array(X_dataset);
        y_dataset = np.array(y_dataset);
        
        X_dataset = np.reshape(X_dataset, (X_dataset.shape[0], X_dataset.shape[1], dataset_cols));
        
        return NeuronLayerGRU.DataSet(X_dataset, y_dataset, dataset_cols);
    

    #-----------------------------------------------------------------------
    # fromTensorGRU(self, dataset, window = 64):
    #-----------------------------------------------------------------------
    # Poskladej vysledek vzdy z posledniho behu treninkove sady
    # a vrat vysledek o rozmeru [0: (dataset.shape[0] - 1)] krat [0 : dataset.shape[2]]
    # priklad: ma li tenzor rozmer 100 x 64 x 16, pak vrat vysledek [0:100-1], 64, [0,16-1]
    # funkce vraci: y_result - 2D array vysledku predikce
    #-----------------------------------------------------------------------
    def fromTensorBIDI(dataset):
        
        ds = dataset[0 : (dataset.shape[0]),  (dataset.shape[1] - 1) , 0 : dataset.shape[2]]
        return(ds);
        
    
    #---------------------------------------------------------------------------
    # DataFactory
    #---------------------------------------------------------------------------
    def prepareParmsPredict(self):
        
        i = 0;
        for i in self.df_parmX:
            df_parmX_predict[i] = self.df_parmX[i]+"_predict";
        i = 0;
             

    #---------------------------------------------------------------------------
    # setDataX(self, df,  size_train, size_valid, size_test)
    #---------------------------------------------------------------------------
    def setDataX(self, df, df_test,  size_train, size_valid, size_test, txdt_b=False, shuffling=False):
        #OSA XYZ
        try:

            DataTrain_x = self.DataTrain;
            DataTrain_x.train = pd.DataFrame(df[0 : size_train][self.df_parmX]);
            DataTrain_x.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmX]);
            
            if shuffling:
                DataTrain_x.train = DataTrain_x.train.reset_index(drop=True)
                DataTrain_x.train = shuffle(DataTrain_x.train)
                DataTrain_x.train = DataTrain_x.train.reset_index(drop=True)
                logging.info("--shuffle = True");
            
            DataTrain_x.test  = df_test;
            DataTrain_x.df_parm_x = self.df_parmx;  # data na ose x, pro rovinu X
            DataTrain_x.df_parm_y = self.df_parmX;  # data na ose y, pro rovinu Y
            DataTrain_x.axis = "OSA_XYZ";
            return(DataTrain_x);
    
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    
    #---------------------------------------------------------------------------
    # printGraf - kolekce dat         
    #---------------------------------------------------------------------------
    def printGraf(self, data, datalist, parm):
        ax = data.plot(y=datalist, kind='line', label=datalist)
        ax.get_figure().savefig(self.path_to_result+'/vstup_'+parm+'.pdf', format='pdf');
        plt.close(ax.get_figure());
                

    #---------------------------------------------------------------------------
    # DataFactory getter metody
    #---------------------------------------------------------------------------
    def getDf_parmx(self):
        return self.df_parmx;
    
    def getDf_parmX(self):
        return self.df_parmX;

#---------------------------------------------------------------------------
# MinimalRNNCell
#---------------------------------------------------------------------------
class MinimalRNNCell(layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]


#---------------------------------------------------------------------------
# GraphResult
#---------------------------------------------------------------------------
class GraphResult():

    def __init__(self, path_to_result, model, type, epochs, batch, units, shuffling, txdat1, txdat2, actf="tanh", printgraph=False):
        
        self.path_to_result = path_to_result; 
        self.model = model; 
        self.type = type;
        self.epochs = epochs;
        self.batch = batch;
        self.units = units;
        self.shuffling = shuffling;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        self.actf = actf;
        if "predict" in type:
            self.printgraph = True;
        else:    
            self.printgraph = printgraph;

    #---------------------------------------------------------------------------
    # smoothGraph - trochu vyhlad graf
    #---------------------------------------------------------------------------
    def smoothGraph(self, points, factor=0.9):
        smoothed_points = [];
        
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1];
                smoothed_points.append(previous * factor + point * (1 - factor));
            else:
                smoothed_points.append(point);
                
        return smoothed_points;        

    #---------------------------------------------------------------------------
    # groupAvg         
    #---------------------------------------------------------------------------
    def groupAvg(self, df):
        
        df.set_index('out')
        df_merge = df.groupby(['out']).mean();
        df_result = df.merge(df_merge, how='inner', on='out');
        df_result['diff'] = df_result['out'] - df_result['out_predict_y'];
        df_result['comp'] = (df_result['out_predict_y'] * -1);
        
        list_test   = df_result['out'];
        list_result = df_result['out_predict_y'];
        # report model error
        mse = mean_squared_error(list_test, list_result);
        mae = max_error(list_test, list_result);
        
        return(df_result, mse, mae);
        
        
    #---------------------------------------------------------------------------
    # printGraphCompare
    #---------------------------------------------------------------------------
    def printGraphCompare(self, DataResult, DataTrain, substract=True):
        
        axis = "";
        col_names_y="";
        col_names_x="";
        str_inp1 = ""; 
        col =  0;
        number_of_samples = 0;
        cmap = cm.get_cmap('winter') ;
        
        if not self.printgraph:
            print("Vystupni grafy se nebudou vyrabet - promenna  self.printgraph = " +str(self.printgraph));
            return;
        else:
            print("Vystupni grafy se budou vyrabet");

        try:
            try:
                axis = DataResult.axis
            except Exception as ex:
                print("POZOR !!! Patrne chyba v souboru parametru ai-parms.txt ");
                logging.error("POZOR !!! Patrne chyba v souboru parametru ai-parms.txt ");
                
        
            try:
                col_names_y = list(DataTrain.df_parm_y);
                col_names_x = list(DataTrain.df_parm_x);
            except Exception as ex:
                return;
            
                
    
        # List vstupnich hodnot to string  
            for elem in col_names_x:
                str_inp1 += elem+",\n";

            for i in col_names_y:
        # graf odchylek os X,Y

                if "12" in col_names_y[col] or "x" in col_names_y[col]:
                    axis="OSA_X"
                if "13" in col_names_y[col] or "y" in col_names_y[col]:
                    axis="OSA_Y"
                if "14" in col_names_y[col] or "z" in col_names_y[col]:
                    axis="OSA_Z"
                        
                
                if ("MACH" in col_names_y[col].upper() or "DEV" in col_names_y[col].upper()):
                    df_graph = pd.DataFrame(DataResult.y_test[ : ,col]);
                    df_graph['out'] = DataResult.y_test[ : , col];
                    df_graph['out_predict'] = DataResult.y_result[ : , col];
                    
                    df_graph, mse, mae  = self.groupAvg(df_graph);
                    number_of_samples = len(df_graph.index)

                    #zapis vysledku pro ai-printgraph.py
                    self.saveGraphResultToCsv(axis=axis, df_result=df_graph);
                    
                    max_axe = df_graph['out'].abs().max() * 2;
                    max_mae = df_graph['diff'].abs().max() * 3;

                    
                    
                
                    mpl.rcParams.update({'font.size': 6});
                    ax = df_graph.plot(y=['out', 'out_predict_y', 'comp'] ,
                                       kind='line',
                                       label=['realna data', 'predikce', 'kompenzace'],
                                       cmap=cmap
                               );
                    
                    ax.legend(title = self.model+' ' +axis+ 
                        '\nPath:\n' + self.path_to_result + 
                        '\nVst.:\n' + str_inp1 + 
                        '\nVyst:\n'     + col_names_y[col] +  
                        '\n\nMSE:'      + str(format(mse, '.9f')) + 
                        '\nMAE:'        + str(format(mae, '.9f'))+
                        '\ntxdat1:' + self.txdat1 + 
                        '\ntxdat2:' + self.txdat2, 
                        title_fontsize = 5,
                        fontsize = 5
                    );
                    ax.grid(which='both');
                    ax.grid(which='minor', alpha=0.2);
                    ax.grid(which='major', alpha=0.5);
                    ax.set_xlabel("vzorky")
                    ax.set_ylabel(col_names_y[col])
                    ax.set_ylim(-max_axe,+max_axe);
                    
                    ax.get_figure().savefig(self.path_to_result+'/compare_'+col_names_y[col]+'-'+axis+'.pdf', format='pdf');
                    plt.close(ax.get_figure())

                
                if substract and ("MACH" in col_names_y[col].upper() or "DEV" in col_names_y[col].upper()):
                    df_graph = df_graph.reset_index(drop=True);
                   
                    
                    mpl.rcParams.update({'font.size': 6})
                    ax = df_graph.plot(y=['diff'], kind='line', label=['realna - predikovana data'], grid=True);
                    ax.legend(title = self.model+' ' +axis+ 
                        '\nPath:\n' + self.path_to_result + 
                        '\nVst.:\n' + str_inp1 + 
                        '\nVyst:\n'     + col_names_y[col] +  
                        '\n\nMSE:'      + str(format(mse, '.9f')) + 
                        '\nMAE:'        + str(format(mae, '.9f'))+
                        '\ntxdat1:' + self.txdat1 + 
                        '\ntxdat2:' + self.txdat2, 
                        title_fontsize = 5,
                        fontsize = 5
                        );
                    ax.grid(which='both');
                    ax.grid(which='minor', alpha=0.2);
                    ax.grid(which='major', alpha=0.5);
                    ax.set_xlabel("vzorky");
                    ax.set_ylabel(col_names_y[col]);
                    ax.set_ylim(-max_mae,+max_mae);
                    ax.get_figure().savefig(self.path_to_result+'/substract_'+col_names_y[col]+'-'+axis+'.pdf', format='pdf');
                    plt.close(ax.get_figure())
                    
                if substract and ("MACH" in col_names_y[col].upper() or "DEV" in col_names_y[col].upper()):
                    self.saveToCSV(axis, col_names_y[col], str(format(mae, '.9f')), str(format(mse, '.9f')), str(number_of_samples), self.actf);
                    
                col = col + 1;
                
            return;
                
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
            
    #---------------------------------------------------------------------------
    # saveToCSV         
    #---------------------------------------------------------------------------
    def saveToCSV(self, axis, plcname, mae, mse, number_of_samples, actf ):
        
        header = ["result",
                  "axis",
                  "plcname",
                  "type",
                  "model",
                  "epochs",
                  "batch",
                  "units",
                  "shuffling",
                  "mae",
                  "mse",
                  "number_of_samples",
                  "txdat1",
                  "txdat2",
                  "actf"];

        shuff = "";
        if self.shuffling:
            shuff="_sh";
        else:    
            shuff="_nosh";
        
        filename = "./result/result_"+self.model+"_"+str(self.units)+shuff+"_"+self.actf+".csv"
       
        data = {"result"             : [self.path_to_result],
                "axis"               : [axis],
                "plcname"            : [plcname],
                "type"               : [self.type],
                "model"              : [self.model],
                "epochs"             : [self.epochs],
                "batch"              : [self.batch],
                "units"              : [self.units],
                "shuffling"          : [self.shuffling],
                "mae"                : [mae],
                "mse"                : [mse],
                "number_of_samples"  : [number_of_samples],
                "txdat1"             : [self.txdat1],
                "txdat2"             : [self.txdat2],
                "actf"               : [self.actf]
            }

        try:
        
            file_exists = os.path.isfile(filename) 

            if file_exists:
                # zapis
                df = pd.DataFrame(data, columns=header);
                df.to_csv(filename, mode='a', index=False, header=False);
            else:
                # vyrob a zapis
                df = pd.DataFrame(list(header));
                df.to_csv();
                
                df = pd.DataFrame(data, columns=header)
                df.to_csv(filename, mode='a',  index=False, header=True);
                
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
    
            
    #---------------------------------------------------------------------------
    # smoothGraph - trochu vyhlad graf
    #---------------------------------------------------------------------------
    def saveGraphResultToCsv(self, axis,  df_result):
        
        append = False;
        
        if len(df_result) > 200:
            return;
        
        shuff = "";
        if self.shuffling:
            shuff="_sh";
        else:    
            shuff="_nosh";
        

        
        filename = "./result/predict_"+axis.lower()+"_"+self.model+"_"+str(self.units)+shuff+"_"+self.actf+".csv"
        path = Path(filename)

        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            print(f'Soubor {filename} existuje - append', len(df_result));
            df_result.to_csv(filename, mode = "a", index=True, header=False);
        else:
            print(f'Soubor {filename} neexistuje - create', len(df_result));
            df_result.to_csv(filename, mode = "w", index=True, header=True);
                
        
          


#---------------------------------------------------------------------------
# Neuronova Vrstava DENSE
#---------------------------------------------------------------------------
class NeuronLayerDENSE():
    #definice datoveho ramce
    
    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch, txdat1, txdat2, window, units=256, shuffling=True, actf="tanh"):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        self.txdat1=txdat1;
        self.txdat2=txdat2;
        
        self.df = pd.DataFrame();
        self.df_out = pd.DataFrame();
        self.graph = None;
        self.data  = None;
        self.window = window;
        self.units = units;
        self.shuffling = shuffling;
        self.actf = actf;
        
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;

        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava DENSE 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEtrain(self, DataTrain):

        try:        
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0])
            out_size = len(y_train_data[0])
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler.pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler.pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler.pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler.pkl", 'wb'))
            
        # neuronova sit
            model = Sequential();
            model.add(tf.keras.Input(shape=(inp_size,)));
            model.add(layers.Dense(units=inp_size,   activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
#            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
#            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(out_size));
            
        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])
            
        # natrenuj model na vstupni dataset
            history = model.fit(x_train_data, 
                                y_train_data, 
                            epochs=self.epochs, 
                            batch_size=self.batch, 
                            verbose=2, 
                            validation_data=(x_valid_data, y_valid_data)
                            )
        
        
        #kolik se vynecha o zacatku v grafu?
            start_point = 0
            loss_train = history.history['loss'];
            loss_train = self.graph.smoothGraph(points = loss_train[start_point:], factor = 0.9)
            loss_val = history.history['val_loss'];
            loss_val = self.graph.smoothGraph(points = loss_val[start_point:], factor =  0.9)
            epochs = range(0,len(loss_train));
            plt.clf();
            plt.plot(epochs, loss_train, label='LOSS treninku');
            plt.plot(epochs, loss_val,   label='LOSS validace');
            plt.title('LOSS treninku '+ DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('LOSS');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_loss_'+DataTrain.axis+'.pdf', format='pdf');
            plt.clf();
        
            acc_train =  history.history['acc'];
            acc_train = self.graph.smoothGraph(points=acc_train[start_point:], factor = 0.9)
            acc_val = history.history['val_acc'];
            acc_val = self.graph.smoothGraph(points=acc_val[start_point:], factor = 0.9)
            epochs = range(0,len(acc_train));
            plt.clf();
            plt.plot(epochs, acc_train, label='ACC treninku');
            plt.plot(epochs, acc_val, label='ACC validace');
            plt.title('ACC treninku '+DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('ACC');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_acc_'+DataTrain.axis+'.pdf', format='pdf');
            plt.clf();

        #tf.saved_model.save(model, self.path_to_result+'/model')
            model.save('./models/model_'+self.model+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)
        
        # make predictions for the input data
            return (model);
    
            
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava DENSE predict
    #---------------------------------------------------------------------------
    # Zapis scaler 
    #   df = pd.DataFrame({'A':[1,2,3,7,9,15,16,1,5,6,2,4,8,9],
    #                      'B':[15,12,10,11,8,14,17,20,4,12,4,5,17,19],
    #                      'C':['Y','Y','Y','Y','N','N','N','Y','N','Y','N','N','Y','Y']})
    #   df[['A','B']] = min_max_scaler.fit_transform(df[['A','B']])  
    #   pickle.dump(min_max_scaler, open("scaler.pkl", 'wb'))
    # Nacti scaler
    #   scalerObj = pickle.load(open("scaler.pkl", 'rb'))
    #   df_test = pd.DataFrame({'A':[25,67,24,76,23],'B':[2,54,22,75,19]})
    #   df_test[['A','B']] = scalerObj.transform(df_test[['A','B']])
    #   
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_train_scaler.pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_train_scaler.pkl", 'rb'))
            
            x_test   = self.x_train_scaler.transform(x_test);
            
        # predict
            y_result = model.predict(x_test);
            y_result =self.y_train_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            print("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            print("          zkuste nejdrive --typ == train !!!");
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec_x(self, data, graph):
        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                print("Start vcetne treninku, model bude zapsan");
                logging.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrain);
            else:    
                print("Start bez treninku - model bude nacten");
                logging.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrain.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkDENSEpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataResult(self.txdat1, self.model, self.typ);
            graph.printGraphCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrain);
            stopTime = datetime.now();
            logging.info("cas vypoctu[s] %s",  str(stopTime - startTime));

            return();

        except FileNotFoundError as e:
            print(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
            
    #------------------------------------------------------------------------
    # neuralNetworkDENSEexec
    #------------------------------------------------------------------------
    def neuralNetworkDENSEexec(self):

        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result, window=self.window);
            self.graph = GraphResult(path_to_result=self.path_to_result, 
                                     model=self.model, 
                                     type=self.typ,
                                     epochs = self.epochs,
                                     batch = self.batch,
                                     units = self.units,
                                     shuffling = self.shuffling,
                                     txdat1 = self.txdat1,
                                     txdat2 = self.txdat2,
                                     actf = self.actf
                                );
        
            if self.typ == 'predict':
                self.shuffling = False;
        
            self.data.Data = self.data.getData(shuffling=self.shuffling, timestamp_start=self.txdat1, timestamp_stop=self.txdat2);

    # osa X    
            if self.data.DataTrainDim.DataTrain  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkDENSEexec_x(data=self.data, graph=self.graph);
            
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="DENSE", dest_path=self.path_to_result, data=self.data);
                
            return(0);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());


#---------------------------------------------------------------------------
# Neuronova Vrstava LSTM
#---------------------------------------------------------------------------
class NeuronLayerLSTM():
    #definice datoveho ramce
    

    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch, txdat1, txdat2, window, units=256, shuffling=True, actf="tanh"):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        
        self.df = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.graph = None;
        self.data  = None;
        self.window = window;
        self.units  = units;
        self.shuffling = shuffling;
        self.actf = actf;

        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;
        

    #---------------------------------------------------------------------------
    # Neuronova Vrstava LSTM
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMtrain(self, DataTrain):
        window_X = self.window;
        window_Y =  1;
        
        try:
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0]);
            out_size = len(y_train_data[0]);
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler.pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler.pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler.pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler.pkl", 'wb'))
        
        #data pro trenink -3D tenzor
            X_train =  DataFactory.toTensorLSTM(x_train_data, window=window_X);
        #vstupni data train 
            Y_train = DataFactory.toTensorLSTM(y_train_data, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = DataFactory.toTensorLSTM(x_valid_data, window=window_X);
        #vystupni data pro trenink -3D tenzor
            Y_valid = DataFactory.toTensorLSTM(y_valid_data, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
        # neuronova sit
            model = Sequential();
            model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            model.add(LSTM(units = self.units, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(LSTM(units = self.units, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(LSTM(units = self.units, return_sequences=True));
            model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj model na vstupni dataset
            history = model.fit(X_train.X_dataset, 
                                Y_train.X_dataset, 
                                epochs=self.epochs, 
                                batch_size=self.batch, 
                                verbose=2, 
                                validation_data=(X_valid.X_dataset,
                                                 Y_valid.X_dataset)
                            );

        # start point grafu - kolik se vynecha na zacatku
            start_point = 1;

            loss_train = history.history['loss'];
            loss_train = self.graph.smoothGraph(points = loss_train[start_point:], factor = 0.9);
            loss_val = history.history['val_loss'];
            loss_val = self.graph.smoothGraph(points = loss_val[start_point:], factor =  0.9);
            epochs = range(0,len(loss_train));
            plt.clf();
            plt.plot(epochs, loss_train, label='LOSS treninku');
            plt.plot(epochs, loss_val,   label='LOSS validace');
            plt.title('LOSS treninku '+ DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('LOSS');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_loss_'+DataTrain.axis+'.pdf', format='pdf');
        
            acc_train =  history.history['acc'];
            acc_train = self.graph.smoothGraph(points = acc_train[start_point:], factor = 0.9);
            acc_val = history.history['val_acc'];
            acc_val = self.graph.smoothGraph(points = acc_val[start_point:], factor = 0.9);
            epochs = range(0,len(acc_train));
            plt.clf();
            plt.plot(epochs, acc_train, label='ACC treninku');
            plt.plot(epochs, acc_val, label='ACC validace');
            plt.title('ACC treninku '+DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('ACC');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_acc_'+DataTrain.axis+'.pdf', format='pdf');
            
        # zapis modelu    
            model.save('./models/model_'+self.model+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)

        # make predictions for the input data
            return (model);
        
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
        
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava LSTM predict 
    #---------------------------------------------------------------------------

    def neuralNetworkLSTMpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_train_scaler.pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_train_scaler.pkl", 'rb'))
            
            x_test        = self.x_train_scaler.transform(x_test);
            
            x_object      = DataFactory.toTensorLSTM(x_test, window=self.window);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
            y_result = self.y_train_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            print("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            print("          zkuste nejdrive --typ == train !!!");
            traceback.print_exc();
            logging.error(traceback.print_exc());




    #---------------------------------------------------------------------------
    # neuralNetworkLSTMexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMexec_x(self, data, graph):
        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                print("Start vcetne treninku, model bude zapsan");
                logging.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkLSTMtrain(data.DataTrainDim.DataTrain);
            else:    
                print("Start bez treninku - model bude nacten");
                logging.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrain.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkLSTMpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataResult(self.txdat1, self.model, self.typ);
            graph.printGraphCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrain);
            stopTime = datetime.now();
            logging.info("cas vypoctu[s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #------------------------------------------------------------------------
    # neuralNetworkLSTMexec
    #------------------------------------------------------------------------
    def neuralNetworkLSTMexec(self):
        
        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result, window=self.window)
            self.graph = GraphResult(path_to_result=self.path_to_result, 
                                     model=self.model, 
                                     type=self.typ,
                                     epochs = self.epochs,
                                     batch = self.batch,
                                     units = self.units,
                                     shuffling = self.shuffling,
                                     txdat1 = self.txdat1,
                                     txdat2 = self.txdat2,
                                     actf = self.actf
                                );
        
            shuffling = False;
            if self.typ == 'predict':
                self.shuffling = False;
        
            self.data.Data = self.data.getData(shuffling=self.shuffling, timestamp_start=self.txdat1, timestamp_stop=self.txdat2);

    # osa X    
            if self.data.DataTrainDim.DataTrain  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkLSTMexec_x(data=self.data, graph=self.graph);
            
        
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="LSTM", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());




#---------------------------------------------------------------------------
# Neuronova Vrstava GRU
#---------------------------------------------------------------------------
class NeuronLayerGRU():
    #definice datoveho ramce
    

    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch, txdat1, txdat2, window, units=256, shuffling=True, actf="tanh"):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        
        self.df = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.graph = None;
        self.data  = None;
        self.window = window;
        self.units = units;
        self.shuffling = shuffling;
        self.actf = actf;
        
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;

        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava GRU 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUtrain(self, DataTrain):
        window_X = self.window;
        window_Y =  1;
        
        try:
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0]);
            out_size = len(y_train_data[0]);

        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler.pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler.pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler.pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler.pkl", 'wb'))
            
        
        #data pro trenink -3D tenzor
            X_train =  DataFactory.toTensorGRU(x_train_data, window=window_X);
        #vstupni data train 
            Y_train = DataFactory.toTensorGRU(y_train_data, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = DataFactory.toTensorGRU(x_valid_data, window=window_X);
        #vystupni data pro trenink -3D tenzor
            Y_valid = DataFactory.toTensorGRU(y_valid_data, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
        # neuronova sit
            model = Sequential();
            model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            model.add(GRU(units = self.units, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(GRU(units = self.units, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(GRU(units = self.units, return_sequences=True));
            model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj model na vstupni dataset
            history = model.fit(X_train.X_dataset, 
                                Y_train.X_dataset, 
                                epochs=self.epochs, 
                                batch_size=self.batch, 
                                verbose=2, 
                                validation_data=(X_valid.X_dataset,
                                                 Y_valid.X_dataset)
                            );

        # start point grafu - kolik se vynecha na zacatku
            start_point = 1;

            loss_train = history.history['loss'];
            loss_train = self.graph.smoothGraph(points = loss_train[start_point:], factor = 0.9);
            loss_val = history.history['val_loss'];
            loss_val = self.graph.smoothGraph(points = loss_val[start_point:], factor =  0.9);
            epochs = range(0,len(loss_train));
            plt.clf();
            plt.plot(epochs, loss_train, label='LOSS treninku');
            plt.plot(epochs, loss_val,   label='LOSS validace');
            plt.title('LOSS treninku '+ DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('LOSS');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_loss_'+DataTrain.axis+'.pdf', format='pdf');
        
            acc_train =  history.history['acc'];
            acc_train = self.graph.smoothGraph(points = acc_train[start_point:], factor = 0.9);
            acc_val = history.history['val_acc'];
            acc_val = self.graph.smoothGraph(points = acc_val[start_point:], factor = 0.9);
            epochs = range(0,len(acc_train));
            plt.clf();
            plt.plot(epochs, acc_train, label='ACC treninku');
            plt.plot(epochs, acc_val, label='ACC validace');
            plt.title('ACC treninku '+DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('ACC');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_acc_'+DataTrain.axis+'.pdf', format='pdf');
            
        # zapis modelu    
            model.save('./models/model_'+self.model+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)

        # make predictions for the input data
            return (model);
        
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # Neuronova Vrstava GRU predict 
    #---------------------------------------------------------------------------

    def neuralNetworkGRUpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_train_scaler.pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_train_scaler.pkl", 'rb'))
            
            x_test        = self.x_train_scaler.transform(x_test);
            
            x_object      = DataFactory.toTensorGRU(x_test, window=self.window);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
            y_result = self.y_train_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            print("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            print("          zkuste nejdrive --typ == train !!!");
            traceback.print_exc();
            logging.error(traceback.print_exc());

            
    #---------------------------------------------------------------------------
    # neuralNetworkGRUexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUexec_x(self, data, graph):
        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                print("Start vcetne treninku, model bude zapsan");
                logging.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkGRUtrain(data.DataTrainDim.DataTrain);
            else:    
                print("Start bez treninku - model bude nacten");
                logging.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrain.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkGRUpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataResult(self.txdat1, self.model, self.typ);
            graph.printGraphCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrain);
            stopTime = datetime.now();
            logging.info("cas vypoctu[s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #------------------------------------------------------------------------
    # neuralNetworkGRUexec
    #------------------------------------------------------------------------
    def neuralNetworkGRUexec(self):
        
        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result, window=self.window);
            self.graph = GraphResult(path_to_result=self.path_to_result, 
                                     model=self.model, 
                                     type=self.typ,
                                     epochs = self.epochs,
                                     batch = self.batch,
                                     units = self.units,
                                     shuffling = self.shuffling,
                                     txdat1 = self.txdat1,
                                     txdat2 = self.txdat2,
                                     actf = self.actf
                                );
        
            shuffling = False;
            if self.typ == 'predict':
                self.shuffling = False;
        
            self.data.Data = self.data.getData(shuffling=self.shuffling, timestamp_start=self.txdat1, timestamp_stop=self.txdat2);

    # osa X    
            if self.data.DataTrainDim.DataTrain  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkGRUexec_x(data=self.data, graph=self.graph);
            
        
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="GRU", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());



#---------------------------------------------------------------------------
# Neuronova Vrstava BIDIRECTIONAL RNN
#---------------------------------------------------------------------------
class NeuronLayerBIDI():
    #definice datoveho ramce
    

    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch, txdat1, txdat2, window, units=256, shuffling=True, actf="tanh"):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        
        self.df = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.graph = None;
        self.data  = None;
        self.window = window;
        self.units = units;
        self.shuffling = shuffling;
        self.actf = actf;
        
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava BIDI 
    #---------------------------------------------------------------------------
    def neuralNetworkBIDItrain(self, DataTrain):
        window_X = self.window;
        window_Y =  1;
        
        try:
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0]);
            out_size = len(y_train_data[0]);

        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler.pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler.pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler.pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler.pkl", 'wb'))
            
        
        #data pro trenink -3D tenzor
            X_train =  DataFactory.toTensorBIDI(x_train_data, window=window_X);
        #vstupni data train 
            Y_train = DataFactory.toTensorBIDI(y_train_data, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = DataFactory.toTensorBIDI(x_valid_data, window=window_X);
        #vystupni data pro trenink -3D tenzor
            Y_valid = DataFactory.toTensorBIDI(y_valid_data, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
        # neuronova sit
            model = Sequential();
            model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            model.add(layers.Bidirectional(GRU(units = self.units, return_sequences=True)));
            model.add(Dropout(0.2));
            model.add(layers.Bidirectional(GRU(units = self.units, return_sequences=True)));
            model.add(Dropout(0.2));
            model.add(layers.Bidirectional(GRU(units = self.units, return_sequences=True)));
            model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj model na vstupni dataset
            history = model.fit(X_train.X_dataset, 
                                Y_train.X_dataset, 
                                epochs=self.epochs, 
                                batch_size=self.batch, 
                                verbose=2, 
                                validation_data=(X_valid.X_dataset,
                                                 Y_valid.X_dataset)
                            );

        # start point grafu - kolik se vynecha na zacatku
            start_point = 1;

            loss_train = history.history['loss'];
            loss_train = self.graph.smoothGraph(points = loss_train[start_point:], factor = 0.9);
            loss_val = history.history['val_loss'];
            loss_val = self.graph.smoothGraph(points = loss_val[start_point:], factor =  0.9);
            epochs = range(0,len(loss_train));
            plt.clf();
            plt.plot(epochs, loss_train, label='LOSS treninku');
            plt.plot(epochs, loss_val,   label='LOSS validace');
            plt.title('LOSS treninku '+ DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('LOSS');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_loss_'+DataTrain.axis+'.pdf', format='pdf');
        
            acc_train =  history.history['acc'];
            acc_train = self.graph.smoothGraph(points = acc_train[start_point:], factor = 0.9);
            acc_val = history.history['val_acc'];
            acc_val = self.graph.smoothGraph(points = acc_val[start_point:], factor = 0.9);
            epochs = range(0,len(acc_train));
            plt.clf();
            plt.plot(epochs, acc_train, label='ACC treninku');
            plt.plot(epochs, acc_val, label='ACC validace');
            plt.title('ACC treninku '+DataTrain.axis);
            plt.xlabel('Pocet epoch');
            plt.ylabel('ACC');
            plt.legend();
            plt.savefig(self.path_to_result+'/graf_acc_'+DataTrain.axis+'.pdf', format='pdf');
            
        # zapis modelu    
            model.save('./models/model_'+self.model+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)

        # make predictions for the input data
            return (model);
        
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());



    #---------------------------------------------------------------------------
    # Neuronova Vrstava BIDI predict 
    #---------------------------------------------------------------------------

    def neuralNetworkBIDIpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_train_scaler.pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_train_scaler.pkl", 'rb'))
            
            x_test        = self.x_train_scaler.transform(x_test);
            
            x_object      = DataFactory.toTensorBIDI(x_test, window=self.window);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
            y_result = self.y_train_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            print("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            print("          zkuste nejdrive --typ == train !!!");
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkBIDIexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkBIDIexec_x(self, data, graph):
        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                print("Start vcetne treninku, model bude zapsan");
                logging.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkBIDItrain(data.DataTrainDim.DataTrain);
            else:    
                print("Start bez treninku - model bude nacten");
                logging.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrain.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkBIDIpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataResult(self.txdat1, self.model, self.typ);
            graph.printGraphCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrain);
            stopTime = datetime.now();
            logging.info("cas vypoctu[s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #------------------------------------------------------------------------
    # neuralNetworkBIDIexec
    #------------------------------------------------------------------------
    def neuralNetworkBIDIexec(self):
        
        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result, window=self.window);
            self.graph = GraphResult(path_to_result=self.path_to_result, 
                                     model=self.model, 
                                     type=self.typ,
                                     epochs = self.epochs,
                                     batch = self.batch,
                                     units = self.units,
                                     shuffling = self.shuffling,
                                     txdat1 = self.txdat1,
                                     txdat2 = self.txdat2,
                                     actf = self.actf
                                );
        
            shuffling = False;
            if self.typ == 'predict':
                self.shuffling = False;
        
            self.data.Data = self.data.getData(shuffling=self.shuffling, timestamp_start=self.txdat1, timestamp_stop=self.txdat2);

    # osa X    
            if self.data.DataTrainDim.DataTrain  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkBIDIexec_x(data=self.data, graph=self.graph);
            
        
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="BIDI", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());



#------------------------------------------------------------------------
# saveModelToArchiv - zaloha modelu, spusteno jen pri parametru train
#------------------------------------------------------------------------
def saveModelToArchiv(model, dest_path, data):

    axes = np.array([data.DataTrainDim.DataTrain.axis]);
    src_dir  = "./models/model_"+model+"_";
    dest_dir = "/models/model_"+model+"_";
    try:
        if data.DataTrainDim.DataTrain  == None:
            print()
        else:    
            src_dir_  = src_dir + axes[0]
            dest_dir_ = dest_path + dest_dir + axes[0]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
            
        return(0);    
   
    except Exception as ex:
        traceback.print_exc();
        logging.error(traceback.print_exc());
 
#------------------------------------------------------------------------
# MAIN CLASS
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# setEnv
#------------------------------------------------------------------------
def setEnv(path, model, type, parms):

        progname = os.path.basename(__file__);
        current_date =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S");
        path1 = path+model+"_3D";
        path2 = path1+"/"+current_date+"_"+type
                
        
        try: 
            os.mkdir("./log");
        except OSError as error: 
            print(); 

        try: 
            os.mkdir("./result")
        except OSError as error: 
            print(); 

        try: 
            os.mkdir("./temp")
        except OSError as error: 
            print(); 

        try: 
            os.mkdir(path1);
        except OSError as error: 
            print(); 
        
        try: 
            os.mkdir(path2);
        except OSError as error: 
            print();
            
        try: 
            os.mkdir(path2+"/src");
        except OSError as error: 
            print();
            
        try: 
            os.mkdir("./models");
        except OSError as error: 
            print();

        try:
            shutil.copy(progname, path2+"/src");
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii zdrojoveho kodu.", error)
        except:
            print("Chyba pri kopii zdrojoveho kodu.")

        try:
            shutil.copy("ai-parms.txt", path2+"/src");
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii ai-parms.txt.", error)
        except:
            print("Chyba pri kopii ai-parms.txt.")
            
            
        try:
            print(parms,  file=open( path2+"/src/parms.txt", "w"))    
        except:
            print("Chyba pri zapisu parametru parms.txt.")
        
        
        logging.basicConfig(filename='./log/'+progname+'.log',
            filemode='a',level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');
            
        
    
        return path2    

#------------------------------------------------------------------------
# Exception handler
#------------------------------------------------------------------------
def exception_handler(exctype, value, tb):
    logging.error(exctype)
    logging.error(value)
    logging.error(traceback.extract_tb(tb))

#------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------
def help (activations):
    print ("HELP:");
    print ("------------------------------------------------------------------------------------------------------ ");
    print ("pouziti: <nazev_programu> <arg1> <arg2> <arg3> <arg4>");
    print ("ai-neuro.py -t <--typ> -m <--model> -e <--epochs> -b <--batch> ")
    print (" ");
    print ("        --help            list help ")
    print ("        --typ             typ behu 'train' nebo 'predict'")
    print ("                                 train - trenink site")
    print ("                                 predict - beh z nauceneho algoritmu")
    print (" ");
    print ("        --model           model neuronove site 'DENSE', 'LSTM', 'GRU', 'BIDI'")
    print ("                                 DENSE - zakladni model site - nejmene narocny na system")
    print ("                                 LSTM - Narocny model rekurentni site s feedback vazbami")
    print ("                                 GRU  - Narocny model rekurentni hradlove site")
    print ("                                 BIDI - Narocny model rekurentni hradlove site, bidirectional rezim")
    print (" ");
    print ("        --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
    print ("                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
    print ("                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
    print ("                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
    print ("                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
    print (" ");
    print ("        --batch           pocet vzorku v minidavce - cislo v intervalu <32,2048>")
    print ("                                 Velikost dávky je počet hodnot vstupních dat, které zavádíte najednou do modelu.")
    print ("                                 Mějte prosím na paměti, že velikost dávky ovlivňuje"); 
    print ("                                 dobu tréninku, chybu, které dosáhnete, posuny gradientu atd."); 
    print ("                                 Neexistuje obecné pravidlo, jaká velikost dávky funguje nejlépe.");
    print ("                                 Stačí vyzkoušet několik velikostí a vybrat si tu, která dava");
    print ("                                 nejlepsi vysledky. Snažte se pokud mozno nepoužívat velké dávky,");
    print ("                                 protože by to přeplnilo pamet. ");
    print ("                                 Bezne velikosti minidavek jsou 32, 64, 128, 256, 512, 1024, 2048.");
    print (" ");
    print ("                                 Plati umera: cim vetsi davka tim vetsi naroky na pamet.");
    print ("                                              cim vetsi davka tim rychlejsi zpracovani.");
    print ("        --units           pocet vypocetnich jednotek cislo v intervalu <32,1024>")
    print ("                                 Pocet vypocetnich jednotek urcuje pocet neuronu zapojenych do vypoctu.")
    print ("                                 Mějte prosím na paměti, že velikost units ovlivňuje"); 
    print ("                                 dobu tréninku, chybu, které dosáhnete, posuny gradientu atd."); 
    print ("                                 Neexistuje obecné pravidlo, jak urcit optimalni velikost parametru units.");
    print ("                                 Obecne plati, ze maly pocet neuronu vede k nepresnym vysledkum a naopak");
    print ("                                 velky pocet units muze zpusobit preuceni site - tedy stejny efekt jako pri");
    print ("                                 nedostatecnem poctu units. Pamatujte, ze pocet units vyrazne ovlivnuje alokaci");
    print ("                                 pameti. pro 1024 units je treba minimalne 32GiB u siti typu LSTM, GRU nebo BIDI.");
    print (" ");
    print ("                                 Plati umera: cim vetsi units tim vetsi naroky na pamet.");
    print ("                                              cim vetsi units tim pomalejsi zpracovani.");
    print (" ");
    print ("        --shuffle         Nahodne promichani treninkovych dat  <True, False>")
    print ("                                 Nahodnym promichanim dat se docili nezavislosti na casove ose.")
    print ("                                 V nekterych pripadech je tato metoda velmi vyhodna."); 
    print ("                                 shuffle = True se uplatnuje jen v rezimu 'train' a pouze na treninkova"); 
    print ("                                 data. Validacni a testovaci data se nemichaji."); 
    print ("                                 Pokud shuffle neni uveden, je implicitne nastaven na 'True'."); 
    print (" ");
    print ("        --actf            Aktivacni funkce - jen pro parametr DENSE")
    print ("                                 U LSTM, GRU a BIDI se neuplatnuje.")
    print ("                                 Pokud actf neni uvedan, je implicitne nastaven na 'tanh'."); 
    print ("                                 U site GRU, LSTM a BIDI je implicitne nastavena na 'tanh' ");
    print (" ");
    print (" ");
    print ("        --txdat1          timestamp zacatku datove mnoziny pro predict, napr '2022-04-09 08:00:00' ")
    print (" ");
    print ("        --txdat2          timestamp konce   datove mnoziny pro predict, napr '2022-04-09 12:00:00' ")
    print (" ");
    print ("                                 parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere");
    print ("                                 se v uvahu cela mnozina dat k trenovani.");
    print (" ");
    print (" ");
    print (" ");
    print ("priklad: ./ai-neuro.py -t train, -m DENSE, -e 64 -b 128 -s True -af sigmoid -t1 2022-04-09 08:00:00 -t2 2022-04-09 12:00:00");
    print ("nebo:    ./ai-neuro.py --typ train, --model DENSE, --epochs 64 --batch 128 --shuffle True --actf=sigmoid --txdat1 2022-04-09 08:00:00 --txdat2 2022-04-09 12:00:00");
    print('parametr --epochs musi byt cislo typu int <1, 256>')
    print ("POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM, GRU nebo BIDI!!!");
    print ("       pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise ");
    print ("       natrenovany model site..");
    print ("       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.");
    print ("       spusti normalni beh site z jiz natrenovaneho modelu.");
    print ("       Takze: budte trpelivi...");
    print (" ");
    print (" ");
    print ("Vstupni parametry: ");
    print ("  pokud neexistuje v rootu aplikace soubor ai-parms.txt, pak jsou parametry implicitne");
    print ("  prirazeny z promennych definovanych v programu:");
    print ("Jedna se o tyto promenne: ");
    print (" ");
    print ("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print ("  df_parmx = ['machinedata_m0412','teplota_pr01', 'x_temperature']");
    print (" ");
    print ("  #Tenzor predlozeny k uceni site");
    print ("  df_parmX = ['machinedata_m0112','machinedata_m0212','machinedata_m0312','machinedata_m0412','teplota_pr01', 'x_temperature'];");
    print (" ");
    print ("Pokud pozadujete zmenu parametu j emozno primo v programu poeditovat tyto promenne ");
    print (" ");
    print ("a nebo vyrobit soubor ai-parms.txt s touto syntaxi ");
    print ("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print ("  df_parmx = machinedata_m0412,teplota_pr01,x_temperature'");
    print (" ");
    print ("  #Tenzor predlozeny k uceni site");
    print ("  df_parmX = machinedata_m0412,teplota_pr01, x_temperature");
    print (" ");
    print ("a ten nasledne ulozit v rootu aplikace. (tam kde je pythonovsky zdrojak. ");
    print ("POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy vstupniho datoveho CSV souboru (nebo souboruuu)");
    print ("a muzou tam byt i uvozovky: priklad: 'machinedata_m0112','machinedata_m0212', to pro snazsi copy a paste ");
    print ("z datoveho CSV souboru. ");
    print (" ");
    print ("(C) GNU General Public License, autor Petr Lukasik , 2022 ");
    print (" ");
    print ("Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)");
    print ("              miniconda3,");
    print ("              python 3.9, tensorflow 2.8, mathplotlib,  ");
    print ("              tensorflow 2.8,");
    print ("              mathplotlib,  ");
    print ("              scikit-learn-intelex,  ");
    print ("              pandas,  ");
    print ("              numpy,  ");
    print ("              keras   ");
    print (" ");
    print (" ");
    print ("Povolene aktivacni funkce: ");
    print(tabulate(activations, headers=['Akt. funkce', 'Popis']));

    return();

#------------------------------------------------------------------------
# kontrola zda byla zadana platna aktivacni funkce 
# ze seznamu activations...
#------------------------------------------------------------------------
def checkActf(actf, activations):

    for i in activations:
        if i[0] in actf:
            return(True);

    return(False);
    
   
#------------------------------------------------------------------------
# main
#------------------------------------------------------------------------
def main(argv):
    
    global path_to_result;
    path_to_result = "./result";
    
    global g_window;
    g_window = 48;
    
    startTime = datetime.now();

    activations = [["deserialize","Returns activation function given a string identifier"],
                   ["elu","Exponential Linear Unit"],
                   ["exponential","Exponential activation function"],
                   ["gelu","Gaussian error linear unit (GELU) activation function"],
                   ["get","Returns function"],
                   ["hard_sigmoid","Hard sigmoid activation function"],
                   ["linear","Linear activation function (pass-through)"],
                   ["relu","Rectified linear unit activation function"],
                   ["selu","Scaled Exponential Linear Unit"],
                   ["serialize","Returns the string identifier of an activation function"],
                   ["sigmoid","Sigmoid activation function: sigmoid(x) = 1 / (1 + exp(-x))"],
                   ["softmax","Softmax converts a vector of values to a probability distribution"],
                   ["softplus","Softplus activation function: softplus(x) = log(exp(x) + 1)"],
                   ["softsign","Softsign activation function: softsign(x) = x / (abs(x) + 1)"],
                   ["swish","Swish activation function: swish(x) = x * sigmoid(x)"],
                   ["tanh","Hyperbolic tangent activation function"]];
    
    try:
        parm0 = sys.argv[0];
     
        parm1 = "";
        parm2 = "";
        parm3 = 0;
        parm4 = 0;
        parm5 = 0;
        txdat1 = "";
        txdat2 = "";
        shuffling = True;
        actf = "tanh";

        txdat_format = "%Y-%m-%d %h:%m:%s"
        try:
            opts, args = getopt.getopt(sys.argv[1:],"ht:m:e:b:u:s:af:t1:t2:h:x",["typ=", "model=", "epochs=", "batch=", "units=", "shuffle=","actf=", "txdat1=","txdat2=", "help="])
        except getopt.GetoptError:
            print("Chyba pri parsovani parametru:");
            help(activations);
            
        for opt, arg in opts:
            if opt in ("-t", "--typ"):
                parm1 = arg;
            elif opt in ("-m", "--model"):
                parm2 = arg.upper();
            elif opt in ("-e", "--epochs"):
                try:
                    r = range(32-1, 256+1);
                    parm3 = int(arg);
                    if parm3 not in r:
                        print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <32, 256>");
                        help(avtivations);
                        sys.exit(1)    
                        
                except:
                    print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <32, 256>");
                    help(activations);
                    sys.exit(1)    
            elif opt in ("-b", "--batch"):
                try:
                    r = range(32-1, 2048+1);
                    parm4 = int(arg);
                    if parm4 not in r:
                        print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <32, 2048>");
                        help(activations);
                        sys.exit(1)    
                except:    
                    print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <32, 2048>");
                    help(activations);
                    sys.exit(1)
                    
            elif opt in ("-u", "--units"):
                try:
                    r = range(32-1, 2048+1);
                    parm5 = int(arg);
                    if parm5 not in r:
                        print("Chyba pri parsovani parametru: parametr 'units' musi byt cislo typu integer v rozsahu <32, 2048>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Chyba pri parsovani parametru: parametr 'units' musi byt cislo typu integer v rozsahu <32, 2048>");
                    help(activations);
                    sys.exit(1);

            elif opt in ["-af","--actf"]:
                actf = arg.lower();
                if parm2 == "DENSE":
                    if not checkActf(actf, activations):
                        print("Chybna aktivacni funkce - viz help...");
                        help(activations)
                        sys.exit(1);

            elif opt in ["-t1","--txdat1"]:
                txdat1 = arg;
                if txdat1:
                    try:
                        res = bool(parser.parse(txdat1));
                    except ValueError:
                        print("Chyba formatu txdat1, musi byt YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    

            elif opt in ["-t2","--txdat2"]:
                txdat2 = arg;
                if txdat2:
                    try:
                        res = bool(parser.parse(txdat2));
                    except ValueError:
                        print("Chyba formatu txdat2, musi byt YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    

            elif opt in ["-s","--shuffle"]:
                if arg.upper() == "TRUE":
                    shuffling = True;
                else:
                    shuffling = False;    

            elif opt in ["-h","--help"]:
                help(activations);
                sys.exit(0);

        
        if len(sys.argv) < 8:
            help(activations);
            sys.exit(1);
        
        parms = "start s parametry: typ="+parm1+\
                " model="+parm2+\
                " epochs="+str(parm3)+\
                " batch="+str(parm4)+\
                " units="+str(parm5)+\
                " shuffle="+str(shuffling)+\
                " txdat1="+txdat1+\
                " txdat2="+txdat2+\
                " actf="+actf; 
                    
        logging.info(parms);
        path_to_result = setEnv(path=path_to_result, model=parm2, type=parm1, parms=parms);

        startTime = datetime.now();
        logging.info("start...");
        logging.info("Verze TensorFlow :" + tf.__version__)
        print("Verze TensorFlow :", tf.__version__)
        
        if parm2 == "LSTM":
            actf = "tanh";
            neural = NeuronLayerLSTM(path_to_result=path_to_result, 
                                     typ=parm1, 
                                     model=parm2, 
                                     epochs=parm3, 
                                     batch=parm4,
                                     txdat1=txdat1,
                                     txdat2=txdat2,
                                     window=g_window,
                                     units=parm5,
                                     shuffling=shuffling, 
                                     actf=actf 
                                );
            neural.neuralNetworkLSTMexec();
            
        elif parm2 == "DENSE":
            neural = NeuronLayerDENSE(path_to_result=path_to_result, 
                                     typ=parm1, 
                                     model=parm2, 
                                     epochs=parm3, 
                                     batch=parm4,
                                     txdat1=txdat1,
                                     txdat2=txdat2,
                                     window=g_window,
                                     units=parm5,
                                     shuffling=shuffling,
                                     actf=actf 
                                );
            neural.neuralNetworkDENSEexec();
            
        elif parm2 == 'GRU':
            actf = "tanh";
            neural = NeuronLayerGRU(path_to_result=path_to_result, 
                                     typ=parm1, 
                                     model=parm2, 
                                     epochs=parm3, 
                                     batch=parm4,
                                     txdat1=txdat1,
                                     txdat2=txdat2,
                                     window=g_window,
                                     units=parm5,
                                     shuffling=shuffling,  
                                     actf=actf 
                                );
            neural.neuralNetworkGRUexec();
            
        elif parm2 == 'BIDI':
            actf = "tanh";
            neural = NeuronLayerBIDI(path_to_result=path_to_result, 
                                     typ=parm1, 
                                     model=parm2, 
                                     epochs=parm3, 
                                     batch=parm4,
                                     txdat1=txdat1,
                                     txdat2=txdat2,
                                     window=g_window,
                                     units=parm5,
                                     shuffling=shuffling,  
                                     actf=actf 
                                );
            neural.neuralNetworkBIDIexec();
            
            
    
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        logging.error(traceback.print_exc());
        help(activations);
        
    finally:    
        stopTime = datetime.now();
        print("cas vypoctu [s]", stopTime - startTime );
        logging.info("cas vypoctu [s] %s",  str(stopTime - startTime));
        logging.info("stop...");
        sys.exit(0);


#------------------------------------------------------------------------
# main entry point
#------------------------------------------------------------------------
        
if __name__ == "__main__":

    main(sys.argv[1:])
    

    


