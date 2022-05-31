#!/usr/bin/python3

#------------------------------------------------------------------------------
# ai-neuro3D 
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
#       conda create -n tf python=3.9
#
#  4. aktivovat behove prostredi tf (pred tim je nutne zevrit a znovu
#     otevrit terminal aby se conda aktivovala.
#       conda activate  tf
#
#  5. instalovat tyto moduly (pomoci conda)
#       conda install tensorflow 
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

from sklearn.preprocessing import MinMaxScaler;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import max_error;
from sklearn.utils import shuffle
from numpy import asarray;
#from matplotlib import pyplot;
from dataclasses import dataclass;
from datetime import datetime

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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
#from keras.utils.vis_utils import plot_model
from matplotlib import cm;
from datetime import datetime
from _cffi_backend import string


class DataFactory():
    

    df_parmX_predict = [];
    df_parmY_predict = [];
    df_parmZ_predict = [];
    
    @dataclass
    class DataTrainX:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z

    @dataclass
    class DataTrainY:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z

    @dataclass
    class DataTrainZ:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z

    @dataclass
    class DataTrainDim:
          DataTrainX: object;
          DataTrainY: object;
          DataTrainZ: object;
          
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
          DataResultY: object;
          DataResultZ: object;

    def __init__(self, path_to_result):
        
    #Vystupni list parametru - co budeme chtit po siti predikovat
        self.df_parmx = ['machinedata_m0412','teplota_pr01', 'x_temperature']
        self.df_parmy = ['machinedata_m0413','teplota_pr02', 'y_temperature']
        self.df_parmz = ['machinedata_m0414','teplota_pr03', 'z_temperature']
        
    #Tenzor predlozeny k uceni site
        self.df_parmX = ['machinedata_m0412','teplota_pr01', 'x_temperature']
        self.df_parmY = ['machinedata_m0413','teplota_pr02', 'y_temperature']
        self.df_parmZ = ['machinedata_m0414','teplota_pr03', 'z_temperature']
        
        self.path_to_result = path_to_result;
        self.getParmsFromFile();
        
    #---------------------------------------------------------------------------
    # DataFactory
    #---------------------------------------------------------------------------
    def getData(self, shuffling=False, timestamp_start='2022-04-09 08:00:00', timestamp_stop='2022-04-09 13:00:00'):
        
        txdt_b = False;
        
        if((timestamp_start and timestamp_start.strip()) and (timestamp_stop and timestamp_stop.strip())):
            txdt_b = False;
        
        try:        
            self.DataTrainDim.DataTrainX = None;
            self.DataTrainDim.DataTrainY = None;
            self.DataTrainDim.DataTrainZ = None;
        
        #files = os.path.join("./br_data", "tm-ai_parm01*.csv");
        #files = os.path.join("./br_data", "tm-ai_2*.csv");
            files = os.path.join("./br_data", "merged_parm*.csv");
        # list souboru pro join
            joined_list = glob.glob(files);
        # sort souboru pro join
            joined_list.sort(key=None, reverse=False);
            df = pd.concat([pd.read_csv(csv_file, index_col=0, header=0, encoding="utf-8") for csv_file in joined_list], 
                       axis=0, 
                       ignore_index=False
                    );
        
        # vyber dat dle timestampu
            df["timestamp"] = pd.to_datetime(df["timestamp_x"].str.slice(0, 18));
            df = df[df["timestamp"].between('2022-02-15 08:41:00', '2022-04-09 12:00:00')];
            df_test = df[df["timestamp"].between(timestamp_start, timestamp_stop)];
            

            if shuffling:
                df = df.reset_index(drop=True)
                df = shuffle(df)
                df = df.reset_index(drop=True)
            
            size = len(df.index)
            size_train = math.floor(size * 6 / 12)
            size_valid = math.floor(size * 4 / 12)
            size_test =  math.floor(size * 2 / 12)  

            if self.df_parmx == None or self.df_parmX == None:
                print("");
            else:
                self.DataTrainDim.DataTrainX = self.setDataX(df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b 
                                                        );
            
            if self.df_parmy == None or self.df_parmY == None:
                print("");
            else:
                self.DataTrainDim.DataTrainY = self.setDataY(df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b 
                                                        );
            
            if self.df_parmz == None or self.df_parmZ == None:
                print("");
            else:
                self.DataTrainDim.DataTrainZ = self.setDataZ(df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b 
                                                        );

            return self.DataTrainDim(self.DataTrainDim.DataTrainX, self.DataTrainDim.DataTrainY, self.DataTrainDim.DataTrainZ);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

        
        
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
                    line = line.replace("df_parmx=", "");
                    self.df_parmx = line.split(",");
                    if "null" in line:
                        self.df_parmx = None;
                        
                y = line.startswith("df_parmy=");
                if y:
                    line = line.replace("df_parmy=", "");
                    self.df_parmy = line.split(",")
                    if "null" in line:
                        self.df_parmy = None;

                z = line.startswith("df_parmz=");
                if z:
                    line = line.replace("df_parmz=", "");
                    self.df_parmz = line.split(",")
                    if "null" in line:
                        self.df_parmz = None;
                
                X = line.startswith("df_parmX=");
                if X:
                    line = line.replace("df_parmX=", "");
                    self.df_parmX = line.split(",")
                    if "null" in line:
                        self.df_parmX = None;
            
                Y = line.startswith("df_parmY=");
                if Y:
                    line = line.replace("df_parmY=", "");
                    self.df_parmY = line.split(",")
                    if "null" in line:
                        self.df_parmY = None;

                Z = line.startswith("df_parmZ=");
                if Z:
                    line = line.replace("df_parmZ=", "");
                    self.df_parmZ = line.split(",")
                    if "null" in line:
                        self.df_parmZ = None;
                
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
    # y_dataset predstavuje 60 časových rámců krat prvni prvek casoveho 
    # ramce pole X_dataset
    #
    # funkce vraci: X_dataset - 3D tenzor dat pro uceni site
    #               y_dataset - vektor vstupnich dat (model)
    #               dataset_cols - pocet sloupcu v datove sade. 
    #-----------------------------------------------------------------------
    
    
    def toTensorLSTM(dataset, window=64):
        
        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                        #jen v pripade ze vst. data nejsou definovana
                        
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
        
        return NeuronLayerLSTM.DataSet(X_dataset, y_dataset, dataset_cols);

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
    # y_dataset predstavuje 60 časových rámců krat prvni prvek casoveho 
    # ramce pole X_dataset
    #
    # funkce vraci: X_dataset - 3D tenzor dat pro uceni site
    #               y_dataset - vektor vstupnich dat (model)
    #               dataset_cols - pocet sloupcu v datove sade. 
    #-----------------------------------------------------------------------
    def toTensorGRU(dataset, window=64):
        
        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                        #jen v pripade ze vst. data nejsou definovana
                        
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
        
        return NeuronLayerLSTM.DataSet(X_dataset, y_dataset, dataset_cols);
    #---------------------------------------------------------------------------
    # DataFactory
    #---------------------------------------------------------------------------
    def prepareParmsPredict(self):
        
        i = 0;
        for i in self.df_parmX:
            df_parmX_predict[i] = self.df_parmX[i]+"_predict";
        i = 0;
        for i in self.df_parmY:
            df_parmY_predict[i] = self.df_parmY[i]+"_predict"; 
        i = 0;
        for i in self.df_parmX:
            df_parmZ_predict[i] = self.df_parmZ[i]+"_predict"; 
             

    #---------------------------------------------------------------------------
    # setDataX(self, df,  size_train, size_valid, size_test)
    #---------------------------------------------------------------------------
    def setDataX(self, df, df_test,  size_train, size_valid, size_test, txdt_b=False):
        #OSA X
        try:
            DataTrain_x = self.DataTrainX;
            DataTrain_x.train = pd.DataFrame(df[0 : size_train][self.df_parmX]);
            DataTrain_x.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmX]);
            
            if txdt_b:
                DataTrain_x.test  = df_test;
            else:
                DataTrain_x.test  = pd.DataFrame(df[ size_train + size_valid : size_train + size_valid + size_test ][self.df_parmX]);
                
            DataTrain_x.df_parm_x = self.df_parmx;  # data na ose x, pro rovinu X
            DataTrain_x.df_parm_y = self.df_parmX;  # data na ose y, pro rovinu Y
            DataTrain_x.axis = "OSA_X"
            return(DataTrain_x);
    
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    
    #---------------------------------------------------------------------------
    # setDataY(self, df,  size_train, size_valid, size_test)
    #---------------------------------------------------------------------------
    def setDataY(self, df, df_test, size_train, size_valid, size_test, txdt_b=False):
        #OSA Y
        try:
            DataTrain_y = self.DataTrainY;
            DataTrain_y.train = pd.DataFrame(df[0 : size_train][self.df_parmY]);
            DataTrain_y.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmY])
            DataTrain_y.test  = pd.DataFrame(df[ size_train + size_valid : size_train + size_valid + size_test ][self.df_parmY])
            
            if txdt_b:
                DataTrain_y.test  = df_test;
            else:
                DataTrain_y.test  = pd.DataFrame(df[ size_train + size_valid : size_train + size_valid + size_test ][self.df_parmY])
                
            DataTrain_y.df_parm_x = self.df_parmy;  # data na ose x, pro rovinu Y
            DataTrain_y.df_parm_y = self.df_parmY;  # data na ose y, pro rovinu Y
            DataTrain_y.axis = "OSA_Y"
            return(DataTrain_y);
        
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # setDataZ(self, df,  size_train, size_valid, size_test)
    #---------------------------------------------------------------------------
    def setDataZ(self, df, df_test, size_train, size_valid, size_test, txdt_b=False):
        #OSA Z
        try:
            DataTrain_z = self.DataTrainZ;
            DataTrain_z.train = pd.DataFrame(df[0 : size_train][self.df_parmZ]);
            DataTrain_z.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmZ])
            
            if txdt_b:
                DataTrain_z.test  = df_test;
            else:    
                DataTrain_z.test  = pd.DataFrame(df[ size_train + size_valid : size_train + size_valid + size_test ][self.df_parmZ])
            
            DataTrain_z.df_parm_x = self.df_parmz;  # data na ose x, pro rovinu Z
            DataTrain_z.df_parm_y = self.df_parmZ;  # data na ose y, pro rovinu Z
            DataTrain_z.axis = "OSA_Z"
            return(DataTrain_z);
        
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
    
    def getDf_parmy(self):
        return self.df_parmy;
    
    def getDf_parmz(self):
        return self.df_parmz;
        
    def getDf_parmX(self):
        return self.df_parmX;
    
    def getDf_parmY(self):
        return self.df_parmY;
    
    def getDf_parmZ(self):
        return self.df_parmZ;


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

    def __init__(self, path_to_result, model, type):
        self.path_to_result = path_to_result; 
        self.model = model; 
        self.type = type;
          
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
        
        list_test   = df_result['out'];
        list_result = df_result['out_predict_y'];
        # report model error
        mse = mean_squared_error(list_test, list_result);
        mae = max_error(list_test, list_result);
        
        return(df_result, mse, mae);
        
        
    #---------------------------------------------------------------------------
    # printGrafCompare
    #---------------------------------------------------------------------------
    def printGrafCompare(self, DataResult, DataTrain, substract=True):
        
        axis = "";
        col_names_y="";
        col_names_x="";
        str_inp1 = ""; 
        col =  0;
        
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
                
                if ("MACH" in col_names_y[col].upper() or "DEV" in col_names_y[col].upper()):
                    df_graph = pd.DataFrame(DataResult.y_test[ : ,col]);
                    df_graph['out'] = DataResult.y_test[ : , col];
                    df_graph['out_predict'] = DataResult.y_result[ : , col];
                    
                    df_graph, mse, mae  = self.groupAvg(df_graph);
                    
                    max_axe = df_graph['out'].abs().max() * 2;
                    max_mae = df_graph['diff'].abs().max() * 3;
                
                    mpl.rcParams.update({'font.size': 6})
                    ax = df_graph.plot(y=['out', 'out_predict_y'], kind='line', label=['realna data', 'predikce']);
                    ax.legend(title = self.model+' ' +axis+ 
                        '\nPath:\n' + self.path_to_result + 
                        '\nVst.:\n' + str_inp1 + 
                        '\nVyst:\n'     + col_names_y[col] +  
                        '\n\nMSE:'      + str(format(mse, '.9f')) + 
                        '\nMAE:'        + str(format(mae, '.9f')),
                        title_fontsize = 6,
                        fontsize = 6
                    );
                    ax.set_xlabel("vzorky")
                    ax.set_ylabel(col_names_y[col])
                    ax.set_ylim(-max_axe,+max_axe);
                        
                    ax.get_figure().savefig(self.path_to_result+'/compare_'+col_names_y[col]+'-'+axis+'.pdf', format='pdf');
                    plt.close(ax.get_figure())

                
                if substract and ("MACH" in col_names_y[col].upper() or "DEV" in col_names_y[col].upper()):
                    #df_graph = shuffle(df_graph);
                    df_graph = df_graph.reset_index(drop=True)
                    
                    mpl.rcParams.update({'font.size': 6})
                    ax = df_graph.plot(y=['diff'], kind='line', label=['realna - predikovana data']);
                    ax.legend(title = self.model+' ' +axis+ 
                        '\nPath:\n' + self.path_to_result + 
                        '\nVst.:\n' + str_inp1 + 
                        '\nVyst:\n'     + col_names_y[col] +  
                        '\n\nMSE:'      + str(format(mse, '.9f')) + 
                        '\nMAE:'        + str(format(mae, '.9f')),
                        title_fontsize = 6,
                        fontsize = 6
                        );
                    ax.set_xlabel("vzorky")
                    ax.set_ylabel(col_names_y[col])
                    ax.set_ylim(-max_mae,+max_mae);
                        
                    ax.get_figure().savefig(self.path_to_result+'/substract_'+col_names_y[col]+'-'+axis+'.pdf', format='pdf');
                    plt.close(ax.get_figure())
                    
                self.saveToCSV(self.model, axis, col_names_y[col], str(format(mae, '.9f')), str(format(mse, '.9f')))
                col = col + 1;
                
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
            
    #---------------------------------------------------------------------------
    # printGraf - kolekce dat         
    #---------------------------------------------------------------------------
    def saveToCSV(self, model, axis, plcname, mae, mse):
        
        header = ["datetime", "model", "axis", "plcname", "mae", "mse", "type"];
        filename = "./result/result_csv.csv"
        
        data = {'datetime'   : [self.path_to_result],
                   'model'   : [model],
                   'axis'    : [axis],
                   'plcname' : [plcname],
                   'mae'     : [mae],
                   'mse'     : [mse],
                   'type'    : [self.type]
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
# Neuronova Vrstava DENSE
#---------------------------------------------------------------------------
class NeuronLayerDENSE():
    #definice datoveho ramce
    
    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        
        self.df = pd.DataFrame();
        self.df_out = pd.DataFrame();
        self.graph = None;
        self.data  = None;

        
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
            x_train = MinMaxScaler(feature_range=(0, 1))
            x_train = x_train.fit_transform(x_train_data)
            y_train = MinMaxScaler(feature_range=(0, 1))
            y_train = y_train.fit_transform(y_train_data)
        
        # normalizace dat k uceni a vstupnich validacnich dat 
            x_valid = MinMaxScaler(feature_range=(0, 1))
            x_valid = x_valid.fit_transform(x_valid_data)
            y_valid = MinMaxScaler(feature_range=(0, 1))
            y_valid = y_valid.fit_transform(y_valid_data)

            
        # neuronova sit
            model = Sequential();
            model.add(tf.keras.Input(shape=(inp_size,)));
            model.add(layers.Dense(1024, activation='sigmoid', kernel_initializer='he_normal'))
            model.add(layers.Dense(1024, activation='sigmoid', kernel_initializer='he_normal'))
            model.add(layers.Dense(1024, activation='sigmoid', kernel_initializer='he_normal'))
            model.add(layers.Dense(out_size))
            
        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])
            
        # natrenuj model na vstupni dataset
            history = model.fit(x_train, 
                            y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch, 
                            verbose=2, 
                            validation_data=(x_valid, y_valid)
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
    def neuralNetworkDENSEpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
        # normalizace vstupnich a vystupnich testovacich dat 
            x_test_scale  = MinMaxScaler(feature_range=(0, 1));
            x_test        = x_test_scale.fit_transform(x_test);
            y_test_scale  = MinMaxScaler(feature_range=(0, 1));
            y_test        = y_test_scale.fit_transform(y_test);
        
        # predict
            y_result = model.predict(x_test);
        
            x_test   = x_test_scale.inverse_transform(x_test);
            y_test   = y_test_scale.inverse_transform(y_test)

        #y_result_scale = MinMaxScaler();
            y_result = y_test_scale.inverse_transform(y_result);
            
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
                print("Start osy X vcetne treninku, model bude zapsan");
                logging.info("Start osy X vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrainX);
            else:    
                print("Start osy X bez treninku - model bude nacten");
                logging.info("Start osy X bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainX.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkDENSEpredict(model_x, data.DataTrainDim.DataTrainX);
            graph.printGrafCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrainX);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa X [s] %s",  str(stopTime - startTime));

            return();

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());
            
    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_y 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec_y(self, data, graph):
        try:
            
            startTime = datetime.now();
            model_y = '';
            if self.typ == 'train':
                print("Start osy Y vcetne treninku model bude zapsan");
                logging.info("Start osy Y vcetne treninku, model bude zapsan");
                model_y = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrainY);
            else:    
                print("Start osy Y bez treninku - model bude nacten");
                logging.info("Start osy Y bez treninku - model bude nacten");
                model_y = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainY.axis);
                
            data.DataResultDim.DataResultY = self.neuralNetworkDENSEpredict(model_y, data.DataTrainDim.DataTrainY);
            graph.printGrafCompare(data.DataResultDim.DataResultY, data.DataTrainDim.DataTrainY);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Y [s] %s",  str(stopTime - startTime));
            
            return();

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_z 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec_z(self, data, graph):
        try:
            startTime = datetime.now()
            
            model_z = ''
            if self.typ == 'train':
                print("Start osy Z vcetne treninku, model bude zapsan");
                logging.info("Start osy Z vcetne treninku, model bude zapsan");
                model_z = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrainZ);
            else:    
                print("Start osy Z bez treninku - model bude nacten");
                logging.info("Start osy Z bez treninku - model bude nacten");
                model_z = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainZ.axis);
                
            data.DataResultDim.DataResultZ = self.neuralNetworkDENSEpredict(model_z, data.DataTrainDim.DataTrainZ);
            graph.printGrafCompare(data.DataResultDim.DataResultZ, data.DataTrainDim.DataTrainZ);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Z [s] %s",  str(stopTime - startTime));
            
        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    

    #------------------------------------------------------------------------
    # neuralNetworkDENSEexec
    #------------------------------------------------------------------------
    def neuralNetworkDENSEexec(self):

        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result)
            self.graph = GraphResult(path_to_result=self.path_to_result, model=self.model, type=self.typ)
        
            shuffling = False;
            if self.typ == 'train':
                shuffling = True;
        
            self.data.Data = self.data.getData(shuffling=shuffling);

    # osa X    
            if self.data.DataTrainDim.DataTrainX  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkDENSEexec_x(data=self.data, graph=self.graph);
            
    # osa Y    
            if self.data.DataTrainDim.DataTrainY  == None:
                print("Osa Y je disable...");
                logging.info("Osa Y je disable...");
            else:
                self.neuralNetworkDENSEexec_y(data=self.data, graph=self.graph);
        
    # osa Z    
            if self.data.DataTrainDim.DataTrainZ  == None:
                print("Osa Z je disable...");
                logging.info("Osa Z je disable...");
            else:
                self.neuralNetworkDENSEexec_z(data=self.data, graph=self.graph);

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

    def __init__(self, path_to_result, typ, model, epochs, batch):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        
        self.df = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.graph = None;
        self.data  = None;
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava LSTM 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMtrain(self, DataTrain):
        window_X = 64;
        window_Y =  1;
        
        try:
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0]);
            out_size = len(y_train_data[0]);

            # normalizace dat k uceni a vstupnich treninkovych dat 
            x_train = MinMaxScaler(feature_range=(0, 1));
            x_train = x_train.fit_transform(x_train_data);
            y_train = MinMaxScaler(feature_range=(0, 1));
            y_train = y_train.fit_transform(y_train_data);
        
            # normalizace dat k uceni a vstupnich validacnich dat 
            x_valid = MinMaxScaler(feature_range=(0, 1));
            x_valid = x_valid.fit_transform(x_valid_data);
            y_valid = MinMaxScaler(feature_range=(0, 1));
            y_valid = y_valid.fit_transform(y_valid_data);
        
        #data pro trenink -3D tenzor
            X_train =  DataFactory.toTensorLSTM(x_train, window=window_X);
        #vstupni data train 
            Y_train = DataFactory.toTensorLSTM(y_train, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = DataFactory.toTensorLSTM(x_valid, window=window_X);
        #vystupni data pro trenink -3D tenzor
            Y_valid = DataFactory.toTensorLSTM(y_valid, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
        # neuronova sit
            model = Sequential();
            model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            model.add(LSTM(units = 1024, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(LSTM(units = 1024, return_sequences=True));
            model.add(Dropout(0.25));
            model.add(LSTM(units = 1024, return_sequences=True));
            model.add(Dropout(0.25));
            model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj model na vstupni dataset
            history = model.fit(X_train.X_dataset, 
                                Y_train.X_dataset, 
                                epochs=self.epochs, 
                                batch_size=self.batch, 
                                verbose=2, 
                                validation_data=(X_valid.X_dataset, Y_valid.X_dataset),
                                shuffle=False
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
    # Neuronova Vrstava DENSE predict 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMpredict(self, model, DataTrain):

        try:
        
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);

            # normalizace vstupnich a vystupnich testovacich dat 
            x_test_scaler = MinMaxScaler(feature_range=(0, 1));
            x_test        = x_test_scaler.fit_transform(x_test);
            y_test_scaler = MinMaxScaler(feature_range=(0, 1));
            y_test        = y_test_scaler.fit_transform(y_test);
            
            x_object      = DataFactory.toTensorLSTM(x_test, window=64);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
        
            y_test        = y_test[ : len(y_result)]
            x_test        = x_test_scaler.inverse_transform(x_test);
            y_test        = y_test_scaler.inverse_transform(y_test);
            y_result      = y_test_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()
        
            return DataFactory.DataResult(x_test, y_test, y_result, axis)
        
        except Exception as ex:
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
                print("Start osy X vcetne treninku, model bude zapsan");
                logging.info("Start osy X vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkLSTMtrain(data.DataTrainDim.DataTrainX);
            else:    
                print("Start osy X bez treninku - model bude nacten");
                logging.info("Start osy X bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainX.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkLSTMpredict(model_x, data.DataTrainDim.DataTrainX);
            graph.printGrafCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrainX);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa X [s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkLSTMexec_y 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMexec_y(self, data, graph):
        try:
            
            startTime = datetime.now();
            model_y = '';
            if self.typ == 'train':
                print("Start osy Y vcetne treninku model bude zapsan");
                logging.info("Start osy Y vcetne treninku, model bude zapsan");
                model_y = self.neuralNetworkLSTMtrain(data.DataTrainDim.DataTrainY);
            else:    
                print("Start osy Y bez treninku - model bude nacten");
                logging.info("Start osy Y bez treninku - model bude nacten");
                model_y = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainY.axis);
                
            data.DataResultDim.DataResultY = self.neuralNetworkLSTMpredict(model_y, data.DataTrainDim.DataTrainY);
            graph.printGrafCompare(data.DataResultDim.DataResultY, data.DataTrainDim.DataTrainY);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Y [s] %s",  str(stopTime - startTime));
            return(0);

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());


    #---------------------------------------------------------------------------
    # neuralNetworkLSTMexec_z 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMexec_z(self, data, graph):
        try:
            startTime = datetime.now()
            
            model_z = ''
            if self.typ == 'train':
                print("Start osy Z vcetne treninku, model bude zapsan");
                logging.info("Start osy Z vcetne treninku, model bude zapsan");
                model_z = self.neuralNetworkLSTMtrain(data.DataTrainDim.DataTrainZ);
            else:    
                print("Start osy Z bez treninku - model bude nacten");
                logging.info("Start osy Z bez treninku - model bude nacten");
                model_z = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainZ.axis);
                
            data.DataResultDim.DataResultZ = self.neuralNetworkLSTMpredict(model_z, data.DataTrainDim.DataTrainZ);
            graph.printGrafCompare(data.DataResultDim.DataResultZ, data.DataTrainDim.DataTrainZ);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Z [s] %s",  str(stopTime - startTime));
            return(0);
            
        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());


    #------------------------------------------------------------------------
    # neuralNetworkLSTMexec
    #------------------------------------------------------------------------
    def neuralNetworkLSTMexec(self):
        
        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result)
            self.graph = GraphResult(path_to_result=self.path_to_result, model=self.model, type=self.typ)
        
            shuffling = False;
            if self.typ == 'train':
                shuffling = True;
        
            self.data.Data = self.data.getData(shuffling=shuffling);

    # osa X    
            if self.data.DataTrainDim.DataTrainX  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkLSTMexec_x(data=self.data, graph=self.graph);
            
    # osa Y    
            if self.data.DataTrainDim.DataTrainY  == None:
                print("Osa Y je disable...");
                logging.info("Osa Y je disable...");
            else:
                self.neuralNetworkLSTMexec_y(data=self.data, graph=self.graph);
        
    # osa Z    
            if self.data.DataTrainDim.DataTrainZ  == None:
                print("Osa Z je disable...");
                logging.info("Osa Z je disable...");
            else:
                self.neuralNetworkLSTMexec_z(data=self.data, graph=self.graph);
        
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

    def __init__(self, path_to_result, typ, model, epochs, batch):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        
        self.df = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.graph = None;
        self.data  = None;
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava GRU 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUtrain(self, DataTrain):
        window_X = 64;
        window_Y =  1;
        
        try:
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0]);
            out_size = len(y_train_data[0]);

            # normalizace dat k uceni a vstupnich treninkovych dat 
            x_train = MinMaxScaler(feature_range=(0, 1));
            x_train = x_train.fit_transform(x_train_data);
            y_train = MinMaxScaler(feature_range=(0, 1));
            y_train = y_train.fit_transform(y_train_data);
        
            # normalizace dat k uceni a vstupnich validacnich dat 
            x_valid = MinMaxScaler(feature_range=(0, 1));
            x_valid = x_valid.fit_transform(x_valid_data);
            y_valid = MinMaxScaler(feature_range=(0, 1));
            y_valid = y_valid.fit_transform(y_valid_data);
        
        #data pro trenink -3D tenzor
            X_train =  DataFactory.toTensorGRU(x_train, window=window_X);
        #vstupni data train 
            Y_train = DataFactory.toTensorGRU(y_train, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = DataFactory.toTensorGRU(x_valid, window=window_X);
        #vystupni data pro trenink -3D tenzor
            Y_valid = DataFactory.toTensorGRU(y_valid, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
        # neuronova sit
            model = Sequential();
            model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            model.add(GRU(units = 1024, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(GRU(units = 1024, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(GRU(units = 1024, return_sequences=True));
            model.add(Dropout(0.2));
            model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj model na vstupni dataset
            history = model.fit(X_train.X_dataset, 
                                Y_train.X_dataset, 
                                epochs=self.epochs, 
                                batch_size=self.batch, 
                                verbose=2, 
                                validation_data=(X_valid.X_dataset, Y_valid.X_dataset),
                                shuffle=False
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
    # Neuronova Vrstava DENSE predict 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUpredict(self, model, DataTrain):

        try:
        
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);

            # normalizace vstupnich a vystupnich testovacich dat 
            x_test_scaler = MinMaxScaler(feature_range=(0, 1));
            x_test        = x_test_scaler.fit_transform(x_test);
            y_test_scaler = MinMaxScaler(feature_range=(0, 1));
            y_test        = y_test_scaler.fit_transform(y_test);
            
            x_object      = DataFactory.toTensorGRU(x_test, window=64);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
        
            y_test        = y_test[ : len(y_result)]
            x_test        = x_test_scaler.inverse_transform(x_test);
            y_test        = y_test_scaler.inverse_transform(y_test);
            y_result      = y_test_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()
        
            return DataFactory.DataResult(x_test, y_test, y_result, axis)
        
        except Exception as ex:
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
                print("Start osy X vcetne treninku, model bude zapsan");
                logging.info("Start osy X vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkGRUtrain(data.DataTrainDim.DataTrainX);
            else:    
                print("Start osy X bez treninku - model bude nacten");
                logging.info("Start osy X bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainX.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkGRUpredict(model_x, data.DataTrainDim.DataTrainX);
            graph.printGrafCompare(data.DataResultDim.DataResultX, data.DataTrainDim.DataTrainX);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa X [s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkGRUexec_y 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUexec_y(self, data, graph):
        try:
            
            startTime = datetime.now();
            model_y = '';
            if self.typ == 'train':
                print("Start osy Y vcetne treninku model bude zapsan");
                logging.info("Start osy Y vcetne treninku, model bude zapsan");
                model_y = self.neuralNetworkGRUtrain(data.DataTrainDim.DataTrainY);
            else:    
                print("Start osy Y bez treninku - model bude nacten");
                logging.info("Start osy Y bez treninku - model bude nacten");
                model_y = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainY.axis);
                
            data.DataResultDim.DataResultY = self.neuralNetworkGRUpredict(model_y, data.DataTrainDim.DataTrainY);
            graph.printGrafCompare(data.DataResultDim.DataResultY, data.DataTrainDim.DataTrainY);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Y [s] %s",  str(stopTime - startTime));
            return(0);

        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            
        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());


    #---------------------------------------------------------------------------
    # neuralNetworkGRUexec_z 
    #---------------------------------------------------------------------------
    def neuralNetworkGRUexec_z(self, data, graph):
        try:
            startTime = datetime.now()
            
            model_z = ''
            if self.typ == 'train':
                print("Start osy Z vcetne treninku, model bude zapsan");
                logging.info("Start osy Z vcetne treninku, model bude zapsan");
                model_z = self.neuralNetworkGRUtrain(data.DataTrainDim.DataTrainZ);
            else:    
                print("Start osy Z bez treninku - model bude nacten");
                logging.info("Start osy Z bez treninku - model bude nacten");
                model_z = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrainZ.axis);
                
            data.DataResultDim.DataResultZ = self.neuralNetworkGRUpredict(model_z, data.DataTrainDim.DataTrainZ);
            graph.printGrafCompare(data.DataResultDim.DataResultZ, data.DataTrainDim.DataTrainZ);
            stopTime = datetime.now();
            logging.info("cas vypoctu - osa Z [s] %s",  str(stopTime - startTime));
            return(0);
            
        except FileNotFoundError as e:
            print(f"Nenalezen model site pro osu Y zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            logging.error(f"Nenalezen model site pro osu Y , zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());


    #------------------------------------------------------------------------
    # neuralNetworkGRUexec
    #------------------------------------------------------------------------
    def neuralNetworkGRUexec(self):
        
        try:
            print("Pocet GPU jader: ", len(tf.config.experimental.list_physical_devices('GPU')))
            self.data = DataFactory(path_to_result=self.path_to_result)
            self.graph = GraphResult(path_to_result=self.path_to_result, model=self.model, type=self.typ)
        
            shuffling = False;
            if self.typ == 'train':
                shuffling = True;
        
            self.data.Data = self.data.getData(shuffling=shuffling);

    # osa X    
            if self.data.DataTrainDim.DataTrainX  == None:
                print("Osa X je disable...");
                logging.info("Osa X je disable...");
            else:
                self.neuralNetworkGRUexec_x(data=self.data, graph=self.graph);
            
    # osa Y    
            if self.data.DataTrainDim.DataTrainY  == None:
                print("Osa Y je disable...");
                logging.info("Osa Y je disable...");
            else:
                self.neuralNetworkGRUexec_y(data=self.data, graph=self.graph);
        
    # osa Z    
            if self.data.DataTrainDim.DataTrainZ  == None:
                print("Osa Z je disable...");
                logging.info("Osa Z je disable...");
            else:
                self.neuralNetworkGRUexec_z(data=self.data, graph=self.graph);
        
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="GRU", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            logging.error(traceback.print_exc());











#------------------------------------------------------------------------
# saveModelToArchiv - zaloha modelu, spusteno jen pri parametru train
#------------------------------------------------------------------------
def saveModelToArchiv(model, dest_path, data):

    axes = np.array(["OSA_X", "OSA_Y", "OSA_Z"]);
    src_dir  = "./models/model_"+model+"_";
    dest_dir = "/models/model_"+model+"_";
    try:
        if data.DataTrainDim.DataTrainX  == None:
            print()
        else:    
            src_dir_  = src_dir + axes[0]
            dest_dir_ = dest_path + dest_dir + axes[0]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
            
        if data.DataTrainDim.DataTrainY  == None:
            print()
        else:    
            src_dir_  = src_dir + axes[1]
            dest_dir_ = dest_path + dest_dir + axes[1]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
         
        if data.DataTrainDim.DataTrainZ  == None:
            print()
        else:    
            src_dir_  = src_dir + axes[2]
            dest_dir_ = dest_path + dest_dir + axes[2]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
         
        return(0);    
   
    except Exception as ex:
        traceback.print_exc();
        logging.error(traceback.print_exc());
 

#------------------------------------------------------------------------
# setEnv
#------------------------------------------------------------------------
def setEnv(path, model):

        progname = os.path.basename(__file__);
        current_date =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S");
        path1 = path+model
        path2 = path+model+"/"+current_date
        
        try: 
            os.mkdir("./log");
        except OSError as error: 
            print(); 

        try: 
            os.mkdir("./result")
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
            print("Chyba pri kopii zdrojoveho kodu.", error)
        except:
            print("Chyba pri kopii zdrojoveho kodu.")
        
        
        logging.basicConfig(filename='./log/'+progname+'.log',
            filemode='a',level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    
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
def help ():
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
    print ("        --model           model neuronove site 'DENSE', 'LSTM', 'GRU'")
    print ("                                 DENSE - zakladni model site - nejmene narocny na system")
    print ("                                 LSTM - Narocny model rekurentni site s feedback vazbami")
    print ("                                 GRU  - Narocny model rekurentni hradlove site")
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
    print (" ");
    print ("priklad: ./ai-neuro.py -t train, -m DENSE, -e 64 -b 128");
    print ("nebo:    ./ai-neuro.py --typ train, --model DENSE, --epochs 64 --batch 128");
    print('parametr --epochs musi byt cislo typu int <1, 256>')
    print ("POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM nebo GRU!!!");
    print ("       pricemz 'train' je povinny pri prvnim behu site, pri nemz se zapise natrenovany model.");
    print ("       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.");
    print ("       spusti normalni beh site z jiz natrenovaneho modelu.");
    print ("       Takze: budte trpelivi...");
    print (" ");
    print ("Pokud bude program spusten bez parametru, implicitne se dosadi -t train");
    print ("                                                               -m DENSE");
    print ("                                                               -e 64");
    print ("                                                               -b 128");
    print (" ");
    print ("Vstupni parametry: ");
    print ("  pokud neexistuje v rootu aplikace soubor ai-parms.txt, pak jsou parametry implicitne");
    print ("  prirazeny z promennych definovanych v programu:");
    print ("Jedna se o tyto promenne: ");
    print (" ");
    print ("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print ("  df_parmx = ['machinedata_m0412','teplota_pr01', 'x_temperature']");
    print ("  df_parmy = ['machinedata_m0413','teplota_pr02', 'y_temperature']");
    print ("  df_parmz = ['machinedata_m0414','teplota_pr03', 'z_temperature']");
    print (" ");
    print ("  #Tenzor predlozeny k uceni site");
    print ("  df_parmX = ['machinedata_m0112','machinedata_m0212','machinedata_m0312','machinedata_m0412','teplota_pr01', 'x_temperature'];");
    print ("  df_parmY = ['machinedata_m0113','machinedata_m0213','machinedata_m0313','machinedata_m0413','teplota_pr02', 'y_temperature'];");
    print ("  df_parmZ = ['machinedata_m0114','machinedata_m0214','machinedata_m0314','machinedata_m0414','teplota_pr03', 'z_temperature'];");
    print (" ");
    print ("Pokud pozadujete zmenu parametu j emozno primo v programu poeditovat tyto promenne ");
    print (" ");
    print ("a nebo vyrobit soubor ai-parms.txt s touto syntaxi ");
    print ("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print ("  df_parmx = machinedata_m0412,teplota_pr01,x_temperature'");
    print ("  df_parmy = machinedata_m0413,teplota_pr02,y_temperature'");
    print ("  df_parmz = machinedata_m0414,teplota_pr03,z_temperature'");
    print (" ");
    print ("  #Tenzor predlozeny k uceni site");
    print ("  df_parmX = machinedata_m0412,teplota_pr01, x_temperature");
    print ("  df_parmY = machinedata_m0413,teplota_pr02, y_temperature");
    print ("  df_parmZ = machinedata_m0414,teplota_pr03, z_temperature");
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
    return();
    
   
#------------------------------------------------------------------------
# main
#------------------------------------------------------------------------

def main(argv):
    
    global path_to_result
    path_to_result = "./result";

    try:
        parm0 = sys.argv[0];
        
        try:
            opts, args = getopt.getopt(sys.argv[1:],"ht:m:e:b:h:x",["typ=", "model=", "epochs=", "batch=","help="])
        except getopt.GetoptError:
            print("Chyba pri parsovani parametru:");
            help()
            
        for opt, arg in opts:
            if opt in ("-t", "--typ"):
                parm1 = arg;
            elif opt in ("-m", "--model"):
                parm2 = arg.upper();
            elif opt in ("-e", "--epochs"):
                try:
                    r = range(1, 256);
                    parm3 = int(arg);
                    if parm3 not in r:
                        print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <1, 256>");
                        help()
                        sys.exit(1)    
                        
                except:
                    print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <1, 256>");
                    help()
                    sys.exit(1)    
            elif opt in ("-b", "--batch"):
                try:
                    r = range(32, 2048);
                    parm4 = int(arg);
                    if parm4 not in r:
                        print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <32, 2048>");
                        help()
                        sys.exit(1)    
                except:    
                    print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <32, 2048>");
                    help()
                    sys.exit(1)    
                    
            elif opt in ["-h","-help"]:
                help()
                sys.exit(0)

        
        if len(sys.argv) < 9:
            help()
            parm1 = 'train'
            parm2 = 'DENSE'
            parm3 = 32
            parm4 = 128
            #sys.exit(1);
        
        path_to_result = setEnv(path=path_to_result, model=parm2);
        
        startTime = datetime.now()
        logging.info("start...");
        logging.info("Verze TensorFlow :" + tf.__version__)
        print("Verze TensorFlow :", tf.__version__)
        
        if parm2 == 'LSTM':
            neural = NeuronLayerLSTM(path_to_result=path_to_result, typ=parm1, model=parm2, epochs=parm3, batch=parm4);
            neural.neuralNetworkLSTMexec();
            
        elif parm2 == 'DENSE':
            neural = NeuronLayerDENSE(path_to_result=path_to_result, typ=parm1, model=parm2, epochs=parm3, batch=parm4);
            neural.neuralNetworkDENSEexec();
            
        elif parm2 == 'GRU':
            neural = NeuronLayerGRU(path_to_result=path_to_result, typ=parm1, model=parm2, epochs=parm3, batch=parm4);
            neural.neuralNetworkGRUexec();
            
    
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        logging.error(traceback.print_exc());
        help()
        
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
    

    


