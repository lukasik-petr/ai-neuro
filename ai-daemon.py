#!/usr/bin/python3

#------------------------------------------------------------------------------
# ai-daemon
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
# import vseho co souvisi s demonem...
import sys, os, getopt; 
import traceback;
import time; 
import atexit; 
import signal; 
import grp;
import daemon; 
import lockfile;

import logging;
from signal import SIGTERM
import glob as glob;
import pandas as pd;
import seaborn as sns;
import tensorflow as tf;
import math;
import numpy as np;
import shutil;
from matplotlib import cm;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import platform;
import pandas.api.types as ptypes
import pickle;

from lockfile.pidlockfile import PIDLockFile;
from os.path import exists;

from dateutil import parser
from sklearn.preprocessing import MinMaxScaler;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import max_error;
from sklearn.utils import shuffle;
from sklearn.utils import assert_all_finite;
from numpy import asarray;

from dataclasses import dataclass;
from datetime import datetime, timedelta, timezone;
from tabulate import tabulate;
from pathlib import Path;
from daemon import pidfile;
from pandas.core.frame import DataFrame

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

from _cffi_backend import string

from scipy.signal import butter, lfilter, freqz

from opcua import ua
from opcua import *
from opcua.common.ua_utils import data_type_to_variant_type

from subprocess import call;
from plistlib import InvalidFileException
from tensorflow.python.eager.function import np_arrays
from pandas.errors import EmptyDataError
from keras.saving.utils_v1.mode_keys import is_train



#---------------------------------------------------------------------------
# nastaveni globalnich parametru logu pro demona
#---------------------------------------------------------------------------
logger = None;
log_handler = None;
df_debug_count=0;
df_debug_header=[];

#---------------------------------------------------------------------------
# OPCAgent
#---------------------------------------------------------------------------
class OPCAgent():

    @dataclass
    class PLCData:
    # osy XYZ
        CompX:     object              #Aktualni hodnota kompenzace v ose X
        CompY:     object              #Aktualni hodnota kompenzace v ose Y
        CompZ:     object              #Aktualni hodnota kompenzace v ose Z
    # osy rotace AC
        CompA:     object              #Aktualni hodnota kompenzace v ose A
        CompC:     object              #Aktualni hodnota kompenzace v ose C
        
    # osy XYZ
        setCompX:     object           #Predikovana hodnota kompenzace v ose X
        setCompY:     object           #Predikovana hodnota kompenzace v ose Y
        setCompZ:     object           #Predikovana hodnota kompenzace v ose Z
    # osy rotace AC
        setCompA:     object           #Predikovana hodnota kompenzace v ose A
        setCompC:     object           #Predikovana hodnota kompenzace v ose C
        
    
    # konstrukter    
    def __init__(self, logger, batch):
        self.prefix       = "opc.tcp://";
        self.host1        = "opc998.os.zps"; # BR-PLC
        self.port1        = "4840";
        self.host2        = "opc999.os.zps";# HEIDENHANIN-PLC
        self.port2        = "48010";
        self.is_ping      = False;
        self.logger       = logger;
        self.batch        = batch;
        self.plc_interval = 500/1000; # 500 [ms] sekunda(y)
        
        self.df_debug     = pd.DataFrame();
        self.uri1         = self.prefix+self.host1+":"+self.port1;
        self.uri2         = self.prefix+self.host2+":"+self.port2;
        
    #---------------------------------------------------------------------------
    # myFloatFormat         
    #---------------------------------------------------------------------------
    def myFloatFormat(self, x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

    #---------------------------------------------------------------------------
    # myIntFormat         
    #---------------------------------------------------------------------------
    def myIntFormat(self, x):
        return ('%.f' % x).rstrip('.');


    #---------------------------------------------------------------------------
    # isPing ????         
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.is_ping;
    #---------------------------------------------------------------------------
    # ping         
    #---------------------------------------------------------------------------
    def ping_(self, host):
        
        parameter = '-n' if platform.system().lower()=='windows' else '-c';                                                 
                                                                                                                            
        command = ['ping', parameter, '1', host];                                                                           
        response = call(command);                                                                                           
                                                                                                                            
        if response == 0:    
            self.is_ping = True;                                                                                               
            return self.is_ping;                                                                                                    
        else:                                                                                                               
            self.is_ping = False;                                                                                               
            return self.is_ping;
        
                                                                                                            
        
    #---------------------------------------------------------------------------
    # opcCollectorBR_PLC - opc server BR
    #---------------------------------------------------------------------------
    def opcCollectorBR_PLC(self):
        
        global plc_isRunning;
        
        plc_isRunning = True;
        # tabulka nodu v br plc         
        plc_br_table        = np.array([["temp_ch01",     "ns=6;s=::AsGlobalPV:teplota_ch01"],
                                        ["temp_lo01",     "ns=6;s=::AsGlobalPV:teplota_lo01"],
                                        ["temp_lo03",     "ns=6;s=::AsGlobalPV:teplota_lo03"],
                                        ["temp_po01",     "ns=6;s=::AsGlobalPV:teplota_po01"],
                                        ["temp_pr01",     "ns=6;s=::AsGlobalPV:teplota_pr01"],
                                        ["temp_pr02",     "ns=6;s=::AsGlobalPV:teplota_pr02"],
                                        ["temp_pr03",     "ns=6;s=::AsGlobalPV:teplota_pr03"],
                                        ["temp_sl01",     "ns=6;s=::AsGlobalPV:teplota_sl01"],
                                        ["temp_sl02",     "ns=6;s=::AsGlobalPV:teplota_sl02"],
                                        ["temp_sl03",     "ns=6;s=::AsGlobalPV:teplota_sl03"],
                                        ["temp_sl04",     "ns=6;s=::AsGlobalPV:teplota_sl04"],
                                        ["temp_st01",     "ns=6;s=::AsGlobalPV:teplota_st01"],
                                        ["temp_st02",     "ns=6;s=::AsGlobalPV:teplota_st02"],
                                        ["temp_st03",     "ns=6;s=::AsGlobalPV:teplota_st03"],
                                        ["temp_st04",     "ns=6;s=::AsGlobalPV:teplota_st04"],
                                        ["temp_st05",     "ns=6;s=::AsGlobalPV:teplota_st05"],
                                        ["temp_st06",     "ns=6;s=::AsGlobalPV:teplota_st06"],
                                        ["temp_st07",     "ns=6;s=::AsGlobalPV:teplota_st07"],
                                        ["temp_st08",     "ns=6;s=::AsGlobalPV:teplota_st08"],
                                        ["temp_vr01",     "ns=6;s=::AsGlobalPV:teplota_vr01"],
                                        ["temp_vr02",     "ns=6;s=::AsGlobalPV:teplota_vr02"],
                                        ["temp_vr03",     "ns=6;s=::AsGlobalPV:teplota_vr03"],
                                        ["temp_vr04",     "ns=6;s=::AsGlobalPV:teplota_vr04"],
                                        ["temp_vr05",     "ns=6;s=::AsGlobalPV:teplota_vr05"],
                                        ["temp_vr06",     "ns=6;s=::AsGlobalPV:teplota_vr06"],
                                        ["temp_vr07",     "ns=6;s=::AsGlobalPV:teplota_vr07"],
                                        ["temp_vz02",     "ns=6;s=::AsGlobalPV:teplota_vz02"],
                                        ["temp_vz03",     "ns=6;s=::AsGlobalPV:teplota_vz03"],
                                        ["light_ambient", "ns=6;s=::AsGlobalPV:vstup_osvit"],
                                        ["temp_ambient",  "ns=6;s=::AsGlobalPV:vstup_teplota"],
                                        ["humid_ambient", "ns=6;s=::AsGlobalPV:vstup_vlhkost"]]);

        #if not self.ping_(self.host1):
        #    plc_isRunning = False;
        #    return(plc_br_table, plc_isRunning);
   
        client = Client(self.uri1)
        try:        
            client.connect();
            #sys.stderr.write("Client" + uri+ "Connected\n");
            #self.logger.info("Client" + uri+ "Connected")
            plc_br_table = np.c_[plc_br_table, np.zeros(len(plc_br_table))];
            
            for i in range(len(plc_br_table)):
                node = client.get_node(str(plc_br_table[i, 1]));
                typ  = type(node.get_value());
                val = float(self.myFloatFormat(node.get_value())) if typ is float else node.get_value();
                #val = node.get_value() if typ is float else node.get_value();
                plc_br_table[i, 2] = val;
            
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc())+"\n");
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect();
            return(plc_br_table, plc_isRunning);
 


    #---------------------------------------------------------------------------
    # opcCollectorHH_PLC - opc server PLC HEIDENHAIN
    #---------------------------------------------------------------------------
    def opcCollectorHH_PLC(self):
        
        global plc_isRunning;
        plc_isRunning = True;
        #tabulka nodu v plc heidenhain
        plc_hh_table        = np.array([["datetime",     ""],
                                        ["tool",         "ns=2;s=Technology data.ACTUAL_TOOL_T"],
                                        ["state",        "ns=2;s=Technology data.PROGRAM_STATE"],
                                        ["program",      "ns=2;s=Technology data.PIECE_PROGRAM"],
                                        ["load_s1",      "ns=2;s=S1.Load"],
                                        ["mcs_s1",       "ns=2;s=S1.MCS"],
                                        ["speed_s1",     "ns=2;s=S1.Motor_speed"],
                                        ["temp_s1",      "ns=2;s=S1.Temperature"],
                                        ["load_x",       "ns=2;s=X.Load"],
                                        ["mcs_x",        "ns=2;s=X.MCS"],
                                        ["speed_x",      "ns=2;s=X.Motor_speed"],
                                        ["temp_x",       "ns=2;s=X.Temperature"],
                                        ["load_y",       "ns=2;s=Y.Load"],
                                        ["mcs_y",        "ns=2;s=Y.MCS"],
                                        ["speed_y",      "ns=2;s=Y.Motor_speed"],
                                        ["temp_y",       "ns=2;s=Y.Temperature"],
                                        ["load_z",       "ns=2;s=Z.Load"],
                                        ["mcs_z",        "ns=2;s=Z.MCS"],
                                        ["speed_z",      "ns=2;s=Z.Motor_speed"],
                                        ["temp_z",       "ns=2;s=Z.Temperature"],
                                        ["dev_datetime1","ns=2;s=Machine data.M0111"],
                                        ["dev_x1",       "ns=2;s=Machine data.M0112"],
                                        ["dev_y1",       "ns=2;s=Machine data.M0113"],
                                        ["dev_z1",       "ns=2;s=Machine data.M0114"],
                                        ["dev_datetime2","ns=2;s=Machine data.M0211"],
                                        ["dev_x2",       "ns=2;s=Machine data.M0212"],
                                        ["dev_y2",       "ns=2;s=Machine data.M0213"],
                                        ["dev_z2",       "ns=2;s=Machine data.M0214"],
                                        ["dev_datetime3","ns=2;s=Machine data.M0311"],
                                        ["dev_x3",       "ns=2;s=Machine data.M0312"],
                                        ["dev_y3",       "ns=2;s=Machine data.M0313"],
                                        ["dev_z3",       "ns=2;s=Machine data.M0314"],
                                        ["dev_datetime4","ns=2;s=Machine data.M0411"],
                                        ["dev_x4",       "ns=2;s=Machine data.M0412"],
                                        ["dev_y4",       "ns=2;s=Machine data.M0413"],
                                        ["dev_z4",       "ns=2;s=Machine data.M0414"],
                                        ["dev_datetime5",""],
                                        ["dev_x5",       ""],
                                        ["dev_y5",       ""],
                                        ["dev_z5",       ""]]);
                                        
                                        
        #if not self.ping_(self.host2):
        #    plc_isRunning = False;
        #    return(plc_hh_table, plc_isRunning);
                                                    
        client = Client(self.uri2);
        try:        
            client.connect();
            #sys.stderr.write("Client" + uri+ "Connected\n");
            #self.logger.info("Client" + uri+ "Connected")
            
            plc_hh_table = np.c_[plc_hh_table, np.zeros(len(plc_hh_table))];
            
            for i in range(len(plc_hh_table)):
                if "datetime" in plc_hh_table[i, 0]:
                    plc_hh_table[i, 2] = datetime.now().strftime("%Y-%m-%d %H:%M:%S");
                else:
                    if plc_hh_table[i, 1]:
                        node = client.get_node(str(plc_hh_table[i, 1]));
                        typ  = type(node.get_value());
                        val = float(self.myFloatFormat(node.get_value())) if typ is float else node.get_value();
                        #val = node.get_value() if typ is float else node.get_value();
                        plc_hh_table[i, 2] = val;
            
        except OSError as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect(); 
            return(plc_hh_table, plc_isRunning);

    #---------------------------------------------------------------------------
    # opcCollectorSendToPLC - zapis kompenzacni parametry do PLC HEIDENHAIN
    #
    # OPC Strom: TM-AI
    #              +---Compensation
    #                   +---CompX         
    #                   +---CompY         
    #                   +---CompZ         
    #                   +---setCompX         
    #                   +---setCompY         
    #                   +---setCompZ         
    #                   +---write_comp_val_TM_AI
    # 
    #---------------------------------------------------------------------------
    def opcCollectorSendToPLC(self, df_plc):
        
        global plc_isRunning;
        plc_isRunning = True;

        #return plc_isRunning;  # pouze pro ladeni !!!!!

        uri = self.prefix+self.host2+":"+self.port2;
        plcData = self.PLCData;

        
        if not self.ping_(self.host2):
            plc_isRunning = False;
            return plc_isRunning;

        if not self.ping_(self.host1):
            plc_isRunning = False;
            return plc_isRunning;
        
        client = Client(self.uri2);
        try:        
            client.connect();
                
            root = client.get_root_node();
            # Nacti aktualni hodnoty kompenzace CompX, CompY, CompZ
            # get: CompX                
            node = client.get_node("ns=2;s=Machine data.CompX");
            plcData.CompX = node.get_value();
            
            # get: CompY                
            node = client.get_node("ns=2;s=Machine data.CompY");
            plcData.CompY = node.get_value();
            
            # get: CompZ                
            node = client.get_node("ns=2;s=Machine data.CompZ");
            plcData.CompZ = node.get_value();
            
            # Zapis aktualni hodnoty kompenzace CompX, CompY, CompZ
            plcData.setCompX = int(df_plc[df_plc.columns[1]][0]);
            plcData.setCompY = int(df_plc[df_plc.columns[2]][0]);
            plcData.setCompZ = int(df_plc[df_plc.columns[3]][0]);
            
            
            node_x = client.get_node("ns=2;s=Machine data.setCompX");
            node_x.set_value(ua.DataValue(ua.Variant(plcData.setCompX, ua.VariantType.Int32)));
            
            node_y = client.get_node("ns=2;s=Machine data.setCompY");
            node_y.set_value(ua.DataValue(ua.Variant(plcData.setCompY, ua.VariantType.Int32)));
            
            node_z = client.get_node("ns=2;s=Machine data.setCompZ");
            node_z.set_value(ua.DataValue(ua.Variant(plcData.setCompZ, ua.VariantType.Int32)));
                                
            # Aktualizuj hodnoty v PLC - ns=2;s=Machine data.write_comp_val_TM_AI
            parent = client.get_node("ns=2;s=Machine data")
            method = client.get_node("ns=2;s=Machine data.write_comp_val_TM_AI");
            parent.call_method(method); 
            
            
            
            # Nacti aktualni hodnoty kompenzace CompX, CompY, CompZ
            # get: CompX                
            node = client.get_node("ns=2;s=Machine data.CompX");
            plcData.CompX = node.get_value();
            
            # get: CompY                
            node = client.get_node("ns=2;s=Machine data.CompY");
            plcData.CompY = node.get_value();
            
            # get: CompZ                
            node = client.get_node("ns=2;s=Machine data.CompZ");
            plcData.CompZ = node.get_value();
            return plc_isRunning;
            
        
        except OSError as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
        
        finally:
            client.disconnect();
                
    
    #---------------------------------------------------------------------------
    # prepJitter,  setJitter
    # v pripade ze se v prubehu cteciho cykluz OPC zadna data nezmeni 
    # je nutno jim pridat umele sum, ktery nezhorsi presnost ale umozni
    # neuronove siti predikci. Sit se nedokaze vyrovnat s konstantnim
    # prubehem datove sady.
    #---------------------------------------------------------------------------
    def prepJitter(self, df_size):
        
        jitter = np.random.normal(0, .0005, df_size);
        for i in range(len(jitter)):
            jitter[i] = self.myFloatFormat(jitter[i]);
        return jitter;    
        #return(pd.DataFrame(jitter, columns=["jitter"]));

    #---------------------------------------------------------------------------
    # prepJitter,  setJitter
    #---------------------------------------------------------------------------
    def setJitter(self, df, df_parms, jitter_=False):
        
        if not jitter_:
            return(df);
        
        df_size = len(df);
        
        for col in df.head():
            if  col in df_parms and ptypes.is_numeric_dtype(df[col]):
                jitter = self.prepJitter(df_size);
                df[col].apply(lambda x: np.asarray(x) + np.asarray(jitter));
        return(df);
    
    

    #---------------------------------------------------------------------------
    # opcCollectorGetPredictData - nacti PLC HEIDENHAIN + PLC BR
    # zapis nactena data do br_data - rozsireni treninkove mnoziny 
    # o data z minulosti
    #
    # nacti self.batch vzorku v intervalu self.plc_interval sekund(y) a posli je
    # k predikci
    # self.plc_interval je nastaven na 0.5[s]
    #---------------------------------------------------------------------------
    def opcCollectorGetPredictData(self, df_parms):

        global plc_isRunning;

        if not self.ping_(self.host2):
            plc_isRunning = False;
            return(None);

        if not self.ping_(self.host1):
            plc_isRunning = False;
            return(None);

        # zapis dat z OPC pro rozsireni treninkove mnoziny        
        current_date =  datetime.now().strftime("%Y-%m-%d");
        path_to_df = "./br_data/tm-ai_"+current_date+".csv";
        
        
        #sys.stderr.write("Nacitam "+str(self.batch)+" vzorku dat pro predict - v intervalu 1 [s]\n");
        #self.logger.warning("Nacitam "+str(self.batch)+" vzorku dat pro predict - v intervalu 1 [s]");
        
        for i in range(self.batch):
            br_plc, plc_isRunning = self.opcCollectorBR_PLC();
            if not plc_isRunning:
                return(None);
            
            hh_plc, plc_isRunning = self.opcCollectorHH_PLC();
            if not plc_isRunning:
                return(None);
        
            hh_plc = np.concatenate((hh_plc, br_plc)).T;
            cols = np.array(hh_plc[0]);
            data = list((hh_plc[2]));
            
            if i == 0:
                df_predict = pd.DataFrame(columns = cols);
                
            df_predict.loc[len(df_predict)] = data;
            #sys.stderr.write("." + str(i) +"\r");

            time.sleep(self.plc_interval);
            
        # add jitter 
        df_predict = self.setJitter(df_predict, df_parms, True);    
        # zapis pristi treninkova data
        if exists(path_to_df):
            sys.stderr.write("\nNacteno "+str(self.batch)+" vzorku dat pro predict, pripisuji k:"+path_to_df+ "\n");
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="a", index=False, header=False);
        else:    
            sys.stderr.write("\nNacteno "+str(self.batch)+" vzorku dat pro predict, zapisuji do:"+path_to_df+ "\n");
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="w", index=False, header=True);
            
        return(df_predict);
    
    #---------------------------------------------------------------------------
    # opcCollectorGetDebugData - totez co opcCollectorGetPredictData ovsem
    # data se nectou z OPC ale  z CSV souboru. Toto slouzi jen pro ladeni
    # aby neby zavisle na aktivite OPC serveruuuuu.
    # v pripade ladeni se nezapisuji treninkova data....
    #---------------------------------------------------------------------------
    def opcCollectorGetDebugData(self, df_parms):

        global df_debug_count;
        global df_debug_header;
        df_predict = pd.DataFrame();
        current_date =  datetime.now().strftime("%Y-%m-%d");
        csv_file = "./br_data/predict-debug.csv";
        
        try:
            df_predict         = pd.read_csv(csv_file,
                                             sep=",|;", 
                                             engine='python',  
                                             header=0, 
                                             encoding="utf-8",
                                             skiprows=df_debug_count,
                                             nrows=self.batch
                                        );
        except  EmptyDataError as ex:
            return None;
                                       
        
        df_len = int(len(df_predict));
        if df_len <= 0:
            return None;

        if df_debug_count == 0:
            df_debug_header = df_predict.columns.tolist();
        else:
            df_predict.columns = df_debug_header;    
                
        df_debug_count += self.batch;
        
        # add jitter
        df_predict = self.setJitter(df_predict, df_parms, False);
        df_predict.to_csv("./result/temp"+current_date+".csv");
        time.sleep(1);
            
        return(df_predict);


#---------------------------------------------------------------------------
# DataFactory
# 1. Nacti treninkova data z historie
# 2. Nacti predikcni data z PLC
# 3. Zapis vysledky
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

    def __init__(self, path_to_result, window, logger, debug_mode, batch, current_date):
        
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
        self.window         = window;
        self.df_multiplier  = 1;   
        self.train          = pd.DataFrame();
        self.valid          = pd.DataFrame();
        self.predict        = pd.DataFrame();
        self.logger         = logger;
        self.debug_mode     = debug_mode;
        self.batch          = batch;
        self.current_date   = current_date;
        
        
        self.parms  = [];
        self.header =["typ", 
                      "model", 
                      "epochs", 
                      "units", 
                      "batch", 
                      "actf", 
                      "shuffling", 
                      "txdat1", 
                      "txdat2",
                      "curr_txdat"]; 


        #parametry z parm file - nacte parametry z ./parms/parms.txt
        self.getParmsFromFile();
        # new OPCAgent()
        self.opc = OPCAgent(logger=self.logger, batch=self.batch);

    #---------------------------------------------------------------------------
    # isPing         
    #---------------------------------------------------------------------------
    def isPing(self):
        return(self.opc.isPing());
    #---------------------------------------------------------------------------
    # myFloatFormat         
    #---------------------------------------------------------------------------
    def myFloatFormat(self,x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

    #---------------------------------------------------------------------------
    # myIntFormat         
    #---------------------------------------------------------------------------
    def myIntFormat(self, x):
        return ('%.f' % x).rstrip('.');

    
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
                #self.logger.info("--shuffle = True");
            
            DataTrain_x.test  = df_test;
            DataTrain_x.df_parm_x = self.df_parmx;  # data na ose x, pro rovinu X
            DataTrain_x.df_parm_y = self.df_parmX;  # data na ose y, pro rovinu Y
            DataTrain_x.axis = "OSA_XYZ";
            
            self.train = DataTrain_x.train;
            self.valid = DataTrain_x.valid;
            self.predict = DataTrain_x.test;
            
            return(DataTrain_x);
    
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            sys.stderr.write(str(traceback.print_exc()));

    #---------------------------------------------------------------------------
    # interpolateDF
    # interpoluje data splinem - vyhlazeni schodu na merenych artefaktech
    #---------------------------------------------------------------------------
    def interpolateDF(self, df, smoothing_factor, ip):

        if not ip:
            sys.stderr.write("interpolace artefaktu nebude provedena ip = False\n");
            return df;
        else:
            sys.stderr.write("interpolace artefaktu, smoothing_factor:" + str(smoothing_factor)+"\n");
        
        col_names = list(self.df_parmX);
        x = np.arange(0,len(df));

        for i in range(len(col_names)):
            if "dev" in col_names[i]:
                spl =  inter.UnivariateSpline(x, df[col_names[i]], s=smoothing_factor);
                df[col_names[i]] = spl(x);

        return df;

    
    #---------------------------------------------------------------------------
    # getData
    #---------------------------------------------------------------------------
    def getData(self,
                shuffling       = False,
                timestamp_start = "2022-01-01 00:00:01", 
                timestamp_stop  = "2042-12-31 23:59:59", 
                type            = "predict"):
        
        txdt_b  = False;
        df      = pd.DataFrame();
        df_test = pd.DataFrame();
        
        try:
           
            #self.DataTrainDim.DataTrain = None;
            
            #if "train" in type: 
            files = os.path.join("./br_data", "tm-ai_2022*.csv");
            
            # list souboru pro join
            joined_list = glob.glob(files);
            
            # sort souboru pro join
            joined_list.sort(key=None, reverse=False);

            usecols = ["datetime"];
            for col in self.df_parmX:
                usecols.append(col);

            df = pd.concat([pd.read_csv(csv_file,
                                         sep=",|;", 
                                         engine='python',  
                                         header=0, 
                                         encoding="utf-8",
                                         usecols = usecols 
                                       )
                                    for csv_file in joined_list],
                                    axis=0, 
                                    ignore_index=True
                    );
              # Odfiltruj data kdy stroj byl vypnut
            df = df[(df["dev_x4"] != 0) & (df["dev_y4"] != 0) & (df["dev_z4"] != 0)];
            # bordel pri domluve nazvoslovi...            
            df.columns = df.columns.str.lower();
            # interpoluj celou mnozinu data  
            df = self.interpolateDF(df, 0.01, False);            
            
            # vyber dat dle timestampu
            df["timestamp"] = pd.to_datetime(df["datetime"].str.slice(0, 18));
                
            # treninkova a validacni mnozina    
            df = df[(df["timestamp"] > timestamp_start) & (df["timestamp"] <= timestamp_stop)];
            
            if len(df) <= 1:
                sys.stderr.write("Data pro trenink maji nulovou velikost - exit(0)\n");
                self.logger.error("Data pro trenink maji nulovou velikost - exit(0)");
                os._exit(0);
                
            
            df["index"] = pd.Index(range(0, len(df), 1));
            df.set_index("index", inplace=True);

            size = len(df);
            size_train = math.floor(size * 8 / 12);
            size_valid = math.floor(size * 4 / 12);
            size_test  = math.floor(size * 0 / 12);

            if self.debug_mode:
                # nacti data z predict-debug.csv                        
                df_test = self.opc.opcCollectorGetDebugData(self.df_parmX);
            else:
                # nacti data z OPC.
                df_test = self.opc.opcCollectorGetPredictData(self.df_parmX);
                
            
            if df_test is None:
                sys.stderr.write("Nebyla nactena zadna data pro predikci exit(1)\n");
                self.logger.error("Nebyla nactena zadna data pro predikci exit(1)");
                os._exit(1);
                
            if len(df_test) == 0:
                sys.stderr.write("Patrne nebezi nektery OPC server  - exit(1)\n");
                self.logger.error("Patrne nebezi nektery OPC server - exit(1)");
                os._exit(1);
                       
            if  len(df_test) > 0 and self.window >= len(df_test):
                sys.stderr.write("Prilis maly vzorek dat pro predikci - exit(1)\n");
                self.logger.error("Prilis maly vzorek dat pro predikci - exit(1)");
                os._exit(1);
                    
            if self.df_parmx == None or self.df_parmX == None:
                pass;
            else:
                self.DataTrainDim.DataTrain = self.setDataX( df=df, 
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
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            
    #-----------------------------------------------------------------------
    # saveDataToPLC  - result
    # index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
    #-----------------------------------------------------------------------
    def saveDataToPLC(self, timestamp_start, model, typ, saveresult=True):

        col_names_y = list(self.DataTrain.df_parm_y);
        filename = "./result/plc_archiv/plc_"+model+"_"+str(self.current_date)[0:10]+".csv";
        
        #curent timestamp UTC
        current_time = time.time();
        utc_timestamp = datetime.utcfromtimestamp(current_time);

        l_plc = [];        
        l_plc_col = [];        
        
        l_plc.append( str(utc_timestamp)[0:19]);
        l_plc_col.append("utc");
        
        df_result = pd.DataFrame(self.DataResultDim.DataResultX.y_result, columns = col_names_y);

        for col in col_names_y:
            if "dev" in col:
                mmean = self.myIntFormat(df_result[col].mean() *10000);   #prevod pro PLC (viz dokument Teplotni Kompenzace AI)
                l_plc.append(mmean);                                      #  10 = 0.001 atd...
                l_plc_col.append(col+"mean");
                
        df_plc = pd.DataFrame([l_plc], columns=[l_plc_col]);
        
                                      
        path = Path(filename);
        
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            sys.stderr.write(f'Soubor {filename} existuje - append\n');
            df_plc.to_csv(filename, mode = "a", index=False, header=False, float_format='%.5f');
        else:
            sys.stderr.write(f'Soubor {filename} neexistuje - create\n');
            df_plc.to_csv(filename, mode = "w", index=False, header=True, float_format='%.5f');

        # data do PLC
        result_opc = self.opc.opcCollectorSendToPLC(df_plc=df_plc );
        if result_opc:
            sys.stderr.write("Data do PLC byla zapsana\n");
            self.logger.warning("Data do PLC byla zapsana");
        else:    
            sys.stderr.write("Data do PLC nebyla zapsana !!!!!!\n");
            self.logger.error("Data do PLC nebyla zapsana !!!!!!");
        
            

        # data ke zkoumani zapisujeme v pripade behu typu "train" a zaroven v debug modu
        if "train" in typ and self.debug_mode is True:
            saveresult=True;
        else:
            saveresult=True;
            
        if saveresult:
            sys.stderr.write("Vystupni soubor " + filename + " vznikne.\n");
            self.saveDataResult(timestamp_start, model, typ, saveresult);
            return;
        else:
            sys.stderr.write("Vystupni soubor " + filename + " nevznikne !!!, saveresult = " +str(saveresult) +"\n");
            return;
        
        return;    
        
        
        
        
    #-----------------------------------------------------------------------
    # saveDataResult  - result
    # index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
    #-----------------------------------------------------------------------
    def saveDataResult(self, timestamp_start, model, typ, saveresult=True):
        
        filename = "./result/predicted_"+model+".csv"
        
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
            self.logger.error(traceback.print_exc());
            
        try:
            self.DataTrain.test.reset_index(drop=True, inplace=True)
            

            df_result = pd.DataFrame();
            df_result  = pd.DataFrame(self.DataResultDim.DataResultX.y_result, columns = col_names_y);
            df_result.drop(col_names_drop, inplace=True, axis=1);
            df_result  = pd.DataFrame(np.array(df_result), columns =col_names_predict);
            
            df_result2 = pd.DataFrame();
            df_result2 = pd.DataFrame(self.DataTrain.test);

            #merge - left inner join
            df_result  = pd.concat([df_result.reset_index(drop=True),
                                    df_result2.reset_index(drop=True)],
                                    axis=1);
            
            
            # U gru se na posledni vete vyskytuje NaN
            for col in col_names_dev:
                col = col+"_predict";
                df_result[col] = df_result[col].fillna(0);

            # Absolute Error
            for col in col_names_dev:
                ae = df_result[col].astype(float) - df_result[col+"_predict"].astype(float);
                df_result[col+"_ae"] = ae;
            # Mean Squared Error
            for col in col_names_dev:
                mse = mean_squared_error(df_result[col].astype(float),df_result[col+"_predict"].astype(float));
                df_result[col+"_mse"] = mse;
                
            list_cols     = list({"idx"}); 
            list_cols_avg = list({"idx"}); 
            
            for col in col_names_dev:
                if "dev" in col:
                    list_cols.append(col+"_predict");
                    list_cols_avg.append(col+"_predict_avg");

            
            # MAE avg cols
            # MAE avg cols
            for col in col_names_dev:
                ae = (df_result[col].astype(float) - df_result[col+"_predict"].astype(float));
                df_result[col+"_ae"] = ae;

            path = Path(filename)
            if path.is_file():
                append = True;
            else:
                append = False;
        
            if append:             
                sys.stderr.write(f"Soubor {filename} existuje - append: " + str(len(df_result))+"\n");
                df_result.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
            else:
                sys.stderr.write(f"Soubor {filename} neexistuje - create: " + str(len(df_result))+"\n");
                df_result.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');
                
            self.saveParmsMAE(df_result, model)    

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
        
        return;    

    #-----------------------------------------------------------------------
    # saveParmsMAE - zapise hodnoty MAE v zavislosti na pouzitych parametrech
    #-----------------------------------------------------------------------
    def saveParmsMAE(self, df,  model):

        filename = "./result/parms_mae_"+model+".csv"
        
        #pridej maximalni hodnotu AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    self.header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    self.parms.append(float(res));        
                else:
                    self.header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    self.parms.append(float(res));        
        
        #pridej mean AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    self.header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    self.parms.append(float(res));        
                else:
                    self.header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    self.parms.append(float(res));
        
        df_ae = pd.DataFrame(data=[self.parms], columns=self.header);
        
        path = Path(filename)
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            sys.stderr.write(f'Soubor {filename} existuje - append\n');
            df_ae.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
        else:
            sys.stderr.write(f'Soubor {filename} neexistuje - create\n');
            df_ae.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');
            
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
            sys.stderr.write("parametry nacteny z "+ parmfile +"\n");       
            self.logger.info("parametry nacteny z "+ parmfile);                 
            
                
        except:
            sys.stderr.write("Soubor parametru "+ parmfile + " nenalezen!\n");                
            sys.stderr.write("Parametry pro trenink site budou nastaveny implicitne v programu\n");                 
            self.logger.info("Soubor parametru " + parmfile + " nenalezen!");
        
        return();  
    
    #---------------------------------------------------------------------------
    # DataFactory
    #---------------------------------------------------------------------------
    def prepareParmsPredict(self):
        
        i = 0;
        for i in self.df_parmX:
            df_parmX_predict[i] = self.df_parmX[i]+"_predict";
        i = 0;
             
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
            sys.stderr.write("prilis maly  vektor dat k uceni!!! parametr window je vetsi nez delka vst. vektoru \n");
            self.logger.error("prilis maly  vektor dat k uceni!!! parametr window je vetsi nez delka vst. vektoru");
        
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
        


    #---------------------------------------------------------------------------
    # DataFactory getter metody
    #---------------------------------------------------------------------------
    def getDf_parmx(self):
        return self.df_parmx;
    
    def getDf_parmX(self):
        return self.df_parmX;

    def getDfTrainData(self):
        return (self.train);
    
    def getDfValidData(self):
        return (self.valid);
    
    def getDfPredictData(self):
        return (self.predict);
    
    def setParms(self, parms):
        self.parms = parms;
    
    def getParms(self):
        return self.parms;
    
    def setHeader(self, header):
        self.header = header;
    
    def getHeader(self):
        return self.header;

    

#---------------------------------------------------------------------------
# Neuronova Vrstava DENSE
#---------------------------------------------------------------------------
class NeuronLayerDENSE():
    #definice datoveho ramce
    
    @dataclass
    class DataSet:
        X_dataset: object;             #data k uceni
        y_dataset: object;             #vstupni data
        cols:      int;                #pocet sloupcu v datove sade

    def __init__(self, 
                 path_to_result, 
                 typ, 
                 model, 
                 epochs, 
                 batch, 
                 txdat1, 
                 txdat2, 
                 window, 
                 units, 
                 shuffling, 
                 actf, 
                 logger,
                 debug_mode,
                 current_date=""
            ):
        
        self.path_to_result = path_to_result; 
        self.typ    = typ;
        self.model_ = model;
        self.epochs = epochs;
        self.batch  = batch;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        self.logger = logger;
        self.debug_mode = debug_mode;
        self.current_date=current_date;

        self.df     = pd.DataFrame();
        self.df_out = pd.DataFrame();
        self.graph  = None;
        self.window = window;
        self.units  = units;
        self.shuffling = shuffling;
        self.actf   = actf;
        self.data           = None;
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;

    #---------------------------------------------------------------------------
    # isPing ????
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.data.isPing();

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
            initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            model.add(tf.keras.Input(shape=(inp_size,)));
            model.add(layers.Dense(units=inp_size,       activation=self.actf, kernel_initializer=initializer));
            model.add(layers.Dense(units=self.units,     activation=self.actf, kernel_initializer=initializer));
            model.add(layers.Dense(units=self.units,     activation=self.actf, kernel_initializer=initializer));
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
        
            model.save('./models/model_'+self.model_+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)
        
        # make predictions for the input data
            return (model);
    
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava DENSE predict
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
    #
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
        # normalizace vstupnich a vystupnich testovacich dat 
            x_test        =  self.x_train_scaler.transform(x_test);
        # predikce site
            y_result = model.predict(x_test);
        # zapis syrove predikce ke zkoumani    
            y_result  = self.y_train_scaler.inverse_transform(y_result);
            
            columns=DataTrain.df_parm_y
            dfy= pd.DataFrame();
            dfy  = pd.DataFrame(y_result, columns=columns);


            if self.debug_mode:  # zapis syrova data do raw.csv
                model.summary();
                if exists("./result/raw.csv"):
                    dfy.to_csv("./result/raw.csv",  encoding="utf-8", mode="a", index=False, header=False);
                else:    
                    dfy.to_csv("./result/raw.csv",  encoding="utf-8", mode="w", index=False, header=True);

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            sys.stderr.write("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem\n");
            sys.stderr.write("          zkuste nejdrive --typ == train !!!\n");
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec_x(self, data, graph):

        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                sys.stderr.write("Start vcetne treninku, model bude zapsan\n");
                self.logger.warning("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrain);
            else:    
                sys.stderr.write("Start bez treninku - model bude nacten\n");
                self.logger.warning("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model_+'_'+ data.DataTrainDim.DataTrain.axis);
            
            if  data.DataTrainDim.DataTrain.test is None or len(data.DataTrainDim.DataTrain.test) == 0:
                sys.stderr.write("Data pro predikci nejsou k dispozici....\n");
                self.logger.error("Data pro predikci nejsou k dispozici....");
                return();
           
            
            data.DataResultDim.DataResultX = self.neuralNetworkDENSEpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataToPLC(self.txdat1, self.model_, self.typ);
                
            stopTime = datetime.now();
            sys.stderr.write("cas vypoctu[s] " + str(stopTime - startTime) + "\n");
            self.logger.info("cas vypoctu[s] %s",  str(stopTime - startTime));

            return();

        except FileNotFoundError as e:
            sys.stderr.write(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!\n");    
            self.logger.error(f"Nenalezen model, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");
        except Exception as ex:
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());
            
    #------------------------------------------------------------------------
    # neuralNetworkDENSEexec
    #------------------------------------------------------------------------
    def neuralNetworkDENSEexec(self):

        try:
            sys.stderr.write("\nPocet GPU jader: "+ str(len(tf.config.experimental.list_physical_devices('GPU')))+"\n")
            self.data = DataFactory(path_to_result=self.path_to_result, 
                                    window=self.window,
                                    logger=self.logger,
                                    batch=self.batch,
                                    debug_mode=self.debug_mode,
                                    current_date=self.current_date);
        
                
            parms = [self.typ,
                     self.model_,
                     self.epochs,
                     self.units,
                     self.batch,
                     self.actf,
                     str(self.shuffling), 
                     self.txdat1, 
                     self.txdat2,
                     str(self.current_date)];
            
            self.data.setParms(parms);
            
            self.data.Data = self.data.getData(shuffling=self.shuffling, 
                                               timestamp_start=self.txdat1, 
                                               timestamp_stop=self.txdat2,
                                               type=self.typ);
            
            if self.data.getDfTrainData().empty and "predict" in self.typ:
                sys.stderr.write("Data pro predict, nejsou k dispozici\n");
                self.logger.error("Data pro predict, nejsou k dispozici");
                return(0);

    # Execute.....
            self.neuralNetworkDENSEexec_x(data=self.data, graph=self.graph);
            
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                pass;
                #u demona nebudeme archivovat model....
                #saveModelToArchiv(model="DENSE", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            self.logger.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # setter - getter
    #---------------------------------------------------------------------------
    # dense, lstm
    def getModel(self):
        return self.model_;
    # predict, train
    def getTyp(self):
        return self.typ;
     
    def setTyp(self, typ):
        self.typ = typ;


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

    def __init__(self, 
                 path_to_result, 
                 typ, 
                 model, 
                 epochs, 
                 batch, 
                 txdat1, 
                 txdat2, 
                 window, 
                 units, 
                 shuffling, 
                 actf,
                 logger,
                 debug_mode,
                 current_date):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model_ = model;
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
        self.logger = logger;
        self.debug_mode = debug_mode;
        self.current_date=current_date,
        self.data           = None;
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
        #    model.add(Dropout(0.2));
        #   model.add(LSTM(units = self.units, return_sequences=True));
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

            
        # zapis modelu    
            model.save('./models/model_'+self.model_+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)

        # make predictions for the input data
            return (model);
        
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
        
        
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
            sys.stderr.write("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem\n");
            sys.stderr.write("          zkuste nejdrive --typ == train !!!\n");
            traceback.print_exc();
            self.logger.error(traceback.print_exc());




    #---------------------------------------------------------------------------
    # neuralNetworkLSTMexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMexec_x(self, data, graph):
        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                sys.stderr.write("Start vcetne treninku, model bude zapsan\n");
                self.logger.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkLSTMtrain(data.DataTrainDim.DataTrain);
            else:    
                sys.stderr.write("Start bez treninku - model bude nacten\n");
                self.logger.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model_+'_'+ data.DataTrainDim.DataTrain.axis);
            
            data.DataResultDim.DataResultX = self.neuralNetworkLSTMpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataToPLC(self.txdat1, self.model_, self.typ);
                
            stopTime = datetime.now();
            self.logger.info("cas vypoctu[s] %s",  str(stopTime - startTime));
            return(0);        

        except FileNotFoundError as e:
            sys.stderr.write(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");    
            self.logger.error(f"Nenalezen model site pro osu X, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());

    #------------------------------------------------------------------------
    # neuralNetworkLSTMexec
    #------------------------------------------------------------------------
    def neuralNetworkLSTMexec(self):
        
        try:
            sys.stderr.write("Pocet GPU jader: "+ str(len(tf.config.experimental.list_physical_devices('GPU')))+"\n");

            self.data = DataFactory(path_to_result=self.path_to_result, 
                                    window=self.window,
                                    logger=self.logger,
                                    batch=self.batch,
                                    debug_mode=self.debug_mode,
                                    current_date=self.current_date );
        
            parms = [self.typ,
                     self.model_,
                     self.epochs,
                     self.units,
                     self.batch,
                     self.actf,
                     str(self.shuffling), 
                     self.txdat1, 
                     self.txdat2,
                     str(self.current_date)];
            
            self.data.setParms(parms);
        
            self.data.Data = self.data.getData(shuffling=self.shuffling, timestamp_start=self.txdat1, timestamp_stop=self.txdat2);

    # osa X    
            if self.data.DataTrainDim.DataTrain  == None:
                sys.stderr.write("Osa X je disable...\n");
                self.logger.warning("Osa X je disable...");
            else:
                self.neuralNetworkLSTMexec_x(data=self.data, graph=self.graph);
            
        
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                saveModelToArchiv(model="LSTM", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());


    #---------------------------------------------------------------------------
    # setter - getter
    #---------------------------------------------------------------------------
    def getModel(self):
        return self.model_;

    def getTyp(self):
        return self.typ;

    def setTyp(self, typ):
        self.typ = typ;

#------------------------------------------------------------------------
# Daemon    
#------------------------------------------------------------------------
class NeuroDaemon():
    
    def __init__(self, 
                 pidf, 
                 logf, 
                 path_to_result, 
                 model, epochs, 
                 batch, 
                 units, 
                 shuffling, 
                 txdat1, 
                 txdat2, 
                 actf, 
                 window, 
                 debug_mode,
                 current_date
            ):
        
        self.pidf           = pidf; 
        self.logf           = logf; 
        self.path_to_result = path_to_result;
        self.model_         = model;
        self.epochs         = epochs;
        self.batch          = batch;
        self.units          = units;
        self.shuffling      = shuffling;
        self.txdat1         = txdat1; 
        self.txdat2         = txdat2;
        self.actf           = actf;
        self.window         = window;
        self.debug_mode     = debug_mode;
        self.current_date   = current_date;
        
        self.neural         = None;
        self.logger         = None;
        self.train_counter  = 0;
        self.train_counter_max = 20;

        
        
        
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def getLogFileHandles(self,logger):
        """ Get a list of filehandle numbers from logger
            to be handed to DaemonContext.files_preserve
        """
        handles = []
        for handler in logger.handlers:
            handles.append(handler.stream.fileno())
            if logger.parent:
                handles += self.getLogFileHandles(logger.parent)
        return handles




#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def setLogger(self, logf):
        progname = os.path.basename(__file__);
        '''
        logging.basicConfig(filename=logf,
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S');
        '''            
        logging.basicConfig(filename=logf,
                            format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            filemode='a',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            level=logging.WARNING)
                    

        log_handler = logging.StreamHandler();

        logger = logging.getLogger("parent");
        logger.addHandler(log_handler)

        #logging.getLogger("opcua").setLevel(logging.Critical)
        
        return(logger);


    
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def setLogHandler(self):
        return;
    '''
        progname = os.path.basename(__file__);

        logging.basicConfig(filename="./log/"+progname+".log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S');

        log_handler = logging.StreamHandler();
        logger = logging.getLogger("parent");
        logger.addHandler(log_handler)
        return(log_handler);
    '''    

#------------------------------------------------------------------------
# start daemon pro parametr LSTM
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemonLSTM(self):

        global plc_isRunning;
        
        self.logger = self.setLogger(self.logf);
        self.train_counter = 0;

        self.neural = NeuronLayerLSTM(path_to_result=path_to_result, 
                                      typ            = "train", 
                                      model          = self.model_, 
                                      epochs         = self.epochs, 
                                      batch          = self.batch,
                                      txdat1         = self.txdat1,
                                      txdat2         = self.txdat2,
                                      window         = self.window,
                                      units          = self.units,
                                      shuffling      = self.shuffling,
                                      actf           = self.actf, 
                                      logger         = self.logger,
                                      debug_mode     = self.debug_mode,
                                      current_date   = self.current_date
                                    );

        

        typ = "train";
        
        if plc_isRunning:
            sleep_interval =  1;     # 1 sekunda
        else:
            sleep_interval =  600;     #600 sekund
                
            
        #predikcni beh
        while True:
            current_date =  datetime.now().strftime("%Y-%m-%d %H:%M:%S");
            #train
            if plc_isRunning and self.train_counter == 0:
                self.neural.setTyp("train");
            #predict
            if plc_isRunning and self.train_counter > 0:
                self.neural.setTyp("predict");
                
            self.logger.info("TYP:"+ current_date + self.neural.getTyp());

            if plc_isRunning:
                
                self.neural.neuralNetworkLSTMexec();

                if self.train_counter < self.train_counter_max:
                    sys.stderr.write("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +"\n");
                    self.logger.warning("PLC ON:"+ current_date + " cnt:" + str(self.train_counter));
                    self.train_counter = 1;
                else:
                    sys.stderr.write("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +" nasleduje trenink\n");
                    self.logger.warning("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +" nasleduje trenink");
                    self.train_counter = 0; # priprav na dalsi trenink
            else:
                sys.stderr.write("PLC OFF:"+ current_date +"\n");
                self.logger.warning("PLC OFF:"+ current_date);
                self.train_counter = 0;        
                    
            time.sleep(sleep_interval);
            
            
        
#------------------------------------------------------------------------
# start daemon pro parametr DENSE
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemonDENSE(self):
        
        global plc_isRunning;
        
        self.logger = self.setLogger(self.logf);
        self.train_counter = 0;

        self.neural = NeuronLayerDENSE(path_to_result=path_to_result, 
                                      typ            = "train", 
                                      model          = self.model_, 
                                      epochs         = self.epochs, 
                                      batch          = self.batch,
                                      txdat1         = self.txdat1,
                                      txdat2         = self.txdat2,
                                      window         = self.window,
                                      units          = self.units,
                                      shuffling      = self.shuffling,
                                      actf           = self.actf, 
                                      logger         = self.logger,
                                      debug_mode     = self.debug_mode,
                                      current_date   = self.current_date
                                    );

        

        typ = "train";
        
        if plc_isRunning:
            sleep_interval =  1;     # 1 sekunda
        else:
            sleep_interval =  600;     #600 sekund
                
            
        #predikcni beh
        while True:
            current_date =  datetime.now().strftime("%Y-%m-%d %H:%M:%S");
            #train
            if plc_isRunning and self.train_counter == 0:
                self.neural.setTyp("train");
            #predict
            if plc_isRunning and self.train_counter > 0:
                self.neural.setTyp("predict");
                
            self.logger.warning("TYP:"+ current_date + self.neural.getTyp());

            if plc_isRunning:
                
                self.neural.neuralNetworkDENSEexec();

                if self.train_counter < self.train_counter_max:
                    sys.stderr.write("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +"\n");
                    self.logger.warning("PLC ON:"+ current_date + " cnt:" + str(self.train_counter));
                    self.train_counter += 1;
                else:
                    sys.stderr.write("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +" nasleduje trenink\n");
                    self.logger.warning("PLC ON:"+ current_date + " cnt:" + str(self.train_counter) +" nasleduje trenink");
                    self.train_counter = 0; # priprav na dalsi trenink
            else:
                sys.stderr.write("PLC OFF:"+ current_date +"\n");
                self.logger.warning("PLC OFF:"+ current_date);
                self.train_counter = 0;        
                    
            time.sleep(sleep_interval);
            
        
    #------------------------------------------------------------------------
    # info daemon
    #------------------------------------------------------------------------
    def info(self):
        sys.stderr.write("daemon pro sledovani a kompenzaci teplotnich zmen stroje\n");
        return;
    #------------------------------------------------------------------------
    # daemonize...
    #    do the UNIX double-fork magic, see Stevens' "Advanced
    #    Programming in the UNIX Environment" for details (ISBN 0201563177)
    #    http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
    #
    #------------------------------------------------------------------------
    def daemonize(self):
        
        sys.stderr.write("daemonize.....\n");

        #l_handler = self.setLogHandler();
        
        context = daemon.DaemonContext(working_directory='./',
                                       #pidfile=lockfile.FileLock(self.pidf),
                                       stdout=sys.stdout,
                                       stderr=sys.stderr,
                                       umask=0o002,
                                       pidfile=pidfile.TimeoutPIDLockFile(self.pidf),                                       
                                       #files_preserve = [l_handler]
                  );

        context.signal_map = {
            #signal.SIGKILL:  self.stop(),
            #signal.SIGINT:  self.stop(),
            signal.SIGHUP:  'terminate',
            #signal.SIGUSR1: reload_program_config,
        }
        
        return(context);

    #------------------------------------------------------------------------
    # start daemon
    #------------------------------------------------------------------------
    def start(self):
        # Kontrola existence pid - daemon run...
        try:                                                                                                                
            pf = open(self.pidf,'r');                                                                                    
            pid = int(pf.read().strip());                                                                                   
            pf.close();                                                                                                     
        except:                                                                                                             
            pid = None;                                                                                                     
                                                                                                                            
        if pid:                                                                                                             
            message = "pid procesu %d existuje!!!. Daemon patrne bezi - exit(1)\n";                                         
            sys.stderr.write(message % pid);                                                                       
            os._exit(1);                                                                                                    

        context = self.daemonize();
        
        try:
            with context:
                if "DENSE" in self.model_:
                    self.runDaemonDENSE();
                elif "LSTM" in self.model_:
                    self.runDaemonLSTM();
                else:    
                    sys.stderr.write("chybny parametr model:"+self.model_+"\n");
                    
        except (Exception, getopt.GetoptError)  as ex:
            traceback.print_exc();
            sys.stderr.write(+str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            help(activations);

    #------------------------------------------------------------------------
    # start daemon
    #------------------------------------------------------------------------
    def stop(self):
        try:                                                                                                                
            pf = open(self.pidf,'r');                                                                                    
            pid = int(pf.read().strip());                                                                                   
            pf.close();                                                                                                     
        except:                                                                                                             
            pid = None;                                                                                                     
                                                                                                                            
        if pid is None:
            return;                                                                                                             
            #message = "pid procesu neexistuje!!!. Daemon patrne nebezi - exit(1)\n";                                         
            #sys.stderr.write(message);                                                                       
            #os._exit(1);
        else:                                                                                                        
            message = "pid procesu %d existuje!!!. Daemon %d stop....\n";                                         
            sys.stderr.write(message % pid);                                                                       
            os._exit(0);

            
    #------------------------------------------------------------------------
    # getter, setter metody
    #------------------------------------------------------------------------
    def getDebug(self):
        return self.debug_mode;
    
    def setDebug(self, debug_mode):
        self.debug_mode = debug_mode;

    def getLogf(self):
        return self.logf;
    
    def getPidf(self):
        return self.pidf;


#------------------------------------------------------------------------
# MAIN CLASS
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# saveModelToArchiv - zaloha modelu, spusteno jen pri parametru train
#------------------------------------------------------------------------
def saveModelToArchiv(model, dest_path, data):

    axes = np.array([data.DataTrainDim.DataTrain.axis]);
    src_dir  = "./models/model_"+model+"_";
    dest_dir = "/models/model_"+model+"_";
    try:
        if data.DataTrainDim.DataTrain  == None:
            pass;
        else:    
            src_dir_  = src_dir + axes[0]
            dest_dir_ = dest_path + dest_dir + axes[0]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
            
        return(0);    
   
    except Exception as ex:
        traceback.print_exc();
        sys.stderr.write(str(traceback.print_exc()));
        #logger.error(traceback.print_exc());
 


#------------------------------------------------------------------------
# setEnv
#------------------------------------------------------------------------
def setEnv(path, model, type):

        progname = os.path.basename(__file__);
        current_date =  datetime.now().strftime("%Y-%m-%d %H:%M:%S");
        path1 = path+model+"_3D";
        path2 = path1+"/"+current_date+"_"+type
                
        
        try: 
            os.mkdir("./log");
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir("./pid");
        except OSError as error: 
            pass; 
 
        try: 
            os.mkdir("./result")
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir("./result/plc_archiv")
        except OSError as error: 
            pass; 
 

        try: 
            os.mkdir("./temp");
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir(path2);
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir(path2+"/src");
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir("./models");
        except OSError as error: 
            pass;
         
        return path2, current_date;    

#------------------------------------------------------------------------
# Exception handler
#------------------------------------------------------------------------
def exception_handler(exctype, value, tb):
    pass;
    #logger.error(exctype)
    #logger.error(value)
    #logger.error(traceback.extract_tb(tb))

#------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------
def help (activations):
    print ("HELP:");
    print ("------------------------------------------------------------------------------------------------------ ");
    print ("pouziti: <nazev_programu> <arg-1> <arg-2> <arg-3>,..., <arg-n>");
    print (" ");
    print ("        --help            list help ")
    print (" ");
    print (" ");
    print ("        --model           model neuronove site 'DENSE', 'LSTM', 'GRU', 'BIDI'")
    print ("                                 DENSE - zakladni model site - nejmene narocny na system")
    print ("                                 LSTM - Narocny model rekurentni site s feedback vazbami")
    print (" ");
    print ("        --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
    print ("                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
    print ("                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
    print ("                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
    print ("                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
    print (" ");
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
    print ("                                 se v uvahu cela mnozina dat k trenovani, to znamena:");
    print ("                                 od pocatku mereni: 2022-02-15 00:00:00 ");
    print ("                                 do konce   mereni: current timestamp() - 1 [den] ");
    print (" ");
    print ("POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM, GRU nebo BIDI!!!");
    print ("       pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise ");
    print ("       natrenovany model site..");
    print ("       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.");
    print ("       spusti normalni beh site z jiz natrenovaneho modelu.");
    print ("       Takze: budte trpelivi...");
    print (" ");
    print (" ");
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
# preddefinovane hyperparametry neuronove site
# ["typ=", "model=", "epochs=", "batch=", "units=", "shuffle=","actf=", "txdat1=","txdat2=", "help="])
#   parm0  = sys.argv[0];   - nazev programu
#   typ    = "train";       - typ behu site <train, predict>
#   model  = "DENSE";       - model site
#   epochs = 128;           - pocet treninkovych metod
#   batch  = 128;           - velikost davky
#   units  = 512;           - pocet neuronu
#   txdat1 = "";            - timestamp start - vyber dat treninkove a validacni mnoziny
#   txdat2 = "";            - timestamp stop  - vyber dat treninkove a validacni mnoziny
#   shuffling = False;      - promichat nahodne data <True, False>
#   actf = "tanh";          - aktivacni funkce
#   pid = 0;
#------------------------------------------------------------------------
def main(argv):
    
    global path_to_result;
    global current_date;
    global plc_isRunning
    global g_window;
    g_window = 24;
    plc_isRunning = True;
    
    path_to_result = "./result";
    current_date ="";
    pidf = "./pid/ai-daemon.pid"
    logf = "./log/ai-daemon.log"
    
    '''
    debug_mode - parametr pro ladeni programu, pokud nejedou OPC servery, 
    data se nacitaji ze souboru, csv_file = "./br_data/predict-debug.csv", 
    ktery ma stejny format jako data z OPC serveruuu.
    
    debug_mode se zaroven posila do DataFactory.getData, kde se rozhoduje,
    zda se budou data cist z OPC a nebo z predict-debug souboru. 
    V predict-debug souboru jsou data nasbirana z OPC serveru 
    v intervalu<2022-07-19 08:06:01, 2022-07-19 11:56:08> v sekundovych
    vzorcich.
    '''
    
    
    # parametry 
    parm0          = sys.argv[0];        
    model          = "DENSE";
    epochs         = 48;
    batch          = 64;
    units          = 71;
    txdat1         = "2022-02-15 00:00:00";
    txdat2         = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S");
    shuffling      = False;
    actf           = "relu";
    pid            = 0;
    status         = "";
    startTime      = datetime.now();
    type           = "train";
    debug_mode     = False;
    #debug_mode     = True;
    
        
    activations = [["deserialize", "Returns activation function given a string identifier"],
                   ["elu", "Exponential Linear Unit"],
                   ["exponential", "Exponential activation function"],
                   ["gelu", "Gaussian error linear unit (GELU) activation function"],
                   ["get", "Returns function"],
                   ["hard_sigmoid", "Hard sigmoid activation function"],
                   ["linear", "Linear activation function (pass-through)"],
                   ["relu", "Rectified linear unit activation function"],
                   ["selu","Scaled Exponential Linear Unit"],
                   ["serialize","Returns the string identifier of an activation function"],
                   ["sigmoid","Sigmoid activation function: sigmoid(x) = 1 / (1 + exp(-x))"],
                   ["softmax","Softmax converts a vector of values to a probability distribution"],
                   ["softplus","Softplus activation function: softplus(x) = log(exp(x) + 1)"],
                   ["softsign","Softsign activation function: softsign(x) = x / (abs(x) + 1)"],
                   ["swish","Swish activation function: swish(x) = x * sigmoid(x)"],
                   ["tanh","Hyperbolic tangent activation function"],
                   ["None","pro GRU a LSTM site"]];


        #init objektu daemona
    path_to_result, current_date = setEnv(path=path_to_result, model=model, type=type);
        
    try:
        sys.stderr.write("start...\n");
        #logger.info("start...");

        #kontrola platne aktivacni funkce        
        if not checkActf(actf, activations):
            print("Chybna aktivacni funkce - viz help...");
            help(activations)
            sys.exit(1);
            
        #logger.info("Verze TensorFlow :" + tf.__version__);
        print("Verze TensorFlow :", tf.__version__);
        
        txdat_format = "%Y-%m-%d %h:%m:%s"

        opts = [];
        args = [];
        try:
            opts, args = getopt.getopt(sys.argv[1:],
                                       "hs:d:p:l:m:e:b:u:a:t1:t2:h:x",
                                      ["status=",
                                       "debug_mode=", 
                                       "pidfile=", 
                                       "logfile=", 
                                       "model=", 
                                       "epochs=", 
                                       "batch=", 
                                       "units=", 
                                       "actf=", 
                                       "txdat1=", 
                                       "txdat2=", 
                                       "help="]
                                    );
            
        except getopt.GetoptError:
            print("Chyba pri parsovani parametru:",opts);
            help(activations);
            sys.exit(1);
            
        for opt, arg in opts:
            
            if opt in ["-s","--status"]:
                status = arg;
                
                if "start" in status or "stop" in status or "restart" in status or "status" in status:
                    pass;
                else:
                    print("Chyba stavu demona povoleny jen <start, stop, restart a status");
                    help(activations);
                    sys.exit(1);    
                    
            elif opt in ["-d","--debug_mode"]:    #debug nodebug....
                
                if "nodebug" in arg.lower():
                    debug_mode = False;
                    txdat2 = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S");
                else:
                    debug_mode = True;
                    txdat2 = "2022-06-29 23:59:59";
            
            elif opt in ["-p","--pidfile"]:
                pidf = arg;
        
            elif opt in ["-l","--logfile"]:
                logf = arg;

            elif opt in ("-m", "--model"):
                model = arg.upper();
                
            elif opt in ("-e", "--epochs"):
                try:
                    r = range(32-1, 256+1);
                    epochs = int(arg);
                    if epochs not in r:
                        print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <32, 256>");
                        help(activations);
                        sys.exit(1)    
                        
                except:
                    print("Chyba pri parsovani parametru: parametr 'epochs' musi byt cislo typu integer v rozsahu <32, 256>");
                    help(activations);
                    sys.exit(1);
                        
            elif opt in ("-u", "--units"):
                try:
                    r = range(8-1, 2048+1);
                    units = int(arg);
                    if units not in r:
                        print("Chyba pri parsovani parametru: parametr 'units' musi byt cislo typu integer v rozsahu <8, 2048>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Chyba pri parsovani parametru: parametr 'units' musi byt cislo typu integer v rozsahu <8, 2048>");
                    help(activations);
                    sys.exit(1);

            elif opt in ["-af","--actf"]:
                actf = arg.lower();
                if actf == "DENSE":
                    if not checkActf(actf, activations):
                        print("Chybna aktivacni funkce - viz help...");
                        help(activations)
                        sys.exit(1);
                        
            elif opt in ("-b", "--batch"):
                try:
                    r = range(16-1, 2048+1);
                    batch = int(arg);
                    if batch not in r:
                        print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <16, 2048>");
                        help(activations);
                        sys.exit(1)    
                except:    
                    print("Chyba pri parsovani parametru: parametr 'batch' musi byt cislo typu integer v rozsahu <16, 2048>");
                    help(activations);
                    sys.exit(1)
                    

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

         
            elif opt in ["-h","--help"]:
                help(activations);
                sys.exit(0);
        
        if len(sys.argv) < 2:
            help(activations);
            sys.exit(1);
            
            
        #-----------------------------------------------------------------------------
        #obsluha demona
        #-----------------------------------------------------------------------------
        if "start" in status:
            try:
                
                # new NeuroDaemon
                daemon_ = NeuroDaemon(pidf           = pidf,
                                      logf           = logf,
                                      path_to_result = path_to_result,
                                      model          = model,
                                      epochs         = epochs,
                                      batch          = batch,
                                      units          = units,
                                      shuffling      = shuffling,
                                      txdat1         = txdat1, 
                                      txdat2         = txdat2,
                                      actf           = actf,
                                      window         = g_window,
                                      debug_mode     = debug_mode,
                                      current_date   = current_date
                                );
       
                daemon_.info();
                
                if debug_mode:
                    sys.stderr.write("ai-daemon run v debug mode...\n");
                    if "DENSE" in model:
                        daemon_.runDaemonDENSE();
                    else:    
                        daemon_.runDaemonLSTM();
                else:    
                    sys.stderr.write("ai-daemon start...\n");
                    if "DENSE" in model:
                        daemon_.runDaemonDENSE();
                    else:    
                        daemon_.runDaemonLSTM();
                    #daemon_.start();
                    
            except:
                traceback.print_exc();
                sys.stderr.write(str(traceback.print_exc()));
                sys.stderr.write("ai-daemon start exception...\n");
                pass
            
        elif "stop" in status:
            sys.stderr.write("ai-daemon stop...\n");
            daemon_.stop();
            
        elif "restart" in status:
            sys.stderr.write("ai-daemon restart...\n");
            daemon_.restart()
            
        elif "status" in status:
            try:
                pf = file(PIDFILE,'r');
                pid = int(pf.read().strip())
                pf.close();
            except IOError:
                pid = None;
            except SystemExit:
                pid = None;
            if pid:
                sys.stderr.write("Daemon ai-daemon je ve stavu run...\n");
            else:
                sys.stderr.write("Daemon ai-daemon je ve stavu stop....\n");
        else:
            sys.stderr.write("Neznamy parametr:<"+status+">\n");
            sys.exit(0)
        
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        #logger.error(traceback.print_exc());
        help(activations);
        
    finally:    
        stopTime = datetime.now();
        #print("cas vypoctu [s]", stopTime - startTime );
        #logger.info("cas vypoctu [s] %s",  str(stopTime - startTime));
        sys.stderr.write("\nstop obsluzneho programu pro demona - ai-daemon...\n");
        #logger.info("stop obsluzneho programu pro demona - ai-daemon...");
        sys.exit(0);




#------------------------------------------------------------------------
# main entry point
#------------------------------------------------------------------------
        
if __name__ == "__main__":

    main(sys.argv[1:])
    

