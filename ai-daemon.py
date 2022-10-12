#!/usr/bin/python3

#------------------------------------------------------------------------------
# ai-daemon
# (C) GNU General Public 1License,
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
import socket;
import grp;
import daemon; 
import lockfile;
import random;
import webbrowser;
import glob as glob;
import pandas as pd;
import seaborn as sns;
import tensorflow as tf;
import math;
import numpy as np;
import shutil;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import platform;
import pandas.api.types as ptypes
import pickle;

from matplotlib import cm;
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
from concurrent.futures import ThreadPoolExecutor
from signal import SIGTERM

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
#scipy
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import InterpolatedUnivariateSpline;
from scipy.interpolate import UnivariateSpline;

#opcua
from opcua import ua
from opcua import *
from opcua.common.ua_utils import data_type_to_variant_type

from subprocess import call;
from plistlib import InvalidFileException
from tensorflow.python.eager.function import np_arrays
from pandas.errors import EmptyDataError
from keras.saving.utils_v1.mode_keys import is_train

#logger
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from logging import handlers



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
    def __init__(self, batch):
        
        self.logger    = logging.getLogger('root');

        self.prefix    = "opc.tcp://";
        self.host1     = "opc998.os.zps"; # BR-PLC
        self.port1     = "4840";
        self.host2     = "opc999.os.zps";# HEIDENHANIN-PLC
        self.port2     = "48010";
        self.is_ping   = False;
        self.batch     = batch;
        
        self.df_debug = pd.DataFrame();
        self.uri1 = self.prefix+self.host1+":"+self.port1;
        self.uri2 = self.prefix+self.host2+":"+self.port2;
        
        self.plc_timer = 2 #[s]
        
        
        
    #---------------------------------------------------------------------------
    # ping_         
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
        
    #------------------------------------------------------------------------------        
    # pingSocket - nepotrebuje root prava...
    #------------------------------------------------------------------------------
    def pingSocket(self, host, port):

        self.is_ping = False;
        ping_cnt = 0;

        socket_ = None;
    # Loop while less than max count or until Ctrl-C caught
        while ping_cnt < 2:
            ping_cnt += 1;    
            try:
                # New Socket
                socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
                socket_.settimeout(1);
                # Try to Connect
                socket_.connect((host, int(port)));
                self.is_ping = True;
                return(self.is_ping);

            except socket.error as err:
                self.logger.info("pingSocket: failed with error %s" %(err));
                self.is_ping = False;
                return(self.is_ping);

                # Connect TimeOut
            except socket.timeout:
                self.logger.info("pingSocket: socket.timeout return(False)");
                self.is_ping = False;
                return(self.is_ping);
                
                # OS error
            except OSError as e:
                self.logger.info("pingSocket: OSError return(False)");
                self.is_ping = False;
                return(self.is_ping);
                

                # Connect refused
            except ConnectionRefusedError as e:    
                self.logger.info("pingSocket: ConnectionRefusedError return(False)");
                self.is_ping = False;
                return(self.is_ping);
                
                # Other error
            except:    
                self.logger.info("pingSocket: OtherError return(False)");
                self.is_ping = False;
                return(self.is_ping);

            finally:
                #socket_.shutdown(socket.SHUT_RD);
                socket_.close();

        # end while            

        self.logger.info("pingSocket: ping return(False)");
        return(self.is_ping);    
                                                                                                            
        
    #---------------------------------------------------------------------------
    # opcCollectorBR_PLC - opc server BR
    #---------------------------------------------------------------------------
    def opcCollectorBR_PLC(self):
        
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

        if not self.pingSocket(self.host1, self.port1):
            plc_isRunning = False;
            return(plc_br_table, plc_isRunning);
   
        client = Client(self.uri1)
        try:        
            client.connect();
            self.logger.info("Client: " + self.uri1 + " connect.....")
            plc_br_table = np.c_[plc_br_table, np.zeros(len(plc_br_table))];
            
            for i in range(len(plc_br_table)):
                node = client.get_node(str(plc_br_table[i, 1]));
                typ  = type(node.get_value());
                val = float(self.myFloatFormat(node.get_value())) if typ is float else node.get_value();
                #val = node.get_value() if typ is float else node.get_value();
                plc_br_table[i, 2] = val;

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect();
            return(plc_br_table, plc_isRunning);
 


    #---------------------------------------------------------------------------
    # opcCollectorHH_PLC - opc server PLC HEIDENHAIN
    #---------------------------------------------------------------------------
    def opcCollectorHH_PLC(self):
        
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
                                        
                                        
        if not self.pingSocket(self.host2, self.port2):
            plc_isRunning = False;
            return(plc_hh_table, plc_isRunning);
                                                    
        client = Client(self.uri2);
        try:        
            client.connect();
            self.logger.info("Client: " + self.uri2+ " connect.....")
            
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
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        except Exception as ex:
            traceback.print_exc();
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
    #---------------------------------------------------------------------------
    def opcCollectorSendToPLC(self, df_plc):
        
        plc_isRunning = True;
        
        uri = self.prefix+self.host2+":"+self.port2;
        plcData = self.PLCData;

        
        if not self.pingSocket(self.host2, self.port2):
            plc_isRunning = False;
            return plc_isRunning;

        if not self.pingSocket(self.host1, self.port1):
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
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            return plc_isRunning;
            
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            return plc_isRunning;
        
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
    # opcCollectorGetPLCdata - nacti PLC HEIDENHAIN + PLC BR
    # zapis nactena data do br_data - rozsireni treninkove mnoziny 
    # o data z minulosti
    #---------------------------------------------------------------------------
    def opcCollectorGetPlcData(self, df_parms):

        plc_isRunning = True;
        if not self.pingSocket(self.host2, self.port2):
            plc_isRunning = False;
            return(None);

        if not self.pingSocket(self.host1, self.port1):
            plc_isRunning = False;
            return(None);

        # zapis dat z OPC pro rozsireni treninkove mnoziny        
        current_day =  datetime.now().strftime("%Y-%m-%d");
        path_to_df = "./br_data/tm-ai_"+current_day+".csv";
        
        
        self.logger.info("Nacitam "+str(self.batch)+" vzorku dat pro predict - v intervalu: " +str(self.plc_timer)+ "[s]");
        for i in range(self.batch):
            #self.logger.info("."+str(i)+"\r");
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
            time.sleep(self.plc_timer);
            
        # add jitter 
        df_predict = self.setJitter(df_predict, df_parms, True);    
        # zapis pristi treninkova data
        if exists(path_to_df):
            self.logger.info("Nacteno "+str(self.batch)+" vzorku dat pro predict, pripisuji k:"+path_to_df+ "");
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="a", index=False, header=False);
        else:    
            self.logger.info("Nacteno "+str(self.batch)+" vzorku dat pro predict, zapisuji do:"+path_to_df+ "");
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="w", index=False, header=True);
            
        return(df_predict);
    
    #---------------------------------------------------------------------------
    # opcCollectorGetDebugData - totez co opcCollectorGetPLCdata ovsem
    # data se nectou z OPC ale  z CSV souboru. Toto slouzi jen pro ladeni
    # abychom nebyli zavisli na aktivite OPC serveruuuuu.
    # v pripade ladeni se nezapisuji treninkova data....
    #---------------------------------------------------------------------------
    def opcCollectorGetDebugData(self, df_parms):

        plc_isRunning = True;
        
        global df_debug_count;
        global df_debug_header;
        df_predict = pd.DataFrame();
        current_day =  datetime.now().strftime("%Y-%m-%d");
        csv_file = "./br_data/predict-debug.csv";
        
        try:
            df_predict = pd.read_csv(csv_file,
                                     sep=",|;", 
                                     engine='python',  
                                     header=0, 
                                     encoding="utf-8",
                                     skiprows=df_debug_count,
                                     nrows=self.batch
                        );
        except  EmptyDataError as ex:
            return None;
                                       
        
        #df_predict = self.df_debug[self.df_debug_count : self.df_debug_count + self.batch];
        
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
        #df_predict.to_csv("./result/temp"+current_date+".csv");
            
        return(df_predict);
    
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
        plc_isRunning = True;


        if not self.pingSocket(self.host1, self.port1):
            self.logger.debug("PING: False....: %s "%self.host1);
            plc_isRunning = False;
            return(plc_isRunning);

        if not self.pingSocket(self.host2, self.port2):
            self.logger.debug("PING: False....: %s "%self.host2);
            plc_isRunning = False;
            return(plc_isRunning);
        
        return (plc_isRunning);
    


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

    def __init__(self, path_to_result, window, debug_mode, batch, current_date):

        self.logger    = logging.getLogger('root');

        
    #Vystupni list parametru - co budeme chtit po siti predikovat
        self.df_parmx = ['temp_S1',
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
        self.opc = OPCAgent(batch=self.batch);

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
                self.logger.debug("--shuffle = True");
            
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

    #---------------------------------------------------------------------------
    # interpolateDF
    # interpoluje data splinem - vyhlazeni schodu na merenych artefaktech
    #---------------------------------------------------------------------------
    def interpolateDF(self, df, smoothing_factor, ip_yesno):

        if not ip_yesno:
            self.logger.debug("interpolace artefaktu nebude provedena ip = False");
            return df;
        else:
            self.logger.info("interpolace artefaktu, smoothing_factor:" + str(smoothing_factor)+"");
        
        col_names = list(self.df_parmX);
        x = np.arange(0,len(df));

        for i in range(len(col_names)):
            if "dev" in col_names[i]:
                spl =  UnivariateSpline(x, df[col_names[i]], s=smoothing_factor);
                df[col_names[i]] = spl(x);

        return df;

    
    #---------------------------------------------------------------------------
    # getData
    #---------------------------------------------------------------------------
    def getData(self, shuffling=False, 
                timestamp_start='2022-06-29 05:00:00', 
                timestamp_stop='2022-07-01 23:59:59', 
                type="predict"):
        
        txdt_b  = False;
        df      = pd.DataFrame(columns = self.df_parmX);
        df_test = pd.DataFrame(columns = self.df_parmx);
        
        size_train = 0;
        size_valid = 0;
        size_test  = 0;
        
        try:
           
            #self.DataTrainDim.DataTrain = None;
            
            # nacti data pro trenink
            if "train" in type: 
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
                    self.logger.error("Data pro trenink maji nulovou velikost - exit(1)");
                    os._exit(1);
                
            
                df["index"] = pd.Index(range(0, len(df), 1));
                df.set_index("index", inplace=True);

                size = len(df)
                size_train = math.floor(size * 8 / 12);
                size_valid = math.floor(size * 4 / 12);
                size_test  = math.floor(size * 0 / 12);

            # nacti data pro predict
            self.logger.info("Data pro predikci....");
            if self.debug_mode is True:
                df_test = self.opc.opcCollectorGetDebugData(self.df_parmX);
            else:
                self.logger.info("Data pro predikci getPlcData....");
                df_test = self.opc.opcCollectorGetPlcData(self.df_parmX);
                
            if df_test is None:
                self.logger.error("Nebyla nactena zadna data pro predikci");
                
            elif len(df_test) == 0:
                self.logger.error("Patrne nebezi nektery OPC server  ");
                       
            elif  len(df_test) > 0 and self.window >= len(df_test):
                self.logger.error("Prilis maly vzorek dat pro predikci");
                    
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
            self.logger.error(traceback.print_exc());
            
    #-----------------------------------------------------------------------
    # saveDataToPLC  - result
    # index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
    #
    #         return True - nacti dalsi porci dat pro predict
    #         return False - nenacitej dalsi porci dat pro predict.
    #-----------------------------------------------------------------------
    def saveDataToPLC(self, threads_result, timestamp_start, thread_name, typ, saveresult=True):

        
        thread_name = thread_name[0 : len(thread_name) - 1].replace("_","");
        
        col_names_y = list(self.DataTrain.df_parm_y);
        filename = "./result/plc_archiv/plc_"+thread_name+"_"+str(self.current_date)[0:10]+".csv"
        
        df_result = pd.DataFrame(columns = col_names_y);
        df_append = pd.DataFrame();
        
        i = 0
        frames = [];
        #precti vysledky vsech threadu...
        for result in threads_result:
            i += 1;
            if result[0] is None:
                return False;   # nenacitej novou porci dat, thready neukoncily cinnost....
            else:
                df = result[0];
                if df is not None:
                    df["index"] =  df.reset_index().index;
                    frames.append(df);

        df_result = pd.concat(frames); 
        df_result = df_result.groupby("index").mean();
        df_plc = self.formatToPLC(df_result, col_names_y);
                                      
        path = Path(filename)
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            self.logger.info(f'Soubor {filename} existuje - append');
            df_plc.to_csv(filename, mode = "a", index=False, header=False, float_format='%.5f');
        else:
            self.logger.info(f'Soubor {filename} neexistuje - insert');
            df_plc.to_csv(filename, mode = "w", index=False, header=True, float_format='%.5f');

        # data do PLC - debug mode -> disable...
        if self.debug_mode is False:
            result_opc = self.opc.opcCollectorSendToPLC(df_plc=df_plc );
            if result_opc:
                self.logger.warning("Data do PLC byla zapsana !!!!!!");
            else:    
                self.logger.error("Data do PLC nebyla zapsana !!!!!!");
        
        # data ke zkoumani zapisujeme v pripade behu typu "train" a zaroven v debug modu
        if "train" in typ and self.debug_mode is True:
            saveresult=True;
        else:
            saveresult=True;
            
        if saveresult:
            self.logger.debug("Vystupni soubor " + filename + " vznikne.");
            self.saveDataResult(timestamp_start, df_result, thread_name , typ, saveresult);
        else:
            self.logger.debug("Vystupni soubor " + filename + " nevznikne !!!, saveresult = " +str(saveresult) +"");
        
        #vynuluj 
        for result in threads_result:
            result[0] = None;
        
        return True; #nacti novou porci dat    
        
     
    #-----------------------------------------------------------------------
    # formatToPLC  - result
    #-----------------------------------------------------------------------
    def formatToPLC(self, df_result, col_names_y):
        #curent timestamp UTC
        current_time = time.time()
        utc_timestamp = datetime.utcfromtimestamp(current_time);

        l_plc = [];        
        l_plc_col = [];        
        
        l_plc.append( str(utc_timestamp)[0:19]);
        l_plc_col.append("utc");

        for col in col_names_y:
            if "dev" in col:
                mmean = self.myIntFormat(df_result[col].mean() *10000);   #prevod pro PLC (viz dokument Teplotni Kompenzace AI)
                l_plc.append(mmean);                                      #  10 = 0.001 atd...
                l_plc_col.append(col+"mean");
                
        return (pd.DataFrame([l_plc], columns=[l_plc_col]));
        
            
        
        
    #-----------------------------------------------------------------------
    # saveDataResult  - result
    # index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
    #-----------------------------------------------------------------------
    def saveDataResult(self, timestamp_start, df_result,  model, typ, saveresult=True):

        filename = "./result/predicted_"+str(self.current_date)[0:10]+"_"+model+".csv"
        
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
            
            df_result.drop(col_names_drop, inplace=True, axis=1);
            df_result  = pd.DataFrame(np.array(df_result), columns =col_names_predict);
            
            df_result2 = pd.DataFrame();
            df_result2 = pd.DataFrame(self.DataTrain.test);

            #merge1 - left inner join
            df_result  = pd.concat([df_result.reset_index(drop=True),
                                    df_result2.reset_index(drop=True)],
                                    axis=1);
            
            #df_result.to_csv("a.csv");
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

            df_result = self.interpolateDF(df_result, 0.01, False);            
                
        
            if append:             
                self.logger.debug(f"Soubor {filename} existuje - append: " + str(len(df_result))+" bajtu...");
                df_result.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
            else:
                self.logger.debug(f"Soubor {filename} neexistuje - insert: " + str(len(df_result))+" bajtu...");
                df_result.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');
                
            self.saveParmsMAE(df_result, model)    

        except Exception as ex:
            traceback.print_exc();
        
        return;    

    #-----------------------------------------------------------------------
    # saveParmsMAE - zapise hodnoty MAE v zavislosti na pouzitych parametrech
    #-----------------------------------------------------------------------
    def saveParmsMAE(self, df,  model):

        filename = "./result/parms_mae_"+str(self.current_date)[0:10]+"_"+model+".csv"
        
        
        local_parms = [];   
        local_header = [];     
        #pridej maximalni hodnotu AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    local_header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    local_parms.append(float(res));        
                else:
                    local_header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    local_parms.append(float(res));        
        
        #pridej mean AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    local_header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    local_parms.append(float(res));        
                else:
                    local_header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    local_parms.append(float(res));

        local_parms = self.parms + local_parms;
        local_header = self.header + local_header;
        
        df_ae = pd.DataFrame(data=[local_parms], columns=local_header);
        
        path = Path(filename)
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            self.logger.info(f'Soubor {filename} existuje - append');
            df_ae.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
        else:
            self.logger.info(f'Soubor {filename} neexistuje - insert');
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
                line = line.replace("","").replace("'","").replace(" ","");
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
            self.logger.info("parametry nacteny z "+ parmfile +"");       
            
                
        except:
            self.logger.info("Soubor parametru "+ parmfile + " nenalezen!");                
            self.logger.info("Parametry pro trenink site budou nastaveny implicitne v programu");                 
        
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
            self.logger.error("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru ");
            return(None);
        
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

    def getCurrentDate(self):
        return(self.current_date);
    
    def setCurrentDate(self, val):
        self.current_date = val;
        
    #---------------------------------------------------------------------------
    # isPing         
    #---------------------------------------------------------------------------
    def isPing(self):
        is_ping = self.opc.isPing();
        return(is_ping);
    
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
                 debug_mode,
                 current_date="",
                 thread_name=""):

        self.logger    = logging.getLogger('root');
        
        
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
        self.debug_mode = debug_mode;
        self.current_date=current_date,
        self.data           = None;
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;
        self.neural_model   = None;
        

    #---------------------------------------------------------------------------
    # Neuronova Vrstava LSTM
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMtrain(self, DataTrain, thread_name):
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
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler"+thread_name+".pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler"+thread_name+".pkl", 'wb'))
        
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
            neural_model = Sequential();
            neural_model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            neural_model.add(LSTM(units = self.units, return_sequences=True));
            neural_model.add(Dropout(0.2));
            neural_model.add(LSTM(units = self.units, return_sequences=True));
        #   neural_model.add(Dropout(0.2));
        #   neural_model.add(LSTM(units = self.units, return_sequences=True));
            neural_model.add(layers.Dense(Y_train.cols, activation='relu'));

        # definice ztratove funkce a optimalizacniho algoritmu
            neural_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc']);
        # natrenuj neural_model na vstupni dataset
            history = neural_model.fit(X_train.X_dataset, 
                                       Y_train.X_dataset, 
                                       epochs=self.epochs, 
                                       batch_size=self.batch, 
                                       verbose=2, 
                                       validation_data=(X_valid.X_dataset,
                                                        Y_valid.X_dataset)
                            );

            
        # zapis neural_modelu    
            neural_model.save("./models/model_"+thread_name+"_"+DataTrain.axis, overwrite=True, include_optimizer=True)

        # make predictions for the input data
            self.neural_model = neural_model;
            return ();
        
        except Exception as ex:
            traceback.print_exc();
            self.logger.info(traceback.print_exc());
        
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava LSTM predict 
    #---------------------------------------------------------------------------
    def neuralNetworkLSTMpredict(self, DataTrain, thread_name):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);

            self.x_train_scaler =  pickle.load(open("./temp/x_valid_scaler"+thread_name+".pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_valid_scaler"+thread_name+".pkl", 'rb'))
            
            x_test        = self.x_train_scaler.transform(x_test);
            
            x_object      = DataFactory.toTensorLSTM(x_test, window=self.window);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = self.neural_model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
        # vezmi (y_result.shape[1] - 1) - posledni ramec vysledku - nejlepsi mse i mae
            y_result      = y_result[0 : (y_result.shape[0] - 1),  (y_result.shape[1] - 1) , 0 : y_result.shape[2]];
            y_result      = self.y_train_scaler.inverse_transform(y_result);
            
        # plot grafu compare...
            #model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            self.logger.error("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            self.logger.error("          zkuste nejdrive --typ == train !!!");
            self.logger.error(traceback.print_exc());




    #---------------------------------------------------------------------------
    # neuralNetworkLSTMexec_x 
    #---------------------------------------------------------------------------

    def neuralNetworkLSTMexec(self, threads_result, threads_cnt):

        thread_name = "";
        try:
            
            thread_name = threads_result[threads_cnt][1];
            startTime = datetime.now();

            #nacti pouze predikcni data 
            self.logger.info("Nacitam data pro predikci, typ: "+ str(self.typ));
            self.data.Data = self.data.getData(shuffling=self.shuffling, 
                                               timestamp_start=self.txdat1, 
                                               timestamp_stop=self.txdat2,
                                               type=self.typ);


            if self.typ == 'train':
                self.logger.info("Start: "+ thread_name+" vcetne treninku, model bude zapsan");
                self.neuralNetworkLSTMtrain(self.data.DataTrainDim.DataTrain, thread_name);
            else:    
                self.logger.info("Start: "+ thread_name+"  bez treninku...");
            
            if self.data.DataTrainDim.DataTrain.test is None or len(self.data.DataTrainDim.DataTrain.test) == 0: 
                self.logger.info("Data : "+ thread_name+ " pro predikci nejsou k dispozici....");
                
                if self.debug_mode is True:
                    self.logger.info("Exit...");
                    sys.exit(0);
                else:    
                    return();
           
            
            self.data.DataResultDim.DataResultX = self.neuralNetworkLSTMpredict(self.data.DataTrainDim.DataTrain, thread_name);
            col_names_y = list(self.data.DataTrain.df_parm_y);
            threads_result[threads_cnt][0]  =  pd.DataFrame(self.data.DataResultDim.DataResultX.y_result, columns = col_names_y);
            
            if self.data.saveDataToPLC(threads_result, self.txdat1, thread_name, self.typ):
                pass;

            
            stopTime = datetime.now();
            self.logger.info( thread_name+ ": cas vypoctu[s] " + str(stopTime - startTime) + "");

            return();

        except FileNotFoundError as e:
            self.logger.info(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!");    
        except Exception as ex:
            self.logger.info(str(traceback.print_exc()));
            traceback.print_exc();
            
    #---------------------------------------------------------------------------
    # setter - getter
    #---------------------------------------------------------------------------
    def getModel(self):
        return self.model_;
    
    def getData(self):
        return self.data;

    def setData(self, data):
        self.data = data;

    def getTyp(self):
        return self.typ;

    def setTyp(self, typ):
        self.typ = typ;

    def getTxdat1(self):
        return(self.txdat1);
    
    def setTxdat1(self, val):
        self.txdat1 = val;
        
    def getTxdat2(self):
        return(self.txdat2);
    
    def setTxdat2(self, val):
        self.txdat2 = val;

    def getCurrentDate(self):
        return(self.current_date);
    
    def setCurrentDate(self, val):
        self.current_date = val;
        
    #---------------------------------------------------------------------------
    # isPing ????
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.data.isPing();


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
                 debug_mode,
                 current_date="",
                 thread_name=""):

        self.logger    = logging.getLogger('root');
        
        
        self.path_to_result = path_to_result; 
        self.typ    = typ;
        self.model_ = model;
        self.epochs = epochs;
        self.batch  = batch;
        self.txdat1 = txdat1;
        self.txdat2 = txdat2;
        self.debug_mode = debug_mode;
        self.current_date=current_date;
        self.thread_name    = thread_name;

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
        self.neural_model   = None;

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
    def neuralNetworkDENSEtrain(self, DataTrain, thread_name):

        try:
            
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);

            if (x_train_data.size == 0 or y_train_data.size == 0):
                return();
            
            inp_size = len(x_train_data[0])
            out_size = len(y_train_data[0])
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler"+thread_name+".pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler"+thread_name+".pkl", 'wb'))
            
            
        # neuronova sit
            neural_model = Sequential();
            initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            neural_model.add(tf.keras.Input(shape=(inp_size,)));
            neural_model.add(layers.Dense(units=inp_size,       activation=self.actf, kernel_initializer=initializer));
            neural_model.add(layers.Dense(units=self.units,     activation=self.actf, kernel_initializer=initializer));
        #   neural_model.add(Dropout(0.2));
            neural_model.add(layers.Dense(units=self.units,     activation=self.actf, kernel_initializer=initializer));
        #   neural_model.add(Dropout(0.2));
            neural_model.add(layers.Dense(out_size));
            
        # definice ztratove funkce a optimalizacniho algoritmu
            neural_model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])
            
        # natrenuj neural_model na vstupni dataset
            history = neural_model.fit(x_train_data, 
                                       y_train_data, 
                                       epochs=self.epochs, 
                                       batch_size=self.batch, 
                                       verbose=2, 
                                       validation_data=(x_valid_data, y_valid_data)
                            );
        
            neural_model.save("./models/model_"+thread_name+"_"+DataTrain.axis, overwrite=True, include_optimizer=True)
            
            self.neural_model = neural_model;
        
        # make predictions for the input data
            return ();
    
            
        except Exception as ex:
            traceback.print_exc();
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
    def neuralNetworkDENSEpredict(self, DataTrain, thread_name):
        
        #dff = pd.DataFrame();
        #dff = DataTrain.test;
        try:
            axis     = DataTrain.axis;
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_valid_scaler"+thread_name+".pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_valid_scaler"+thread_name+".pkl", 'rb'))
            
        # normalizace vstupnich a vystupnich testovacich dat 
            x_test   =  self.x_train_scaler.transform(x_test);
        # predikce site
            y_result = self.neural_model.predict(x_test);
        # zapis syrove predikce ke zkoumani    
            y_result = self.y_train_scaler.inverse_transform(y_result);
            
            columns  =DataTrain.df_parm_y
            dfy      = pd.DataFrame();
            dfy      = pd.DataFrame(y_result, columns=columns);
            
            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            self.logger.warning("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            self.logger.warning("          zkuste nejdrive --typ == train !!!");
            self.logger.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec(self, threads_result, threads_cnt):

        thread_name = "";
        try:
            
            thread_name = threads_result[threads_cnt][1];
            
            startTime = datetime.now();
            self.logger.info("Nacitam data pro predikci, typ: "+ str(self.typ));
            self.data.Data = self.data.getData(shuffling=self.shuffling, 
                                               timestamp_start=self.txdat1, 
                                               timestamp_stop=self.txdat2,
                                               type=self.typ);
            
            if self.typ == 'train':
                self.logger.info("Start: "+ thread_name+" vcetne treninku, model bude zapsan");
                self.neuralNetworkDENSEtrain(self.data.DataTrainDim.DataTrain, thread_name);
            else:    
                self.logger.info("Start threadu: "+ thread_name+"  bez treninku...");
                #self.model_ = load_model("./models/model_"+thread_name+"_"+ data.DataTrainDim.DataTrain.axis);
                
            if self.data.DataTrainDim.DataTrain.test is None or len(self.data.DataTrainDim.DataTrain.test) == 0:
                self.logger.error("Data threadu : "+ thread_name+ " pro predikci nejsou k dispozici....");

                if self.debug_mode is True:
                    self.logger.info("Exit...");
                    sys.exit(0);
                else:    
                    return();
           
            
            self.data.DataResultDim.DataResultX = self.neuralNetworkDENSEpredict(self.data.DataTrainDim.DataTrain, thread_name);
            col_names_y = list(self.data.DataTrain.df_parm_y);
            threads_result[threads_cnt][0]  =  pd.DataFrame(self.data.DataResultDim.DataResultX.y_result, columns = col_names_y);
            if self.data.saveDataToPLC(threads_result, self.txdat1, thread_name, self.typ):
                pass;
            
            stopTime = datetime.now();
            self.logger.debug( thread_name+ ": cas vypoctu[s] " + str(stopTime - startTime) + "");

            return();

        except FileNotFoundError as e:
            self.logger.error(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!");    
        except Exception as ex:
            self.logger.error(traceback.print_exc());
            traceback.print_exc();
            

    #---------------------------------------------------------------------------
    # setter - getter
    #---------------------------------------------------------------------------
    def getModel(self):
        return self.model_;
    
    def getData(self):
        return self.data;

    def setData(self, data):
        self.data = data;

    def getTyp(self):
        return self.typ;

    def setTyp(self, typ):
        self.typ = typ;
        
    def getTxdat1(self):
        return(self.txdat1);
    
    def setTxdat1(self, val):
        self.txdat1 = val;
        
    def getTxdat2(self):
        return(self.txdat2);
    
    def setTxdat2(self, val):
        self.txdat2 = val;

    def getCurrentDate(self):
        return(self.current_date);
    
    def setCurrentDate(self, val):
        self.current_date = val;
        
    #---------------------------------------------------------------------------
    # isPing ????
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.data.isPing();

#------------------------------------------------------------------------
# Daemon    
#------------------------------------------------------------------------
class NeuroDaemon():
    
    def __init__(self, 
                 pidf, 
                 path_to_result, 
                 model,
                 epochs, 
                 batch, 
                 units, 
                 shuffling, 
                 txdat1, 
                 txdat2, 
                 actf, 
                 window, 
                 debug_mode,
                 current_date,
                 max_threads
            ):

        self.logger    = logging.getLogger('root');
        
        
        self.pidf           = pidf; 
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
        self.typ            = "train";
        
        self.train_counter  = 0;
        self.data           = None;
        self.threads_result = [];
        self.max_threads    = max_threads;
        
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def getLogFileHandles(self):
        """ Get a list of filehandle numbers from logger
            to be handed to DaemonContext.files_preserve
        """
        handles = []
        for handler in self.logger.handlers:
            handles.append(handler.stream.fileno())
            if self.logger.parent:
                handles += self.getLogFileHandles(self.logger.parent)
                
        return (handles);




#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def setLogHandler(self):

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
        

#------------------------------------------------------------------------
# mailto....
#------------------------------------------------------------------------
    def mailtoMSG(self, current_time, recipient, txdat1, txdat2, plc_isRunning):
        
        
        subject = "ai-daemon";
        if plc_isRunning:
            msg0 = "\nai-daemon v rezimu run....\n";
            msg1 = "start v rezimu train, cas:"+str(current_time)+"\n";
            msg2 = "treninkova mnozina, timestamp start :"+txdat1+"\n";
            msg3 = "                    timestamp stop  :"+txdat2+"\n";
        else:    
            msg0 = "\nai-daemon v rezimu sleep (stroj je vypnut)\n";
            msg1 = "start v rezimu train, cas:"+str(current_time)+"\n";
            msg2 = "treninkova mnozina, timestamp start :"+txdat1+"\n";
            msg3 = "                    timestamp stop  :"+txdat2+"\n";

        msg = msg0+msg1+msg2+msg3 ;
        self.logger.info(msg);
        #webbrowser.open("mailto:?to="+ recipient + "&subject=" + subject + "&body=" + msg, new=1);

#------------------------------------------------------------------------
# start daemon pro parametr LSTM
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemonLSTM(self, threads_result, threads_cnt):

        current_day  = datetime.today().strftime('%A');   #den v tydnu
        # new LSTM layer
        thread_name = threads_result[threads_cnt][1];
        epochs = self.epochs;
        units  = self.units;
        timestamp_format = "%Y-%m-%d %H:%M:%S";
        plc_isRunning = True;
        
        if threads_cnt > 0:
            epochs = epochs + 20;
            units  = units  + 61;
        
        neural = NeuronLayerLSTM(path_to_result = path_to_result, 
                                  typ            = "train", 
                                  model          = self.model_, 
                                  epochs         = epochs, 
                                  batch          = self.batch,
                                  txdat1         = self.txdat1,
                                  txdat2         = self.txdat2,
                                  window         = self.window,
                                  units          = units,
                                  shuffling      = self.shuffling,
                                  actf           = self.actf, 
                                  debug_mode     = self.debug_mode,
                                  current_date   = self.current_date,
                                  thread_name    = thread_name
                            );
                            
        neural.setData(self.data);
        plc_isRunning = self.data.isPing();

        if plc_isRunning:
            sleep_interval =   1;             #1 [s]
        else:
            if self.debug_mode:
                sleep_interval =   1;         #1 [s]   
            else:
                sleep_interval = 600;         #600 [s]

        current_date =  datetime.now().strftime(timestamp_format);
        current_day  = datetime.today().strftime('%A');   #den v tydnu
        given_time = datetime.strptime(current_date, timestamp_format);
        
        final_time_day1 = given_time - timedelta(days=1);
        final_time_day1_str = final_time_day1.strftime(timestamp_format);

        final_time_day30 = given_time - timedelta(days=30);
        final_time_day30_str = final_time_day30.strftime(timestamp_format);

        #jsou nabity timestampy pro vyber mnoziny trenikovych dat?
        if neural.getTxdat1() == "":
            neural.setTxdat1(final_time_day30_str);
            self.txdat2 = final_time_day30_str;
            
        if neural.getTxdat2() == "":
            neural.setTxdat2(final_time_day1_str);
            self.txdat2 = final_time_day1_str;

        self.current_date = current_date;
        neural.setCurrentDate(current_date);
        self.data.setCurrentDate(current_date);
            
        txdat1 = neural.getTxdat1();
        txdat2 = neural.getTxdat2();

        self.mailtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);
            
        #predikcni beh
        while True:
            
            current_date =  datetime.now().strftime(timestamp_format);
            plc_isRunning = self.data.isPing();

            
            #prechod na novy den - pretrenuj sit...
            if (datetime.today().strftime('%A') not in current_day):
                self.logger.info("start train:"+ current_date);
                neural.setTyp("train");
                
                current_day  = datetime.today().strftime('%A');   #den v tydnu
                given_time = datetime.strptime(current_date, timestamp_format);
                final_time = given_time - timedelta(days=1);
                final_time_str = final_time.strftime(timestamp_format);
                txdat1 = neural.getTxdat1();
                txdat2 = neural.getTxdat2();
                
                self.txdat2 = final_time_str;
                neural.setTxdat2(final_time_str);
                self.current_date = current_date;
                neural.setCurrentDate(current_date);
                self.data.setCurrentDate(current_date);

                
                self.logger.info("Nacitam data pro predikci");
                self.mailtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);

                
                
            if plc_isRunning:
                if neural.getTyp() == "train":
                    self.logger.info("PLC ON:"+ current_date + " rezim train, pocet aktivnich threadu: " + str(threads_cnt + 1)+"");
                else:
                    self.logger.info("PLC ON:"+ current_date + " rezim predict, pocet aktivnich threadu: " + str(threads_cnt + 1)+"");
                    
            
                neural.neuralNetworkLSTMexec(threads_result, threads_cnt);
                neural.setTyp("predict");
                sleep_interval =  1;          #  1 sekunda

            else:
                self.logger.info("PLC OFF:"+ current_date+"");
                neural.setTyp("train");
                sleep_interval =  600;        # 10 minut
                
                if self.debug_mode:
                    neural.neuralNetworkDENSEexec(threads_result, threads_cnt);
                    neural.setTyp("predict");
                    sleep_interval =  1;      # 1 sekunda
                
                    
            time.sleep(sleep_interval);
        
        
#------------------------------------------------------------------------
# start daemon pro parametr DENSE
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemonDENSE(self, threads_result, threads_cnt):


        i_cnt = 0;

        # new Dense layer
        thread_name = threads_result[threads_cnt][1];
        epochs = self.epochs;
        units  = self.units;
        timestamp_format = "%Y-%m-%d %H:%M:%S";
        plc_isRunning = True;
        
        if threads_cnt > 0:
            epochs = epochs + 20;
            units  = units  + 61;

        neural = NeuronLayerDENSE(path_to_result = path_to_result, 
                                  typ            = "train", 
                                  model          = self.model_, 
                                  epochs         = epochs, 
                                  batch          = self.batch,
                                  txdat1         = self.txdat1,
                                  txdat2         = self.txdat2,
                                  window         = self.window,
                                  units          = units,
                                  shuffling      = self.shuffling,
                                  actf           = self.actf, 
                                  debug_mode     = self.debug_mode,
                                  current_date   = self.current_date,
                                  thread_name    = thread_name
                            );
        
        neural.setData(self.data);
        
        plc_isRunning = self.data.isPing();
        
        if plc_isRunning:
            sleep_interval =   1;             #1 [s]
        else:
            if self.debug_mode:
                sleep_interval =   1;         #1 [s]   
            else:
                sleep_interval = 600;         #600 [s]
                

        current_date = datetime.now().strftime(timestamp_format);
        current_day  = datetime.today().strftime('%A');
        given_time   = datetime.strptime(current_date, timestamp_format);
        
        final_time_day1 = given_time - timedelta(days=1);
        final_time_day1_str = final_time_day1.strftime(timestamp_format);

        final_time_day30 = given_time - timedelta(days=30);
        final_time_day30_str = final_time_day30.strftime(timestamp_format);

        #jsou nabity timestampy pro vyber mnoziny trenikovych dat?
        if neural.getTxdat1() == "":
            neural.setTxdat1(final_time_day30_str);
            self.txdat2 = final_time_day30_str;
            
        if neural.getTxdat2() == "":
            neural.setTxdat2(final_time_day1_str);
            self.txdat2 = final_time_day1_str;

        self.current_date = current_date;
        neural.setCurrentDate(current_date);
        self.data.setCurrentDate(current_date);
            
        txdat1 = neural.getTxdat1();
        txdat2 = neural.getTxdat2();

        self.mailtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);
        
        #predikcni beh
        while True:
            
            current_date =  datetime.now().strftime(timestamp_format);
            plc_isRunning = self.data.isPing();
            
            #prechod na novy den - pretrenuj sit...
            if (datetime.today().strftime('%A') not in current_day):
                self.logger.info("start train:"+ current_date+"");
                neural.setTyp("train");
                
                current_day = datetime.today().strftime('%A');
                given_time = datetime.strptime(current_date, timestamp_format);
                final_time = given_time - timedelta(days=1);
                final_time_str = final_time.strftime(timestamp_format);
                txdat1 = neural.getTxdat1();
                txdat2 = neural.getTxdat2();
                
                self.txdat2 = final_time_str;
                neural.setTxdat2(final_time_str);
                self.current_date = current_date;
                neural.setCurrentDate(current_date);
                self.data.setCurrentDate(current_date);

                self.mailtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);

            if plc_isRunning:
                if neural.getTyp() == "train":
                    self.logger.debug("PLC ON:"+ current_date + " rezim train, pocet aktivnich threadu: " + str(threads_cnt + 1)+"");
                else:
                    self.logger.debug("PLC ON:"+ current_date + " rezim predict, pocet aktivnich threadu: " + str(threads_cnt +1)+"");
                    
                neural.neuralNetworkDENSEexec(threads_result, threads_cnt);
                neural.setTyp("predict");
                sleep_interval =   1;         # 1 sekunda

            else:
                self.logger.info("PLC OFF:"+ current_date+"");
                neural.setTyp("train");
                sleep_interval = 600;         # 10 minut

                if self.debug_mode:
                    neural.neuralNetworkDENSEexec(threads_result, threads_cnt);
                    neural.setTyp("predict");
                    sleep_interval = 1;       # 1 sekunda
                
                
            time.sleep(sleep_interval);
        
#------------------------------------------------------------------------
# start daemon pro parametr DENSE
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemon(self, threads_max_cnt):


        threads_cnt = 0;
        threads_result = [];
        data = None;

        try:
            self.logger.info("Pocet GPU jader: "+ str(len(tf.config.experimental.list_physical_devices('GPU')))+"")
            self.data = DataFactory(path_to_result=self.path_to_result, 
                                    window=self.window,
                                    batch=self.batch,
                                    debug_mode=self.debug_mode,
                                    current_date=self.current_date);
        
                
            parms = ["train",      #self.typ <train, predict>
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
            
            #nacti data pro trenink a predict
            #self.data.Data = self.data.getData(shuffling=self.shuffling, 
            #                         timestamp_start=self.txdat1, 
            #                         timestamp_stop=self.txdat2,
            #                         type=self.typ);
            
            #if self.data.getDfTrainData().empty and "predict" in self.typ:
            #    self.logger.info("Data pro predict, nejsou k dispozici");
            #    return(0);

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            
        
        self.logger.info("Thread executor start, pocet threadu :"+str(threads_max_cnt)+".....");
        try:
            if "DENSE" in self.model_:
                # thread names
                for i in range(threads_max_cnt):
                    #prepare result array
                    arr = [None,"_"+self.model_+"_"+str(i)];
                    threads_result.append(arr);
                
                executor = ThreadPoolExecutor(max_workers = threads_max_cnt);
                futures = [executor.submit(self.runDaemonDENSE, threads_result, threads_cnt ) for threads_cnt in range(threads_max_cnt)];
            if "LSTM" in self.model_:
                # thread names
                for i in range(threads_max_cnt):
                    #prepare result array
                    arr = [None,"_"+self.model_+"_"+str(i)];
                    threads_result.append(arr);
                executor = ThreadPoolExecutor(max_workers = threads_max_cnt);
                futures = [executor.submit(self.runDaemonLSTM, threads_result, threads_cnt ) for threads_cnt in range(threads_max_cnt)];
        except Exception as exc:
                traceback.print_exc();
                self.logger.error(traceback.print_exc());
                
    #------------------------------------------------------------------------
    # info daemon
    #------------------------------------------------------------------------
    def info(self):
        self.logger.info("daemon pro sledovani a kompenzaci teplotnich zmen stroje");
        return;
    #------------------------------------------------------------------------
    # daemonize...
    #    do the UNIX double-fork magic, see Stevens' "Advanced
    #    Programming in the UNIX Environment" for details (ISBN 0201563177)
    #    http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
    #
    #------------------------------------------------------------------------
    def daemonize(self):
        
        self.logger.info("daemonize.....");

        handler = self.setLogHandler();
        
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
            message = "pid procesu %d existuje!!!. Daemon patrne bezi - exit(1)";                                         
            self.logger.info(message % pid);                                                                       
            os._exit(1);                                                                                                    

        context = self.daemonize();
        
        try:
            with context:
                self.runDaemon(threads_max_cnt = self.max_threads);
                    
        except (Exception, getopt.GetoptError)  as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
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
            message = "pid procesu neexistuje!!!. Daemon patrne nebezi - exit(1)";                                         
            self.logger.info(message);                                                                       
            os._exit(1);
        else:                                                                                                        
            message = "pid procesu %d existuje!!!. Daemon %d stop....";                                         
            self.logger.info(message % pid);                                                                       
            os._exit(0);

            
    #------------------------------------------------------------------------
    # getter, setter metody
    #------------------------------------------------------------------------
    def getDebug(self):
        return self.debug_mode;
    
    def setDebug(self, debug_mode):
        self.debug_mode = debug_mode;


#------------------------------------------------------------------------
# MAIN CLASS
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# definice loggeru
#------------------------------------------------------------------------
def setLogger(logf):


    logging.config.fileConfig(fname='log.config');
    logger = logging.getLogger("ai")


    #_formatter = logging.Formatter("%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s");
    
    #logger = logging.getLogger("ai-daemon");
    #logger.setLevel(logging.DEBUG);

    #console_handler = logging.StreamHandler(sys.stdout);
    #console_handler.setFormatter(_formatter);
    #logger.addHandler(console_handler);

    #file_handler = handlers.RotatingFileHandler(logf, maxBytes=(1048576*5), backupCount=7);
    #file_handler.setFormatter(_formatter);
    #logger.addHandler(file_handler);

    return(logger);




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
        self.logger.error(traceback.print_exc());
 


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
    logger.error(exctype)
    logger.error(value)
    logger.error(traceback.extract_tb(tb))
    
#------------------------------------------------------------------------
# Signal handler
#------------------------------------------------------------------------
def signal_handler(self, signal, frame):
    #Catch Ctrl-C and Exit
    sys.exit(0);

#------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------
def help (activations):
    print("HELP:");
    print("------------------------------------------------------------------------------------------------------ ");
    print("pouziti: <nazev_programu> <arg-1> <arg-2> <arg-3>,..., <arg-n>");
    print(" ");
    print("        --help            list help ")
    print(" ");
    print(" ");
    print("        --model           model neuronove site 'DENSE', 'LSTM', 'GRU', 'BIDI'")
    print("                                 DENSE - zakladni model site - nejmene narocny na system")
    print("                                 LSTM - Narocny model rekurentni site s feedback vazbami")
    print(" ");
    print("        --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
    print("                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
    print("                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
    print("                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
    print("                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
    print(" ");
    print("        --units           pocet vypocetnich jednotek cislo v intervalu <32,1024>")
    print("                                 Pocet vypocetnich jednotek urcuje pocet neuronu zapojenych do vypoctu.")
    print("                                 Mějte prosím na paměti, že velikost units ovlivňuje"); 
    print("                                 dobu tréninku, chybu, které dosáhnete, posuny gradientu atd."); 
    print("                                 Neexistuje obecné pravidlo, jak urcit optimalni velikost parametru units.");
    print("                                 Obecne plati, ze maly pocet neuronu vede k nepresnym vysledkum a naopak");
    print("                                 velky pocet units muze zpusobit preuceni site - tedy stejny efekt jako pri");
    print("                                 nedostatecnem poctu units. Pamatujte, ze pocet units vyrazne ovlivnuje alokaci");
    print("                                 pameti. pro 1024 units je treba minimalne 32GiB u siti typu LSTM, GRU nebo BIDI.");
    print(" ");
    print("                                 Plati umera: cim vetsi units tim vetsi naroky na pamet.");
    print("                                              cim vetsi units tim pomalejsi zpracovani.");
    print(" ");
    print("        --actf            Aktivacni funkce - jen pro parametr DENSE")
    print("                                 U LSTM, GRU a BIDI se neuplatnuje.")
    print("                                 Pokud actf neni uvedan, je implicitne nastaven na 'tanh'."); 
    print("                                 U site GRU, LSTM a BIDI je implicitne nastavena na 'tanh' ");
    print(" ");
    print(" ");
    print("        --txdat1          timestamp zacatku datove mnoziny pro predict, napr '2022-04-09 08:00:00' ")
    print(" ");
    print("        --txdat2          timestamp konce   datove mnoziny pro predict, napr '2022-04-09 12:00:00' ")
    print(" ");
    print("                                 parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere");
    print("                                 se v uvahu cela mnozina dat k trenovani, to znamena:");
    print("                                 od pocatku mereni: 2022-02-15 00:00:00 ");
    print("                                 do konce   mereni: current timestamp() - 1 [den] ");
    print(" ");
    print("POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM, GRU nebo BIDI!!!");
    print("       pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise ");
    print("       natrenovany model site..");
    print("       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.");
    print("       spusti normalni beh site z jiz natrenovaneho modelu.");
    print("       Takze: budte trpelivi...");
    print(" ");
    print(" ");
    print(" ");
    print("Pokud pozadujete zmenu parametu j emozno primo v programu poeditovat tyto promenne ");
    print(" ");
    print("a nebo vyrobit soubor ai-parms.txt s touto syntaxi ");
    print("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print("  df_parmx = machinedata_m0412,teplota_pr01,x_temperature'");
    print(" ");
    print("  #Tenzor predlozeny k uceni site");
    print("  df_parmX = machinedata_m0412,teplota_pr01, x_temperature");
    print(" ");
    print("a ten nasledne ulozit v rootu aplikace. (tam kde je pythonovsky zdrojak. ");
    print("POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy vstupniho datoveho CSV souboru (nebo souboruuu)");
    print("a muzou tam byt i uvozovky: priklad: 'machinedata_m0112','machinedata_m0212', to pro snazsi copy a paste ");
    print("z datoveho CSV souboru. ");
    print(" ");
    print("(C) GNU General Public License, autor Petr Lukasik , 2022 ");
    print(" ");
    print("Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)");
    print("              miniconda3,");
    print("              python 3.9, tensorflow 2.8, mathplotlib,  ");
    print("              tensorflow 2.8,");
    print("              mathplotlib,  ");
    print("              scikit-learn-intelex,  ");
    print("              pandas,  ");
    print("              numpy,  ");
    print("              keras   ");
    print(" ");
    print(" ");
    print("Povolene aktivacni funkce: ");
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
    global g_window;
    g_window = 24;
    
    path_to_result = "./result";
    current_date ="";
    pidf = "./pid/ai-daemon.pid"
    logf = "./log/ai-daemon.log"
    
    #registrace signal handleru
    signal.signal(signal.SIGINT, signal_handler);
    
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
    epochs         = 64;
    batch          = 256;
    units          = 71;
    txdat1         = "2022-02-15 00:00:00";
    txdat2         = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S");
    shuffling      = False;
    actf           = "elu";
    pid            = 0;
    status         = "";
    startTime      = datetime.now();
    type           = "train";
    debug_mode     = False;
    max_threads    = 1; 
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
    logger = setLogger(logf);
        
    try:
        logger.info("start...");
        #logger.info("start...");

        #kontrola platne aktivacni funkce        
        if not checkActf(actf, activations):
            print("Chybna aktivacni funkce - viz help...");
            help(activations)
            sys.exit(1);
            
        logger.info("Verze TensorFlow :" + tf.__version__);
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
                    r = range(10-1, 512+1);
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
                                      current_date   = current_date,
                                      max_threads    = max_threads
                                );
       
                daemon_.info();
                
                if debug_mode:
                    logger.info("ai-daemon run v debug mode...");
                    daemon_.runDaemon(threads_max_cnt = max_threads);
                else:    
                    logger.info("ai-daemon start...");
                    daemon_.runDaemon(threads_max_cnt = max_threads);
                    #daemon_.start();
                    
            except:
                traceback.print_exc();
                logger.info(str(traceback.print_exc()));
                logger.info("ai-daemon start exception...");
                pass
            
        elif "stop" in status:
            logger.info("ai-daemon stop...");
            daemon_.stop();
            
        elif "restart" in status:
            logger.info("ai-daemon restart...");
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
                logger.info("Daemon ai-daemon je ve stavu run...");
            else:
                logger.info("Daemon ai-daemon je ve stavu stop....");
        else:
            logger.info("Neznamy parametr:<"+status+">");
            sys.exit(0)
        
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        #logger.error(traceback.print_exc());
        help(activations);
        
    finally:    
        stopTime = datetime.now();
        #print("cas vypoctu [s]", stopTime - startTime );
        logger.info("cas vypoctu [s] %s",  str(stopTime - startTime));
        logger.info("stop obsluzneho programu pro demona - ai-daemon...");
        sys.exit(0);




#------------------------------------------------------------------------
# main entry point
#------------------------------------------------------------------------
        
if __name__ == "__main__":

    main(sys.argv[1:])
    

