#!/bin/bash
FILE_PATH=$(pwd)
export AI_HOME="~/ai/ai-daemon"
#export PYTHONPATH="~/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="~/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0
eval "$(conda shell.bash hook)"
conda activate tf
#cd $AI_HOME/src

ai-help() {
    echo " Start v produkcnim modu. Program se uhnizdi v pameti  "
    echo " jako demon a zacne generovat navrhy korekci pro PLC.  "  
    echo " ./ai-start.sh nastavi hlavni hyperparametry pro       "  
    echo " neuronovou sit.                                       "  
    echo "-------------------------------------------------------"
    echo " popis parametru:                                      "  
    echo "    -s <--status>  - typ behu programu, muze nabyvat   "
    echo "                     hodnot 'start' nebo 'run'.        "
    echo "                     'start' - spusten jako demon      "
    echo "                     'run'   - spusten jako program    "
    echo "                                                       "
    echo "    -m <--mode>    - mod debug/nodebug                 "
    echo "                     pozor !!! nodebug je ostry provoz "
    echo "-------------------------------------------------------"
    echo " popis hyperparametru:                                 "  
    echo " pozor jsou implicitne nastaveny pro optimalni beh site"
    echo "    UNITS='71'     - pocet neuronu ve skryte vrstve    "
    echo "    MODEL='DENSE'  - typ neuronove site                "
    echo "                     (DENSE, LSTM, GRU, CONV1D)        "
    echo "    EPOCHS='48'    - pocet treninkovych epoch          "
    echo "    LAYERS='2'     - pocet skrytych vrstev             "
    echo "                     ostatni hyperparametry lze nasta- "
    echo "                     vit v hlavnim skriptu ai-daemon.sh"
    echo "                                                       "
    echo " pouziti: ./ai-start.sh -s=start -m=nodebug            "
    echo "                                                       "
}

echo "-------------------------------------------------------"
echo " ./ai-start.sh                                         "
echo "-------------------------------------------------------"
if [[ $# -eq 0 ]]
then
    ai-help
    exit 1
fi

for i in "$@"; do
  case $i in
    -s=*|--status=*)
        STATUS="${i#*=}"
        shift # past argument=value
        ;;
    -m=*|--mode=*)
        MODE="${i#*=}"
        shift # past argument=value
        ;;
    -*|--*)
	echo "bash: Neznamy parametr $i"
	ai-help_
	exit 1
        ;;
    *)
      ;;
  esac
done

STATUS="start"               # start|run|stop|restart|force-reload|reload|status
UNITS="71"                   # pocet neuronu
MODEL="DENSE"                # typ neuronove site
EPOCHS="48"                  # pocet treninkovych epoch
LAYERS="2"                   # pocet skrytych vrstev
BATCH="128"                  # pocet vzorku k predikci
DBMODE="nodebug"             # debug mode <debug, nodebug>
INTERPOLATE="False"          # interpolace dat splinem

./ai-daemon.sh $STATUS $UNITS $MODEL $EPOCHS $LAYERS $BATCH $DBMODE $INTERPOLATE

