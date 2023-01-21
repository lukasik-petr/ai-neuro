#!/bin/bash
#cd ~/workspaces/eclipse-python-workspace/ai-daemon/src
#cd /home/plukasik/ai/ai-daemon/src
echo "------------------------------------------------------"
echo " ai-daemon.sh "
echo "------------------------------------------------------"
#----------------------------------------------------------------------
# ai-ad     Startup skript pro ai-ad demona
# optimalni parametry
#           DENSE  |  CONV1D |   GRU |  LAYERS |
#----------------------------------------------------------------------
# UNITS    |    79 |   179   |  179  | 3       |
# EPOCHS   |    57 |    67   |   57  | 3       |
# ACTF     |   ELU |   ELU   |  ELU  | 3       |
# SHUFFLE  |  true |  true   | true  | 3       |
#----------------------------------------------------------------------
FILE_PATH=$(pwd)
# Environment Miniconda.
#export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0

prog=$FILE_PATH"/ai-daemon.py"
pidfile=${PIDFILE-$FILE_PATH/pid/ai-daemon.pid}
logfile=${LOGFILE-$FILE_PATH/log/ai-daemon.log}

STATUS="$1"
if [ -z "$STATUS" ]
then
    echo $"Usage: $prog {start|run|stop|restart|force-reload|reload|status}"
    RETVAL=2
    exit $RETVAL
fi

UNITS="$2"    # LSTM=91, DENSE=79
if [ -z "$UNITS" ]
then
      UNITS="71"
fi

MODEL="$3"    # LSTM=91, DENSE=79 # GRU CONV1D
if [ -z "$MODEL" ]
then
      MODEL="DENSE"
fi

EPOCHS="$4"    # LSTM=91, DENSE=79 # GRU CONV1D
if [ -z "$EPOCHS" ]
then
      EPOCHS="48"
fi

LAYERS=$5    # LSTM=91, DENSE=79
if [ -z "$LAYERS" ]
then
      LAYERS="2"
fi

BATCH=$6    # 64 - 2048
if [ -z "$BATCH" ]
then
      BATCH="128"
fi

DBMODE=$7    # implicitne v debug modu - nezapisuje do PLC
if [ -z "$DBMODE" ]
then
      DBMODE="debug"
fi

INTERPOLATE=$8    # TRUE FALSE -interpolace splinem
if [ -z "$INTERPOLATE" ]
then
      INTERPOLATE="False"
fi



RETVAL=0
#BATCH="1024"         #64 - 2048
ACTF="elu"          #elu , relu, sigmoid ....
TXDAT1="2022-01-01 00:00:01"
#TXDAT2="2099-23-23 23:59:59"
TXDAT2=`date +%Y-%m-%d -d "yesterday"`" 23:59:59"
OPTIONS=""
ILCNT="1"           #1 - 8
SHUFFLE="True"      #True, False

echo "bash: Spusteno s parametry:"  
echo "      STATUS="$STATUS
echo "       UNITS="$UNITS
echo "       MODEL="$MODEL
echo "      EPOCHS="$EPOCHS
echo "      LAYERS="$LAYERS
echo "       BATCH="$BATCH
echo "      DBMODE="$DBMODE
echo " INTERPOLATE="$INTERPOLATE
echo "     SHUFFLE="$SHUFFLE
echo "        ACTF="$ACTF
echo "      TXDAT1="$TXDAT1
echo "      TXDAT2="$TXDAT2

#----------------------------------------------------------------------
# start_daemon - aktivace miniconda a start demona
#----------------------------------------------------------------------
start_daemon(){
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo ""
    echo "----------------------------------------------------------------"
    echo "Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z"
    echo "Start ulohy: "$curr_timestamp
    echo "Treninkova mnozina v rozsahu: "$TXDAT1" : "$TXDAT2
    echo "----------------------------------------------------------------"
    eval "$(conda shell.bash hook)"
    conda activate tf
    python3 ./py-src/ai-daemon.py \
	    --status="$STATUS" \
	    --debug_mode="$DEBUG_MODE"\
            --pidfile="$pidfile" \
	    --logfile="$logfile" \
	    --model="$MODEL" \
	    --epochs="$EPOCHS" \
	    --batch="$BATCH" \
	    --units="$UNITS" \
	    --layers="$LAYERS" \
	    --actf="$ACTF" \
	    --txdat1="$TXDAT1" \
	    --txdat2="$TXDAT2" \
	    --ilcnt="$ILCNT" \
	    --shuffle="$SHUFFLE" \
	    --interpolate="$INTERPOLATE" 
    
    conda deactivate
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo "ai-daemon start: "$curr_timestamp

}

#----------------------------------------------------------------------
# start
#----------------------------------------------------------------------
start() {
        echo -n $"Starting $prog: as daemon... "

        if [[ -f ${pidfile} ]] ; then
            pid=$( cat $pidfile  )
            isrunning=$( ps -elf | grep  $pid | grep $prog | grep -v grep )

            if [[ -n ${isrunning} ]] ; then
                echo $"$prog already running"
                return 0
            fi
        fi
	start_daemon
        RETVAL=$?
        [ $RETVAL = 0 ]
        echo
        return $RETVAL
}


#----------------------------------------------------------------------
# run 
#----------------------------------------------------------------------
run() {
        echo -n $"Starting $prog: "

        if [[ -f ${pidfile} ]] ; then
            pid=$( cat $pidfile  )
            isrunning=$( ps -elf | grep  $pid | grep $prog | grep -v grep )

            if [[ -n ${isrunning} ]] ; then
                echo $"$prog already running"
                return 0
            fi
        fi
	start_daemon
        RETVAL=$?
        [ $RETVAL = 0 ]
        echo
        return $RETVAL
}


#----------------------------------------------------------------------
# stop  
#----------------------------------------------------------------------
stop() {
    if [[ -f ${pidfile} ]] ; then
        pid=$( cat $pidfile )
        isrunning=$( ps -elf | grep $pid | grep $prog | grep -v grep | awk '{print $4}' )

        if [[ ${isrunning} -eq ${pid} ]] ; then
            echo -n $"Stop $prog: "
            kill $pid
	    rm -f $pidfile 
        else
            echo -n $"STOP $prog: "
        fi
        RETVAL=$?
    fi
    echo
    return $RETVAL
}

#----------------------------------------------------------------------
# reload
#----------------------------------------------------------------------
reload() {
    echo -n $"Reloading $prog: "
    echo
}

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
case "$STATUS" in
  start)
      start
    ;;
  stop)
    stop
    ;;
  run)
    run
    ;;
  status)
    status -p $pidfile $eg_daemon
    RETVAL=$?
    ;;
  restart)
    stop
    start
    ;;
  force-reload|reload)
    reload
    ;;
  *)
    echo $"Usage: $prog {start|run|stop|restart|force-reload|reload|status}"
    RETVAL=2
esac
exit $RETVAL

