#!/bin/bash
#cd ~/workspaces/eclipse-python-workspace/ai-daemon/src
cd /home/plukasik/ai/ai-daemon/src

#----------------------------------------------------------------------
# ai-ad      Startup skript pro ai-ad demona
#----------------------------------------------------------------------
FILE_PATH=$(pwd)
# Environment Miniconda.
export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0

prog=$FILE_PATH"/ai-daemon.py"
pidfile=${PIDFILE-$FILE_PATH/pid/ai-daemon.pid}
logfile=${LOGFILE-$FILE_PATH/log/ai-daemon.log}

RETVAL=0
STATUS="$1"
DEBUG_MODE="debug"
MODEL="LSTM"
EPOCHS="57"
BATCH="128"
UNITS="91" #DENSE=79    LSTM = 191
ACTF="elu"
TXDAT1="2022-01-01 00:00:01"
#TXDAT2="2099-23-23 23:59:59"
TXDAT2=`date +%Y-%m-%d -d "yesterday"`" 23:59:59"
OPTIONS=""
ILCNT="6" 


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
    python3 ai-daemon.py \
	    --status="$STATUS" \
	    --debug_mode="$DEBUG_MODE"\
	    --pidfile="$pidfile" \
	    --logfile="$logfile" \
	    --model="$MODEL" \
	    --epochs="$EPOCHS" \
	    --batch="$BATCH" \
	    --units="$UNITS" \
	    --actf="$ACTF" \
	    --txdat1="$TXDAT1" \
	    --txdat2="$TXDAT2" \
	    --ilcnt="$ILCNT"
    
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
case "$1" in
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

