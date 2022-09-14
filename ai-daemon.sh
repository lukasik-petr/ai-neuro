#!/bin/bash

#----------------------------------------------------------------------
# ai-daemon      Startup skript pro ai-daemona
#----------------------------------------------------------------------
FILE_PATH=$(pwd)
# Environment Miniconda.
export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}


prog=$FILE_PATH"/ai-daemon.py"
pidfile=${PIDFILE-$FILE_PATH/pid/ai-daemon.pid}
logfile=${LOGFILE-$FILE_PATH/log/ai-daemon.log}

echo $pidfile
echo $logfile

RETVAL=0
STATUS="$1"
DEBUG_MODE="nodebug"
MODEL="DENSE"
EPOCHS="64"
BATCH="256"
UNITS="71"
ACTF="elu"
TXDAT1='2022-02-15 00:00:00'
#TXDAT2='2022-04-10 23:59:59'
TXDAT2=`date +%Y-%m-%d -d "yesterday"`" 23:59:59"
OPTIONS=""


#----------------------------------------------------------------------
# start_daemon - aktivace miniconda a start demona
#----------------------------------------------------------------------
start_daemon(){
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo "Start ulohy: "$curr_timestamp
    cd ~/ai/ai-daemon/src
    eval "$(conda shell.bash hook)"
    conda activate tf
    echo "Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z"
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
	    --txdat2="$TXDAT2"
    
    conda deactivate
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo "Stop ulohy: "$curr_timestamp

}

#----------------------------------------------------------------------
# start
#----------------------------------------------------------------------
start() {
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
            echo -n $"Stopping $prog: "
            kill $pid
        else
            echo -n $"Stopping $prog: "
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
    echo $"Usage: $prog {start|stop|restart|force-reload|reload|status}"
    RETVAL=2
esac

exit $RETVAL

