#!/bin/bash
FILE_PATH=$(pwd)

export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}



function ai-help_() {
      echo "        "
      echo "            -t1 <--txdat1>-timestamp start 'YYYY-MM-DD HH:MM:SS'"
      echo "            -t2 <--txdat2>-timestamp stop 'YYYY-MM-DD HH:MM:SS'"
      echo "                           timestampy urcuji v jakem intervalu   "
      echo "                           se budou vybirat data pro trenink.   "
      echo "                           nejsou povinne. Pokud nejsou uvedeny  "
      echo "                           je do zpracovani vybrana cela mnozina "
      echo "                           dat urcena pro predikci.  "
      echo "                                         "
      echo "PRIKLAD: ./ai-daemon.sh -t1='2022-02-15 00:00:00' -t2='2022-06-30 12:00:00'"
      return
      
}


if [[ $# -eq 0 ]]
then
    ai-help_
fi

for i in "$@"; do
    case $i in
    -s=*|--status=*)#
	STATUS="${i#*=}"
	shift # past argument=value
        ;;
	
    -t1=*|--txdat1=*)#
	TXDAT1="${i#*=}"
	shift # past argument=value
        ;;
    
    -t2=*|--txdat2=*)
	TXDAT2="${i#*=}"
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


	     
echo "bash: Spusteno s parametry: TXDAT1="$TXDAT1" TXDAT2="$TXDAT2
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
echo "Start ulohy: "$curr_timestamp
#cd ~/ai/ai-daemon/src
eval "$(conda shell.bash hook)"
conda activate tf
echo "Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z"
python3 ai-daemon.py --status="$STATUS" --txdat1="$TXDAT1" --txdat2="$TXDAT2"
conda deactivate
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
echo "Stop ulohy: "$curr_timestamp
