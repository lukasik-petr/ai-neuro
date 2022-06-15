#!/bin/bash
FILE_PATH=$(pwd)

export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}


ai-help_() {
      echo "  parametry -t <--typ>   - spusteno v rezimu trenink nebo predikce "
      echo "                           muze nabyvat hodnot:                    "
      echo "                             'train' - trenink site                   "
      echo "                             'predict' - predikce                     "
      echo "                           POZOR!!! prvni spusteni musi byt v rezimu"
      echo "                           train!"
      echo "                                         "
      echo "            -m <--model>  -typ site, muze nabyvat hodnot:"
      echo "                             'DENSE' - sit typu DENSE - zakladni model"
      echo "                             'GRU'   - sit typu GRU   - rekurentni sit"
      echo "                             'LSTM'  - sit typu LSTM  - rekurentni sit"
      echo "                           POZOR!!! je treba mit na pameti ze rekurentni"
      echo "                           site jsou velmi narocne na systemove zdroje"
      echo "                           trenink u techto siti muze trvat radove hodiny"
      echo "                                         "
      echo "            -e <--epochs> -pocet treninkovych epoch <64,128>"
      echo "                           optimalni velikost epoch je v intervalu"
      echo "                           cca.<64,128> mensi pocet muze znamenat"
      echo "                           ze sit bude 'nedotrenovan' a nebo naopak"
      echo "                           Pocet epoch lze doladit pomoci vystupnich"
      echo "                           grafu graf_acc*.pdf a graf_loss*.pdf"
      echo "                           ktere vznikaji pokazde, kdyz je sit v rezimu"
      echo "                           train.              "
      echo "                                         "
      echo "            -b <--batch>  -velikost vzorku dat  <32,2048>"
      echo "                           optimalni velikost batch je v intervalu"
      echo "                           cca.<32,2048>.                         "
      echo "                           Plati ze cim vetsi parametr batch tim"
      echo "                           rychlejsi zpracovani, ale zaroven vyssi"
      echo "                           naroky na pamet."
      echo "        "
      echo "            -u <--units>  -velikost vypoc.jednotek <32,1024>"
      echo "                           Plati ze cim vetsi parametr units tim"
      echo "                           pomalejsi zpracovani, a vyrazne(!) vyssi"
      echo "                           naroky na pamet."
      echo "        "
      echo "            -s <--shuffle>-Nahodne promichani dat <True,False>"
      echo "                           Pokud shuffle neni uveden, implicitne"
      echo "                           je nastaven na True."
      echo "        "
      echo "            -t1 <--txdat1>-timestamp start 'YYYY-MM-DD HH:MM:SS'"
      echo "            -t2 <--txdat2>-timestamp stop 'YYYY-MM-DD HH:MM:SS'"
      echo "                           timestampy urcuji v jakem intervalu   "
      echo "                           se budou vybirat data pro predikci.   "
      echo "                           nejsou povinne. Pokud nejsou uvedeny  "
      echo "                           je do zpracovani vybrana cela mnozina "
      echo "                           dat urcena pro predikci.  "
      echo "                                         "
      echo "PRIKLAD: ./ai-neuro.sh -t=predict -m=DENSE -e=64 -b=128 -u=512 -s=TRUE -t1='2022-04-09 08:00:00' -t2='2022-04-09 12:00:00'"

}


if [[ $# -eq 0 ]]
then
    ai-help_
    exit 1
fi



for i in "$@"; do
  case $i in
    -t=*|--typ=*)
        TYP="${i#*=}"
        shift # past argument=value
        ;;
    -m=*|--model=*)
        MODEL="${i#*=}"
        shift # past argument=value
        ;;
    -e=*|--epochs=*)
        EPOCHS="${i#*=}"
        shift # past argument=value
        ;;
    -b=*|--batch=*)
	BATCH="${i#*=}"
	shift # past argument=value
        ;;
    -u=*|--units=*)
	UNITS="${i#*=}"
	shift # past argument=value
        ;;
    -s=*|--shuffle=*)
	SHUFFLE="${i#*=}"
	shift # past argument=value
        ;;
    -t1=*|--txdat1=*)
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

echo "bash: Spusteno s parametry: TYP=" $TYP" MODEL="$MODEL" EPOCHS="$EPOCHS" BATCH="$BATCH"  UNITS="$UNITS" SHUFFLE="$SHUFFLE" TXDAT1="$TXDAT1" TXDAT2="$TXDAT2
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
echo "Start ulohy: "$curr_timestamp
cd ~/ai/src
eval "$(conda shell.bash hook)"
conda activate tf
echo "Aproximace prubehu funkci, sit typu " $MODEL " pro jednotlive osy X,Y,Z"
python3 ai-neuro.py --typ "$TYP" --model "$MODEL" --epochs "$EPOCHS" --batch "$BATCH" --units "$UNITS" --shuffle "$SHUFFLE" --txdat1="$TXDAT1" --txdat2="$TXDAT2"
conda deactivate
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
echo "Stop ulohy: "$curr_timestamp


