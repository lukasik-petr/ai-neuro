ai-neuro 
(C) GNU General Public License, 
made by Petr Lukasik for the TM-AI project, 2022
------------------------------------------------------------------------------------------------------ 
 prediction of the effect of temperature on machining accuracy

 Prerequisites: Linux Debian-11.4 or Ubuntu-20.04,
  miniconda3,
  python 3.9,
  tensorflow 2.9,
  mathplotlib,
  scikit-learn-intelex,
  pandas,
  numpy,
  keras

 Avoid Windows if possible. This is recommended in the Some manuals.
 It has been verified and works on Windows as well. However, 
 it is not possible to transfer the conda environment from 
 Linux to Windows. It has to be reinstalled.

 Pozor pro instalaci je nutno udelat nekolik veci
  1. instalace prostredi miniconda 
       a. stahnout z webu miniconda3 v nejnovejsi verzi
       b. chmod +x Miniconda3-py39_4.11.0-Linux-x86_64.sh
       c. ./Miniconda3-py39_4.11.0-Linux-x86_64.sh

  2. update miniconda
       conda update conda

  3. vyrobit behove prostredi 'tf' v miniconda
       conda create -n tf python=3.8

  4. aktivovat behove prostredi tf (pred tim je nutne zavrit a znovu
     otevrit terminal aby se conda aktivovala.
       conda activate  tf

  5. instalovat tyto moduly (pomoci conda)
       conda install tensorflow=2.8
       conda install mathplotlib
       conda install scikit-learn-intelex
       conda install pandas
       conda install numpy
       conda install keras

  6. v prostredi tf jeste upgrade tensorflow
       pip3 install --upgrade tensorflow
------------------------------------------------------------------------------------------------------ 

------------------------------------------------------------------------------------------------------ 
    pouziti: <nazev_programu> <arg1> <arg2> <arg3> <arg4>
    ai-neuro.py -t <--typ> -m <--model> -e <--epochs> -b <--batch> ")
     
            --help            list help ")
            --typ             typ behu 'train' nebo 'predict'")
                                     train - trenink site")
                                     predict - beh z nauceneho algoritmu")
     
            --model           model neuronove site 'DENSE', 'LSTM', 'GRU'")
                                     DENSE - zakladni model site - nejmene narocny na system")
                                     LSTM - Narocny model rekurentni site s feedback vazbami")
                                     GRU  - Narocny model rekurentni hradlove site")
     
            --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
                                     pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
                                     Pri malych cislech se muze stat, ze sit bude nedoucena ")
                                     a pri velkych cislech preucena - coz je totez jako nedoucena.")
                                     Jedna se tedy o podstatny parametr v procesu uceni site.")
     
            --batch           pocet vzorku v minidavce - cislo v intervalu <32,2048>")
                                     Velikost dávky je počet hodnot vstupních dat, které zavádíte najednou do modelu.")
                                     Mějte prosím na paměti, že velikost dávky ovlivňuje 
                                     dobu tréninku, chybu, které dosáhnete, posuny gradientu atd. 
                                     Neexistuje obecné pravidlo, jaká velikost dávky funguje nejlépe.
                                     Stačí vyzkoušet několik velikostí a vybrat si tu, která dava
                                     nejlepsi vysledky. Snažte se pokud mozno nepoužívat velké dávky,
                                     protože by to přeplnilo pamet. 
                                     Bezne velikosti minidavek jsou 32, 64, 128, 256, 512, 1024, 2048.
     
                                     Plati umera: cim vetsi davka tim vetsi naroky na pamet.
                                                  cim vetsi davka tim rychlejsi zpracovani.
            --units           pocet vypocetnich jednotek cislo v intervalu <32,1024>")
                                     Pocet vypocetnich jednotek urcuje pocet neuronu zapojenych do vypoctu.")
                                     Mějte prosím na paměti, že velikost units ovlivňuje 
                                     dobu tréninku, chybu, které dosáhnete, posuny gradientu atd. 
                                     Neexistuje obecné pravidlo, jak urcit optimalni velikost parametru units.
                                     Obecne plati, ze maly pocet neuronu vede k nepresnym vysledkum a naopak
                                     velky pocet units muze zpusobit preuceni site - tedy stejny efekt jako pri
                                     nedostatecnem poctu units. Pamatujte, ze pocet units vyrazne ovlivnuje alokaci
                                     pameti. pro 1024 units je treba minimalne 32GiB u siti typu LSTM nebo GRU.
     
                                     Plati umera: cim vetsi units tim vetsi naroky na pamet.
                                                  cim vetsi units tim pomalejsi zpracovani.
     
            --shuffle         Nahodne promichani treninkovych dat  <True, False>")
                                     Nahodnym promichanim dat se docili nezavislosti na casove ose.")
                                     V nekterych pripadech je tato metoda velmi vyhodna. 
                                     shuffle = True se uplatnuje jen v rezimu 'train' a pouze na treninkova 
                                     data. Validacni a testovaci data se nemichaji. 
                                     Pokud shuffle neni uveden, je implicitne nastaven na 'True'. 
     
            --txdat1          timestamp zacatku datove mnoziny pro predict, napr '2022-04-09 08:00:00' ")
     
            --txdat2          timestamp konce   datove mnoziny pro predict, napr '2022-04-09 12:00:00' ")
     
                                     parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere
                                     se v uvahu cela mnozina dat k trenovani.
     
     
     
    priklad: ./ai-neuro.py -t train, -m DENSE, -e 64 -b 128 -s True -t1 2022-04-09 08:00:00 -t2 2022-04-09 12:00:00
    nebo:    ./ai-neuro.py --typ train, --model DENSE, --epochs 64 --batch 128 --shuffle True  --txdat1 2022-04-09 08:00:00 --txdat2 2022-04-09 12:00:00
    print('parametr --epochs musi byt cislo typu int <1, 256>')
    POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM nebo GRU!!!
           pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise 
           natrenovany model site..
           V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.
           spusti normalni beh site z jiz natrenovaneho modelu.
           Takze: budte trpelivi...
     
     
    Vstupni parametry: 
      pokud neexistuje v rootu aplikace soubor ai-parms.txt, pak jsou parametry implicitne
      prirazeny z promennych definovanych v programu:
    Jedna se o tyto promenne: 
     
      #Vystupni list parametru - co budeme chtit po siti predikovat
      df_parmx = ['machinedata_m0412','teplota_pr01', 'x_temperature']
     
      #Tenzor predlozeny k uceni site
      df_parmX = ['machinedata_m0112','machinedata_m0212','machinedata_m0312','machinedata_m0412','teplota_pr01', 'x_temperature'];
     
    Pokud pozadujete zmenu parametu j emozno primo v programu poeditovat tyto promenne 
     
    a nebo vyrobit soubor ai-parms.txt s touto syntaxi 
      #Vystupni list parametru - co budeme chtit po siti predikovat
      df_parmx = machinedata_m0412,teplota_pr01,x_temperature'
     
      #Tenzor predlozeny k uceni site
      df_parmX = machinedata_m0412,teplota_pr01, x_temperature
     
    a ten nasledne ulozit v rootu aplikace. (tam kde je pythonovsky zdrojak. 
    POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy vstupniho datoveho CSV souboru (nebo souboruuu)
    a muzou tam byt i uvozovky: priklad: 'machinedata_m0112','machinedata_m0212', to pro snazsi copy a paste 
    z datoveho CSV souboru. 
     
    (C) GNU General Public License, autor Petr Lukasik , 2022 
     
    Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)
                  miniconda3,
                  python 3.9, tensorflow 2.8, mathplotlib,  
                  tensorflow 2.8,
                  mathplotlib,  
                  scikit-learn-intelex,  
                  pandas,  
                  numpy,  
                  keras   
    
