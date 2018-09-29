#!/bin/bash

for folder in ./GBM/*; do
    #folder='./testvols/PX0AX_cor'
    count=0
    for im in $folder/*; do
        ((count=count+1))
    done
    ((count=count-1))
    echo $count
    for i in $(seq 0 $count); do
        echo $i
        python main.py --mode=evaluate --imid=$i --save_path=./output_GBM/$folder --lr_path=$folder
    done
done
    
