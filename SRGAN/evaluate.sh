#!/bin/bash
 
#for image_indx in {1..1568..50}
# do
#    #echo "image_indx=$image_indx"
#    for i in {3..50..3}
#      do 
#         python evaluate.py --epoch=$i --img_indx=$image_indx
#     done
# done

echo $PWD
#for i in {3..50..3}
#for i in {3..50..3}
for i in 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48
    do 
        python evaluate.py --epoch=$i
    done