#! /bin/bash
grid='grid'
./main $1 $2 $3 $4 $5 $6 $7 $8
../common/gridPlot $1
gridfile=$1$grid
echo $gridfile
python ../common/plotMatrix.py $gridfile
