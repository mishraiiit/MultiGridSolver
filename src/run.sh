#! /bin/bash
grid='grid'
./main $1 $2 $3 $4
./gridPlot $1
gridfile=$1$grid
echo $gridfile
python plotMatrix.py $gridfile
