g++ -std=c++11 -msse4 -O4 -I ../../lib gridPlot.cpp -o gridPlot
if [ "$2" = "cpu" ]; then
    ./gridPlot $1 "cpu"
    python3 plotMatrix.py "$1grid_cpu"
elif [ "$2" = "gpu" ]; then
    ./gridPlot $1 "gpu"
    python3 plotMatrix.py "$1grid_gpu"
else
    echo "Error: Second parameter must be either 'cpu' or 'gpu'"
    exit 1
fi