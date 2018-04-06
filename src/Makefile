clean:
	rm -rf ./main ./gridPlot
compile_cmk_on:
	g++ -std=c++11 -msse4 -O4 -I ../lib main.cpp -o main
	g++ -std=c++11 -msse4 -O4 -I ../lib gridPlot.cpp -o gridPlot
compile_cmk_off:
	g++ -std=c++11 -D CMK_OFF -msse4 -O4 -I ../lib main.cpp -o main
	g++ -std=c++11 -msse4 -O4 -I ../lib gridPlot.cpp -o gridPlot
