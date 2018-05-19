/*
    Description : This file contains the TicToc class which can be used to time
    a piece of code.

    @author : mishraiiit
*/

#ifndef TIC_TOC
#define TIC_TOC
#include <chrono>
#include <ctime>
#include <iostream>
#include <stdio.h>

class TicToc {
    public:
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::string s;
        TicToc(std::string s) : s(s) {

        }
        void tic() {
            start = std::chrono::system_clock::now();            
        }

        void toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<float> diff = end-start;
            fprintf(stderr, "%s %lf\n", s.c_str(), diff.count());            
        }
};

#endif