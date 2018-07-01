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
#include <string>
#include "../common/termcolor.hpp"
#include <stdio.h>

#define LENGTH 60

class TicToc {
    public:
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::string s;
        int level;
        TicToc(std::string s, int level) : s(s), level(level) {

        }
        void tic() {
            start = std::chrono::system_clock::now();
        }

        void toc() {
            end = std::chrono::system_clock::now();
            std::chrono::duration<float> diff = end-start;
            s = "\033[1;34m[time] \033[0m"+ s;
            for(int i = 0; i < level; i++) {
                fprintf(stderr, " ");
            }
            while(s.size() != LENGTH)
                s = s + ' ';
            fprintf(stderr, "%s : %lf.\n", s.c_str(), diff.count());
        }
};

template<typename T>
void printScreen(const int level, std::string s, const T tm) {
  for(int i = 0; i < level; i++) {
    std::cout << " ";
  }
  std::cout << termcolor::green << termcolor::bold << "[info] " << termcolor::reset;
  while(s.size() + 18 != LENGTH) {
    s = s + ' ';
  }
  std::cout << s << " : " << tm << ".\n";
}

#endif
