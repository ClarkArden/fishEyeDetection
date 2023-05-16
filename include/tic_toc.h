#pragma once
 
#include <ctime>
#include <cstdlib>
#include <chrono>
 
class TicToc
{
  public:
    TicToc()
    {
        tic();
    }
 
    void tic()
    {
        start = std::chrono::system_clock::now();
    }
 
    double toc()
    {
        end = std::chrono::system_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return duration_ms;
    }
    void toc_out(std::string prompt)
    {
        end = std::chrono::system_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << prompt <<"  use time :"<< duration_ms << std::endl;
    }
 
  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};