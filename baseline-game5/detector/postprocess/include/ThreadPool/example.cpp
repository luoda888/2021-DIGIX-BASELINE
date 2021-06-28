#include <iostream>
#include <vector>
#include <chrono>

#include "ThreadPool.h"

std::vector<int> my_task(int i)
{
    std::vector<int> res;
    res.push_back(i*i);
    return res;
}

int main()
{
    
    ThreadPool pool(4);
    std::vector< std::future<std::vector<int>> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue(my_task, i)
        );
    }

    for(auto && result: results)
        std::cout << result.get()[0] << ' ';
    std::cout << std::endl;
    
    return 0;
}
