#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

namespace TimeUtils {
    template<typename Callable> std::chrono::duration<double> elapsedEvaluatingFunc(Callable callable){
        auto t1 = std::chrono::high_resolution_clock::now();
        callable();
        auto t2 = std::chrono::high_resolution_clock::now();

        /* Getting number of milliseconds as a double. */
        std::chrono::duration<double, std::milli> ms_T = t2 - t1;
        return ms_T;
    }
};
#endif
