#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace lib {
    namespace utils {
        class GpuTimer {
           private:
            cudaEvent_t start_event;
            cudaEvent_t stop_event;

           public:
            GpuTimer() {
                cudaEventCreate(&start_event);
                cudaEventCreate(&stop_event);
            }

            ~GpuTimer() {
                cudaEventDestroy(start_event);
                cudaEventDestroy(stop_event);
            }

            void start() { cudaEventRecord(start_event, 0); }

            void stop() { cudaEventRecord(stop_event, 0); }

            /**
             * @return elapsed time in milliseconds
             */
            float elapsed() {
                float elapsed;
                cudaEventSynchronize(stop_event);
                cudaEventElapsedTime(&elapsed, start_event, stop_event);
                return elapsed;
            }
        };
    }  // namespace utils
}  // namespace lib