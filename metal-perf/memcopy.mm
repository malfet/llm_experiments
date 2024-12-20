// Compile me as
// clang++ --std=c++17 memcopy.mm -framework Metal -framework Foundation
// Implements arange and matmul
#include <Metal/Metal.h>

#include <chrono>
#include <iostream>
#include <stdexcept>

const std::string &metal_lib = R"METAL(
kernel void arange_f(device float *A [[buffer(0)]],
                     uint idx [[thread_position_in_grid]]) {
   A[idx] = idx;
}

kernel void arange_i(device uint* A [[buffer(0)]],
                     uint idx [[thread_position_in_grid]]) {
   A[idx] = idx;
}

kernel void one_f(device float *A [[buffer(0)]],
                   uint idx [[thread_position_in_grid]]) {
   A[idx] = 1.0;
}

kernel void one_f4(device float4 *A [[buffer(0)]],
                   uint idx [[thread_position_in_grid]]) {
   A[idx] = 1.0;
}

kernel void one_i(device uint* A [[buffer(0)]],
                  uint idx [[thread_position_in_grid]]) {
   A[idx] = 1;
}

kernel void one_i4(device uint4* A [[buffer(0)]],
                  uint idx [[thread_position_in_grid]]) {
   A[idx] = 1;
}

kernel void inc_f(device float *A [[buffer(0)]],
                   uint idx [[thread_position_in_grid]]) {
   A[idx] += 1.0;
}

kernel void inc_f4(device float4 *A [[buffer(0)]],
                   uint idx [[thread_position_in_grid]]) {
   A[idx] += 1.0;
}

kernel void inc_i(device uint* A [[buffer(0)]],
                  uint idx [[thread_position_in_grid]]) {
   A[idx] += 1;
}

kernel void full_f(device float* A [[buffer(0)]],
                   constant float& v [[buffer(1)]],
                   uint idx [[thread_position_in_grid]]) {
   A[idx] = v;
}
)METAL";

template <typename Callable>
float measure_time(unsigned repeat_cnt, Callable c) {
  using namespace std::chrono;
  auto start = high_resolution_clock::now();
  for (unsigned idx = 0; idx < repeat_cnt; idx++) {
    c();
  }
  auto end = high_resolution_clock::now();
  return duration<float>(end - start).count() / repeat_cnt;
}
id<MTLDevice> getMetalDevice() {
  NSArray *devices = [MTLCopyAllDevices() autorelease];
  if (devices.count == 0) {
    throw std::runtime_error("Metal is not supported");
  }
  return devices[0];
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    throw std::runtime_error("Can't allocate " + std::to_string(length) +
                             " bytes on GPU");
  }
  return rc;
}

id<MTLLibrary> compileLibraryFromSource(id<MTLDevice> device,
                                        const std::string &source) {
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion3_1];
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:options
                     error:&error];
  if (library == nil) {
    throw std::runtime_error(std::string("Failed to compile: ") +
                             error.description.UTF8String);
  }
  return library;
}

id<MTLComputePipelineState> getComputePipelineState(id<MTLLibrary> lib, const char *kernel_name) {
  auto func = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
  if (func == nil) {
    throw std::runtime_error("Can't get function");
  }
  NSError *error = nil;
  auto cpl = [lib.device newComputePipelineStateWithFunction:func error:&error];
  if (cpl == nil) {
    throw std::runtime_error(
        std::string("Failed to construct pipeline state: ") +
        error.description.UTF8String);
  }
  [func release];
  return cpl;
}

template<unsigned ops_per_thread = 1>
void benchmark_kernel(id<MTLLibrary> lib, const char* kernel_name) {
  constexpr auto block_size = 100 * 1024 * 1024;
  auto cpl = getComputePipelineState(lib, kernel_name);
  const auto maxTpG = cpl.maxTotalThreadsPerThreadgroup;
  auto group_size = MTLSizeMake(maxTpG, 1, 1);

  auto dev = lib.device;
  auto buffer = allocSharedBuffer(dev, block_size * sizeof(float));
  auto queue = [dev newCommandQueue];

  auto do_compute = ^() {
    @autoreleasepool {
      auto cmdBuffer = [queue commandBuffer];
      auto encoder = [cmdBuffer computeCommandEncoder];
      [encoder setComputePipelineState:cpl];
      [encoder setBuffer:buffer offset:0 atIndex:0];
      [encoder dispatchThreads:MTLSizeMake(block_size / ops_per_thread, 1, 1)
          threadsPerThreadgroup:group_size];
      [encoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
    }
  };

  // Benchmark performance (including dispatch overhead)
  auto gbps  = (sizeof(float) * block_size / (1024 * 1024 * 1024.0)) / measure_time(200, do_compute);
  std::cout << "Perf of " << kernel_name <<  " is " << gbps << " GB/s" << std::endl;
  [queue release];
  [buffer release];
  [cpl release];
}

int main() {
  auto device = getMetalDevice();
  std::cout << "Using device " << device.name.UTF8String << std::endl;
  auto lib = compileLibraryFromSource(device, metal_lib);
  benchmark_kernel(lib, "arange_f");
  benchmark_kernel(lib, "arange_i");
  benchmark_kernel(lib, "one_f");
  benchmark_kernel(lib, "one_i");
  benchmark_kernel<4>(lib, "one_i4");
  benchmark_kernel<4>(lib, "one_f4");
  benchmark_kernel(lib, "inc_f");
  benchmark_kernel(lib, "inc_i");
  benchmark_kernel<4>(lib, "inc_f4");
  [lib release];
}
