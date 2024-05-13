#include <Metal/Metal.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

void fail(const std::string &str) {
  std::cerr << str << std::endl;
  abort();
}

void fail(const std::string &str1, const std::string &str2) {
  std::cerr << str1 << str2 << std::endl;
  abort();
}

template <typename Callable>
float measure_time(unsigned repeat_cnt, Callable c) {
  using namespace std::chrono;
  // Warmup
  c();
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
    fail("Metal is not supported");
  }
  return devices[0];
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
    fail("Failed to compile: ", error.description.UTF8String);
  }
  return library;
}

id<MTLLibrary> compileLibraryFromFile(id<MTLDevice> device,
                                      const std::string &fname) {
  std::ifstream ifs(fname);
  std::stringstream ss;
  ss << ifs.rdbuf();
  ifs.close();
  return compileLibraryFromSource(device, ss.str());
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, unsigned length) {
  id<MTLBuffer> rc = [device newBufferWithLength:length
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    fail("Can't allocate " + std::to_string(length) + " bytes on GPU");
  }
  return rc;
}

float benchmark_int8mm(id<MTLLibrary> lib, const std::string &lib_name,
                       unsigned M, unsigned N, unsigned K) {
  auto buf_A = allocSharedBuffer(lib.device, M * K * 2);
  auto buf_B = allocSharedBuffer(lib.device, N * K);
  auto buf_C = allocSharedBuffer(lib.device, M * N * 2);
  auto buf_S = allocSharedBuffer(lib.device, N * 2);
  id<MTLFunction> func = [lib newFunctionWithName:@"int8pack_mm_bfloat"];
  if (func == nil) {
    fail("Can:t get function");
  }
  NSError *error = nil;
  id<MTLComputePipelineState> cpl =
      [lib.device newComputePipelineStateWithFunction:func error:&error];
  if (cpl == nil) {
    fail("Failed to construct pipeline state: ", error.description.UTF8String);
  }
  std::vector<unsigned> sizes = {M, N, K, 0};
  const auto maxThreadsPerGroup =
      static_cast<decltype(M)>([cpl maxTotalThreadsPerThreadgroup]);
  id<MTLCommandQueue> queue = [lib.device newCommandQueue];
  auto do_compule = ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
      [encoder setComputePipelineState:cpl];
      [encoder setBuffer:buf_A offset:0 atIndex:0];
      [encoder setBuffer:buf_B offset:0 atIndex:1];
      [encoder setBuffer:buf_S offset:0 atIndex:2];
      [encoder setBuffer:buf_C offset:0 atIndex:3];
      [encoder setBytes:sizes.data()
                 length:sizeof(uint32_t) * sizes.size()
                atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(N, M, 1)
          threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M), 1,
                                            1)];
      [encoder endEncoding];
      [cmdBuffer commit];
      [cmdBuffer waitUntilCompleted];
    }
  };
  auto gflops = (M * N * K * 1e-9) / measure_time(200, do_compule);
  std::cout << "Perf of " << lib_name << " dim " << M << "x" << N << "x" << K << " is " << gflops << " GFLOPs"
            << std::endl;
  return gflops;
}

int main() {
  @autoreleasepool {
    id<MTLDevice> device = getMetalDevice();
    std::cout << "Using device " << device.name.UTF8String << std::endl;
    auto naive_int8mm = compileLibraryFromFile(device, "naive_int8mm.metal");
    auto vectorized_int8mm =
        compileLibraryFromFile(device, "vectorized_int8mm.metal");
    benchmark_int8mm(naive_int8mm, "naive_int8mm", 32, 4096, 4096);
    benchmark_int8mm(vectorized_int8mm, "vectorized_int8mm", 32, 4096, 4096);
  }
  return 0;
}
