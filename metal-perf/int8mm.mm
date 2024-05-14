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


struct Int8MMOpDescriptor {
  Int8MMOpDescriptor(id<MTLDevice> device, unsigned M_, unsigned N_, unsigned K_, unsigned elem_size = 2):M(M_), N(N_), K(K_) {
    buf_A = allocSharedBuffer(device, M * K * elem_size);
    buf_B = allocSharedBuffer(device, N * K);
    buf_C = allocSharedBuffer(device, M * N * elem_size);
    buf_S = allocSharedBuffer(device, N * elem_size);
  }
  void encodeNaiveMM(id<MTLCommandBuffer> cmdBuffer, id<MTLComputePipelineState> cpl, unsigned groupM = 1) const {
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    std::vector<unsigned> sizes = {M, N, K, 0};
    const auto maxThreadsPerGroup =
        static_cast<decltype(M)>([cpl maxTotalThreadsPerThreadgroup]);
    [encoder setComputePipelineState:cpl];
    [encoder setBuffer:buf_A offset:0 atIndex:0];
    [encoder setBuffer:buf_B offset:0 atIndex:1];
    [encoder setBuffer:buf_S offset:0 atIndex:2];
    [encoder setBuffer:buf_C offset:0 atIndex:3];
    [encoder setBytes:sizes.data()
               length:sizeof(uint32_t) * sizes.size()
              atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(N, M / groupM, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M / groupM), 1, 1)];
    [encoder endEncoding];
  }
  unsigned M, N, K;
  id<MTLBuffer> buf_A; // MxK elements
  id<MTLBuffer> buf_B; // NxK elements
  id<MTLBuffer> buf_C; // MxN elements
  id<MTLBuffer> buf_S; // N elements
};

float benchmark_int8mm(id<MTLLibrary> lib, const std::string &lib_name,
                       unsigned M, unsigned N, unsigned K) {
  Int8MMOpDescriptor op_desc(lib.device, M, N, K);
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
  id<MTLCommandQueue> queue = [lib.device newCommandQueue];
  auto do_compule = ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
      op_desc.encodeNaiveMM(cmdBuffer, cpl);
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
    auto reduce_vec4_int8mm = compileLibraryFromFile(device, "reduce_vec4_int8mm.metal");
    benchmark_int8mm(naive_int8mm, "naive_int8mm", 32, 4096, 4096);
    benchmark_int8mm(reduce_vec4_int8mm, "reduce_vec4_int8mm", 32, 4096, 4096);
  }
  return 0;
}
