// Compile me as
// clang++ --std=c++17 sort.mm -framework Metal -framework Foundation
// Implements sort
#include <Metal/Metal.h>

#include <chrono>
#include <iostream>
#include <stdexcept>

const std::string &metal_lib = R"METAL(
using namespace metal;
kernel void bitonic_sort(device float *A [[buffer(0)]],
                         uint idx [[thread_position_in_grid]]) {
  // Assuming all 1024 elem sort
  threadgroup float shmem[1024];
  shmem[idx] = A[idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int step = 0; step < 10; step++) {
    const auto buddy = idx ^ step;
    const auto ascending = (( idx >> step) & 1) == 0;
    const auto a = shmem[idx];
    const auto b = shmem[buddy];
    if ( (a > b) == ascending) {
        shmem[idx] = b;
        shmem[buddy] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  A[idx] = shmem[idx];
}

)METAL";

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

void validate_sort(id<MTLLibrary> lib, const char* kernel_name) {
  constexpr auto block_size = 1024;
  auto cpl = getComputePipelineState(lib, kernel_name);
  const auto maxTpG = cpl.maxTotalThreadsPerThreadgroup;
  auto group_size = MTLSizeMake(maxTpG, 1, 1);

  auto dev = lib.device;
  auto buffer = allocSharedBuffer(dev, block_size * sizeof(float));
  auto queue = [dev newCommandQueue];

  float *data = reinterpret_cast<float*>([buffer contents]);
  for(int idx = 0; idx < block_size; ++idx) {
     data[idx] = sin(.43 * idx);
  }
  @autoreleasepool {
    auto cmdBuffer = [queue commandBuffer];
    auto encoder = [cmdBuffer computeCommandEncoder];
    [encoder setComputePipelineState:cpl];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(block_size, 1, 1)
        threadsPerThreadgroup:group_size];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
  }

  for(int idx = 0; idx < 16; ++idx) {
    std::cout << "data[" << idx << "] = " << data[idx] << std::endl;
  }

  [queue release];
  [buffer release];
  [cpl release];
}





int main() {
  auto device = getMetalDevice();
  std::cout << "Using device " << device.name.UTF8String << std::endl;
  auto lib = compileLibraryFromSource(device, metal_lib);

  validate_sort(lib, "bitonic_sort");
  [lib release];
}
