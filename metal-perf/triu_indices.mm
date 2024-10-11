#include <iostream>
#include <stdexcept>

#include <Metal/Metal.h>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>


int main() {
  // Fetch metal library context from an executable section
  const auto* mach_header = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(0));
  unsigned long mtl_lib_size = 0;
  const auto* mtl_lib_data = getsectiondata(mach_header, "__TEXT", "metal_library", &mtl_lib_size);
  if (mtl_lib_data == nullptr) {
    throw std::runtime_error("Can't find metal library section");
  }

  NSArray *devices = MTLCopyAllDevices();
  id<MTLDevice> device = [devices objectAtIndex:0];
  if (device == nil) {
    throw std::runtime_error("Can't get metal device");
  }
  std::cout << "Using device " << device.name.UTF8String << std::endl;

  auto lib_data = dispatch_data_create(mtl_lib_data, mtl_lib_size, dispatch_get_main_queue(), ^() {});
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
  if (library == nil) {
    throw std::runtime_error(std::string("Can't load library") + error.description.UTF8String);
  }

  id<MTLFunction> func = [library newFunctionWithName:@"triu_indices_long"];
  if (func == nil) {
    throw std::runtime_error("Can't get function");
  }

  auto cpl = [device newComputePipelineStateWithFunction:func error:&error];
  if (cpl == nil) {
    throw std::runtime_error(std::string("Can't initialize compute pipeline ") + error.description.UTF8String);
  }

  // Kernel parameters
  int64_t col_offset = 0;
  int64_t m_first_row = 21;
  int64_t col = 21;
  int64_t rectangle_size = 0;
  int64_t triu_size = 230;

  id<MTLBuffer> rc = [device newBufferWithLength: 2 * triu_size * sizeof(int64_t)
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    throw std::runtime_error("Can't allocate memory");
  }

  // Dispatch kernel
  id<MTLCommandQueue> queue = [device newCommandQueue];
  auto desc = [MTLCommandBufferDescriptor new];
  desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
  id<MTLCommandBuffer> cmdBuffer = [queue commandBufferWithDescriptor:desc];
  id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
  const auto maxThreadsPerGroup = [cpl maxTotalThreadsPerThreadgroup];
  [encoder setComputePipelineState:cpl];
  [encoder setBuffer:rc offset:0 atIndex:0];
  [encoder setBytes:&col_offset length:sizeof(uint64_t) atIndex:1];
  [encoder setBytes:&m_first_row length:sizeof(uint64_t) atIndex:2];
  [encoder setBytes:&col length:sizeof(uint64_t) atIndex:3];
  [encoder setBytes:&rectangle_size length:sizeof(uint64_t) atIndex:4];
  [encoder setBytes:&triu_size length:sizeof(uint64_t) atIndex:5];
  [encoder dispatchThreads:MTLSizeMake(triu_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, static_cast<decltype(maxThreadsPerGroup)>(triu_size)), 1, 1)];
  [encoder endEncoding];
  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];

  // Print results
  auto *values = reinterpret_cast<int64_t*>([rc contents]);
  for(int i = 20; i < 30; ++i) {
    std::cout << i << "; " << values[i] << " x " << values[i + triu_size] << std::endl;
  }

  return 0;
}
