#include <iostream>
#include <stdexcept>

#include <Metal/Metal.h>

#include <mach-o/getsect.h>


int main() {
  NSArray *devices = MTLCopyAllDevices();
  id<MTLDevice> device = [devices objectAtIndex:0];
  if (device == nil) {
    throw std::runtime_error("Can't get metal device");
  }
  std::cout << "Using device " << device.name.UTF8String << std::endl;
  int64_t triu_size = 230;
  id<MTLBuffer> rc = [device newBufferWithLength: 2 * triu_size * sizeof(int64_t)
                                         options:MTLResourceStorageModeShared];
  if (rc == nil) {
    throw std::runtime_error("Can't allocate memory");
  }
  NSError *error = nil;
  unsigned long mtl_lib_size = 0;
  char *mtl_lib_data = getsectdata("__TEXT", "metal_library", &mtl_lib_size);
  if (mtl_lib_data == nullptr) {
    throw std::runtime_error("Can't find metal library section");
  }
  auto lib_data = dispatch_data_create(mtl_lib_data, mtl_lib_size, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
  if (library == nil) {
    throw std::runtime_error(std::string("Can't load library") + error.description.UTF8String);
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  auto desc = [MTLCommandBufferDescriptor new];
  desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
  id<MTLCommandBuffer> cmdBuffer = [queue commandBufferWithDescriptor:desc];
  [cmdBuffer commit];
  [cmdBuffer waitUntilCompleted];
  
  return 0;
}
