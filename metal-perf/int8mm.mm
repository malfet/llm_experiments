#include <Metal/Metal.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
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

inline uint32_t float_as_int(float f) {
  union {
    float f;
    uint32_t i;
  } x;
  x.f = f;
  return x.i;
}

inline float int_as_float(uint32_t i) {
  union {
    float f;
    uint32_t i;
  } x;
  x.i = i;
  return x.f;
}

struct BFloat16 {
  BFloat16(float x) : val(float_as_int(x) >> 16) {}
  operator float() const { return int_as_float(val << 16); }

  uint16_t val;
};
using Float16 = _Float16;

struct Int8MMOpDescriptor {
  Int8MMOpDescriptor(id<MTLDevice> device, const std::string &lib_name_,
                     unsigned M_, unsigned N_, unsigned K_)
      : Int8MMOpDescriptor(device, M_, N_, K_) {
    lib_name = lib_name_;
    lib = compileLibraryFromFile(device, lib_name + ".metal");
  }
  Int8MMOpDescriptor(id<MTLDevice> device, unsigned M_, unsigned N_,
                     unsigned K_)
      : M(M_), N(N_), K(K_), lib(nil) {
    allocBuffers(device);
  }

  virtual void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                               unsigned maxThreadsPerGroup) const {
    [encoder dispatchThreads:MTLSizeMake(N, M, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(maxThreadsPerGroup, M), 1,
                                          1)];
  }

  void encodeMM(id<MTLCommandBuffer> cmdBuffer,
                id<MTLComputePipelineState> cpl) const {
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    std::vector<unsigned> sizes = {M, K, N, 0};
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
    dispatchThreads(encoder, maxThreadsPerGroup);
    [encoder endEncoding];
  }

  template <typename T> void init() {
    T *a_ptr = reinterpret_cast<T *>([buf_A contents]);
    int8_t *b_ptr = reinterpret_cast<int8_t *>([buf_B contents]);
    T *c_ptr = reinterpret_cast<T *>([buf_C contents]);
    T *s_ptr = reinterpret_cast<T *>([buf_S contents]);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> int_distrib(-32, 32);
    std::uniform_real_distribution<> real_distrib(-1.0, 1.0);

    for (unsigned idx = 0; idx < M * K; ++idx) {
      a_ptr[idx] = real_distrib(generator);
    }
    for (unsigned idx = 0; idx < N * K; ++idx) {
      b_ptr[idx] = int_distrib(generator);
    }
    for (unsigned idx = 0; idx < N; ++idx) {
      s_ptr[idx] = (idx + 1.0) / N;
    }
    for (unsigned idx = 0; idx < M * N; ++idx) {
      c_ptr[idx] = -1.0;
    }
  }

  template <typename T>
  bool validate(float atol_lim = 5e-4, float rtol_lim = 5e-3) const {
    T *a_ptr = reinterpret_cast<T *>([buf_A contents]);
    int8_t *b_ptr = reinterpret_cast<int8_t *>([buf_B contents]);
    T *c_ptr = reinterpret_cast<T *>([buf_C contents]);
    T *s_ptr = reinterpret_cast<T *>([buf_S contents]);

    for (unsigned m = 0; m < M; m++) {
      for (unsigned n = 0; n < N; n++) {
        float expected = float(c_ptr[m * N + n]);
        float rc = 0.0;
        for (unsigned k = 0; k < K; k++) {
          rc += float(b_ptr[n * K + k]) * float(a_ptr[m * K + k]);
        }
        rc *= s_ptr[n];
        auto atol = std::abs(rc - expected);
        auto rtol =
            atol / std::max(std::min(std::abs(expected), std::abs(rc)), 1e-6f);
        if (rtol > rtol_lim && atol > atol_lim) {
          std::cerr << "Result " << expected << " vs expected " << rc
                    << " (atol=" << atol << " ,rtol=" << rtol << ") at " << m
                    << ":" << n << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  template <typename T> float benchmark(float atol_lim = 5e-4, float rtol_lim = 5e-3) {
    init<T>();
    id<MTLFunction> func = [lib
        newFunctionWithName:[NSString
                                stringWithFormat:@"int8pack_mm_%s",
                                                 type_string<T>().c_str()]];
    if (func == nil) {
      fail("Can:t get function");
    }
    NSError *error = nil;
    auto cpl = [lib.device newComputePipelineStateWithFunction:func
                                                         error:&error];
    if (cpl == nil) {
      fail("Failed to construct pipeline state: ",
           error.description.UTF8String);
    }
    id<MTLCommandQueue> queue = [lib.device newCommandQueue];
    auto do_compute = ^() {
      @autoreleasepool {
        auto desc = [MTLCommandBufferDescriptor new];
        desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        id<MTLCommandBuffer> cmdBuffer = [queue commandBufferWithDescriptor:desc];
        encodeMM(cmdBuffer, cpl);
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
      }
    };

    // Validate (and capture trace if needed)
    auto captureManager = [MTLCaptureManager sharedCaptureManager];
    auto captureDescriptor = [MTLCaptureDescriptor new];
    captureDescriptor.captureObject = queue;
    captureDescriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    captureDescriptor.outputURL =
        [NSURL fileURLWithPath:[NSString stringWithFormat:@"%s.gputrace",
                                                          lib_name.c_str()]];
    [captureManager startCaptureWithDescriptor:captureDescriptor error:nil];

    do_compute();

    [captureManager stopCapture];

    if (!validate<T>(atol_lim, rtol_lim)) {
      fail("Failed to validate" + lib_name);
    }
    auto gflops = (M * N * K * 1e-9) / measure_time(200, do_compute);
    std::cout << "Perf of " << lib_name << " type " << type_string<T>()
              << " dim " << M << "x" << N << "x" << K << " is " << gflops
              << " GFLOPs" << std::endl;
    return gflops;
  }

private:
  template <typename T> std::string type_string() const;
  template <> std::string type_string<BFloat16>() const { return "bfloat"; }
  template <> std::string type_string<float>() const { return "float"; }
  template <> std::string type_string<Float16>() const { return "half"; }
  void allocBuffers(id<MTLDevice> device, const unsigned elem_size = 4) {
    buf_A = allocSharedBuffer(device, M * K * elem_size);
    buf_B = allocSharedBuffer(device, N * K);
    buf_C = allocSharedBuffer(device, M * N * elem_size);
    buf_S = allocSharedBuffer(device, N * elem_size);
  }

public:
  unsigned M, N, K;    // Input-output matirx dims
  id<MTLBuffer> buf_A; // MxK elements
  id<MTLBuffer> buf_B; // NxK elements
  id<MTLBuffer> buf_C; // MxN elements
  id<MTLBuffer> buf_S; // N elements
  id<MTLLibrary> lib;
  std::string lib_name;
};

struct Int8MMBlockOpDesciptor : public Int8MMOpDescriptor {
  using Int8MMOpDescriptor::Int8MMOpDescriptor;
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    constexpr auto blockSize = 8;
    if (maxThreadsPerGroup < blockSize * blockSize) {
      throw std::runtime_error("Can't dispatch!");
    }
    [encoder dispatchThreads:MTLSizeMake(M * N / 4, blockSize, 1)
        threadsPerThreadgroup:MTLSizeMake(blockSize, blockSize, 1)];
  }
};

struct Int8MMBlockMat4xMat4OpDesciptor : public Int8MMOpDescriptor {
  using Int8MMOpDescriptor::Int8MMOpDescriptor;
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    constexpr auto blockSize = 8;
    if (maxThreadsPerGroup < blockSize * blockSize) {
      throw std::runtime_error("Can't dispatch!");
    }
    [encoder dispatchThreads:MTLSizeMake(M * N / 16, blockSize, 1)
        threadsPerThreadgroup:MTLSizeMake(blockSize, blockSize, 1)];
  }
};

struct Int8MMMat4OpDescriptor : public Int8MMOpDescriptor {
  using Int8MMOpDescriptor::Int8MMOpDescriptor;
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    constexpr auto blockSize = 8;
    if (maxThreadsPerGroup < blockSize * blockSize) {
      throw std::runtime_error("Can't dispatch!");
    }
    [encoder dispatchThreads:MTLSizeMake(N/4, M, 1)
        threadsPerThreadgroup:MTLSizeMake(blockSize, blockSize, 1)];
  }
};

struct Int8MMMat4xMat4OpDescriptor : public Int8MMOpDescriptor {
  using Int8MMOpDescriptor::Int8MMOpDescriptor;
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    constexpr auto blockSize = 8;
    if (maxThreadsPerGroup < blockSize * blockSize) {
      throw std::runtime_error("Can't dispatch!");
    }
    [encoder dispatchThreads:MTLSizeMake(N/4, M/4, 1)
        threadsPerThreadgroup:MTLSizeMake(blockSize, blockSize, 1)];
  }
};

struct Int8MMLlamaCppDescriptor : public Int8MMOpDescriptor {
  using Int8MMOpDescriptor::Int8MMOpDescriptor;
  void dispatchThreads(id<MTLComputeCommandEncoder> encoder,
                       unsigned maxThreadsPerGroup) const override {
    [encoder setThreadgroupMemoryLength:12288 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake( (M + 31)/32, (N + 63)/64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
};

int main() {
  unsigned M, N, K;
  std::tie(M, N, K) = std::make_tuple(32, 4128, 4096);
  @autoreleasepool {
    id<MTLDevice> device = getMetalDevice();
    std::cout << "Using device " << device.name.UTF8String << std::endl;
    Int8MMOpDescriptor naive_int8mm(device, "naive_int8mm", M, N, K);
    Int8MMOpDescriptor reduce_vec4_int8mm(device, "reduce_vec4_int8mm", M, N,
                                          K);
    Int8MMMat4OpDescriptor reduce_mat4_int8mm(device, "reduce_mat4_int8mm", M, N,
                                          K);
    Int8MMMat4xMat4OpDescriptor reduce_mat4xmat4_int8mm(device, "reduce_mat4xmat4_int8mm", M, N,
                                          K);
    Int8MMBlockOpDesciptor reduce_group_int8mm(device, "reduce_group_int8mm", M,
                                               N, K);
    Int8MMBlockMat4xMat4OpDesciptor reduce_group_mat4xmat4_int8mm(device, "reduce_group_mat4xmat4_int8mm", M, N, K);
    Int8MMLlamaCppDescriptor llama_cpp_int8mm(device, "llama_cpp_int8mm", M, N, K);

    reduce_group_mat4xmat4_int8mm.benchmark<BFloat16>();
    reduce_group_int8mm.benchmark<BFloat16>();
    reduce_mat4xmat4_int8mm.benchmark<BFloat16>();
    reduce_mat4_int8mm.benchmark<BFloat16>();
    reduce_vec4_int8mm.benchmark<BFloat16>();
    naive_int8mm.benchmark<BFloat16>();
    llama_cpp_int8mm.benchmark<Float16>(2e-3, 1e-2);
  }
  return 0;
}
