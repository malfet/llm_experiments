// Metal GPU roofline benchmark: measures peak compute (FP32/FP16/BF16/INT), memory bandwidth,
// and — on macOS 26.0+ — MPP matmul2d GEMM throughput (float/half/bfloat, TN layout).
// Build: swiftc -O -o metal_roofline metal_roofline.swift -framework Metal -framework Foundation
// Run:   ./metal_roofline
//
// Results (peak throughput):
// ┌────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
// │ Metric         │ M5 Pro (20-core)     │ M4 Max (40-core)     │ M4 Pro (20-core)     │ M2 Pro (19-core)     │ M4 (10-core)         │ M2 (10-core)         │
// ├────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
// │ FP32 FMA TFLOPS│   7.9 @  8 chains    │  15.5 @ 32 chains    │   7.7 @ 16 chains    │   2.9 @ 32 chains    │   3.9 @ 32 chains    │   1.6 @ 32 chains    │
// │ FP16 FMA TFLOPS│  12.7 @ 32 chains    │  15.6 @ 32 chains    │   7.9 @ 24 chains    │   5.5 @ 32 chains    │   4.0 @ 32 chains    │   3.0 @ 32 chains    │
// │ BF16 FMA TFLOPS│   8.2 @  8 chains    │  15.7 @ 32 chains    │   7.9 @ 24 chains    │   1.0 @  8 chains    │   4.0 @ 32 chains    │   0.5 @  8 chains    │
// │ INT16 GIOPS    │   4.1 @  6 chains    │   4.0 @  5 chains    │   2.0 @  3 chains    │   1.7 @  5 chains    │   1.0 @  3 chains    │   0.9 @  3 chains    │
// │ INT32 GIOPS    │   4.1 @  5 chains    │   4.0 @  5 chains    │   2.0 @  2 chains    │   1.7 @  3 chains    │   1.0 @  8 chains    │   0.9 @  3 chains    │
// │ INT64 GIOPS    │   1.0 @  2 chains    │   1.0 @  1 chain     │   0.5 @  1 chain     │   0.4 @  2 chains    │   0.3 @  1 chain     │   0.2 @  1 chain     │
// │ DRAM  GB/s     │   275 copy / 282 fill│   468 copy / 527 fill│   222 copy / 214 fill│   192 copy / 213 fill│   102 copy / 104 fill│    93 copy /  96 fill│
// │ L2    GB/s     │   729 copy @ 2MB     │  1307 copy @ 12MB    │   308 copy @ 6MB     │   470 copy @ 6MB     │   179 copy @ 2MB     │   217 copy @ 2MB     │
// │ L2 size        │  8-12 MB             │ 12-16 MB             │ 12-16 MB             │  8-12 MB             │   2-4 MB             │   2-4 MB             │
// ├────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
// │ FP32 MM TFLOPS │   7.1                │                      │   4.1                │                      │   3.1                │   1.4                │
// │ FP16 MM TFLOPS │  16.2                │                      │   7.5                │                      │   3.8                │   2.6                │
// │ BF16 MM TFLOPS │  16.2                │                      │   7.5                │                      │   3.8                │   1.3                │
// └────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

import Foundation
import Metal

// MARK: - Metal Shaders
let shaderSource = """
  #include <metal_stdlib>
  using namespace metal;

  // Templated fill kernel: each thread writes N scalar floats
  template<uint N>
  kernel void fill_bench(device float* dst [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
      uint base = gid * N;
      for (uint j = 0; j < N; j++) {
          dst[base + j] = 1.0f;
      }
  }

  template [[host_name("fill_1")]] kernel void fill_bench<1>(device float*, uint);
  template [[host_name("fill_2")]] kernel void fill_bench<2>(device float*, uint);
  template [[host_name("fill_3")]] kernel void fill_bench<3>(device float*, uint);
  template [[host_name("fill_4")]] kernel void fill_bench<4>(device float*, uint);
  template [[host_name("fill_5")]] kernel void fill_bench<5>(device float*, uint);
  template [[host_name("fill_6")]] kernel void fill_bench<6>(device float*, uint);
  template [[host_name("fill_7")]] kernel void fill_bench<7>(device float*, uint);
  template [[host_name("fill_8")]] kernel void fill_bench<8>(device float*, uint);

  // Templated copy kernel: each thread copies N scalar floats
  template<uint N>
  kernel void copy_bench(device const float* src [[buffer(0)]],
                         device float* dst       [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
      uint base = gid * N;
      metal::array<float, N> tmp;
      for (uint j = 0; j < N; j++) {
          tmp[j] = src[base + j];
      }
      for (uint j = 0; j < N; j++) {
          dst[base + j] = tmp[j];
      }
  }

  template [[host_name("copy_1")]] kernel void copy_bench<1>(device const float*, device float*, uint);
  template [[host_name("copy_2")]] kernel void copy_bench<2>(device const float*, device float*, uint);
  template [[host_name("copy_3")]] kernel void copy_bench<3>(device const float*, device float*, uint);
  template [[host_name("copy_4")]] kernel void copy_bench<4>(device const float*, device float*, uint);
  template [[host_name("copy_5")]] kernel void copy_bench<5>(device const float*, device float*, uint);
  template [[host_name("copy_6")]] kernel void copy_bench<6>(device const float*, device float*, uint);
  template [[host_name("copy_7")]] kernel void copy_bench<7>(device const float*, device float*, uint);
  template [[host_name("copy_8")]] kernel void copy_bench<8>(device const float*, device float*, uint);

  // Generic FMA kernel: works for float, half, bfloat
  // Each chain: c = fma(c, c, a) — 2 FLOPs per iteration
  template<typename T, uint N>
  kernel void fma_bench(
      device const T* input  [[buffer(0)]],
      device T*       output [[buffer(1)]],
      constant uint& iters   [[buffer(2)]],
      uint tid [[thread_position_in_grid]])
  {
      uint base = tid * N;
      T a = input[0];
      metal::array<T, N> c;
      for (uint j = 0; j < N; j++) {
          c[j] = input[base + j];
      }
      for (uint i = 0; i < iters; i++) {
          for (uint j = 0; j < N; j++) {
              c[j] = T(fma(c[j], c[j], a));
          }
      }
      for (uint j = 0; j < N; j++) {
          output[base + j] = c[j];
      }
  }

  #define FMA_F32(N) template [[host_name("fma_f32_" #N)]] \
      kernel void fma_bench<float, N>(device const float*, device float*, constant uint&, uint);
  FMA_F32(1) FMA_F32(2) FMA_F32(3) FMA_F32(4)
  FMA_F32(5) FMA_F32(6) FMA_F32(7) FMA_F32(8)
  FMA_F32(12) FMA_F32(16) FMA_F32(24) FMA_F32(32)

  #define FMA_F16(N) template [[host_name("fma_f16_" #N)]] \
      kernel void fma_bench<half, N>(device const half*, device half*, constant uint&, uint);
  FMA_F16(1)  FMA_F16(2)  FMA_F16(3)  FMA_F16(4)
  FMA_F16(5)  FMA_F16(6)  FMA_F16(7)  FMA_F16(8)
  FMA_F16(12) FMA_F16(16) FMA_F16(24) FMA_F16(32)

  #define FMA_BF16(N) template [[host_name("fma_bf16_" #N)]] \
      kernel void fma_bench<bfloat, N>(device const bfloat*, device bfloat*, constant uint&, uint);
  FMA_BF16(1)  FMA_BF16(2)  FMA_BF16(3)  FMA_BF16(4)
  FMA_BF16(5)  FMA_BF16(6)  FMA_BF16(7)  FMA_BF16(8)
  FMA_BF16(12) FMA_BF16(16) FMA_BF16(24) FMA_BF16(32)

  // Generic integer multiply-add: c = c * c + a — 2 IOPs per iteration
  template<typename T, uint N>
  kernel void imad_bench(
      device const T* input  [[buffer(0)]],
      device T*       output [[buffer(1)]],
      constant uint& iters   [[buffer(2)]],
      uint tid [[thread_position_in_grid]])
  {
      uint base = tid * N;
      T a = input[0];
      metal::array<T, N> c;
      for (uint j = 0; j < N; j++) {
          c[j] = input[base + j];
      }
      for (uint i = 0; i < iters; i++) {
          for (uint j = 0; j < N; j++) {
              c[j] = c[j] * c[j] + a;
          }
      }
      for (uint j = 0; j < N; j++) {
          output[base + j] = c[j];
      }
  }

  #define IMAD_I16(N) template [[host_name("imad_i16_" #N)]] \
      kernel void imad_bench<short, N>(device const short*, device short*, constant uint&, uint);
  IMAD_I16(1) IMAD_I16(2) IMAD_I16(3) IMAD_I16(4)
  IMAD_I16(5) IMAD_I16(6) IMAD_I16(7) IMAD_I16(8)

  #define IMAD_I32(N) template [[host_name("imad_i32_" #N)]] \
      kernel void imad_bench<int, N>(device const int*, device int*, constant uint&, uint);
  IMAD_I32(1) IMAD_I32(2) IMAD_I32(3) IMAD_I32(4)
  IMAD_I32(5) IMAD_I32(6) IMAD_I32(7) IMAD_I32(8)

  #define IMAD_I64(N) template [[host_name("imad_i64_" #N)]] \
      kernel void imad_bench<long, N>(device const long*, device long*, constant uint&, uint);
  IMAD_I64(1) IMAD_I64(2) IMAD_I64(3) IMAD_I64(4)
  IMAD_I64(5) IMAD_I64(6) IMAD_I64(7) IMAD_I64(8)
  """

// MARK: - Helpers

func makeDevice() -> (MTLDevice, MTLCommandQueue) {
  guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal not supported")
  }
  guard let queue = device.makeCommandQueue() else {
    fatalError("Cannot create command queue")
  }
  return (device, queue)
}

func makePipeline(_ device: MTLDevice, name: String, source: String) -> MTLComputePipelineState {
  let library = try! device.makeLibrary(source: source, options: nil)
  let function = library.makeFunction(name: name)!
  return try! device.makeComputePipelineState(function: function)
}

func measureGPUTime(
  _ queue: MTLCommandQueue, label: String,
  warmup: Int = 2, iterations: Int = 10,
  encode: (MTLComputeCommandEncoder) -> Void
) -> Double {
  for _ in 0..<warmup {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    encode(enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
  }
  var times: [Double] = []
  for _ in 0..<iterations {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    encode(enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    times.append(cb.gpuEndTime - cb.gpuStartTime)
  }
  times.sort()
  return times[times.count / 2]
}

// Measure throughput for a single kernel config, returns Gops/s
func measureKernelGops(
  _ device: MTLDevice, _ queue: MTLCommandQueue,
  kernelName: String, numThreads: Int, chains: Int,
  elemSize: Int, iters: UInt32, iterBuffer: MTLBuffer
) -> Double {
  let bufferBytes = numThreads * chains * elemSize
  let inputBuffer = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!
  let ptr = inputBuffer.contents().bindMemory(to: UInt8.self, capacity: bufferBytes)
  for i in 0..<bufferBytes { ptr[i] = 1 }
  let outputBuffer = device.makeBuffer(length: bufferBytes, options: .storageModeShared)!
  let pipeline = makePipeline(device, name: kernelName, source: shaderSource)
  let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)

  let time = measureGPUTime(queue, label: kernelName) { enc in
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(inputBuffer, offset: 0, index: 0)
    enc.setBuffer(outputBuffer, offset: 0, index: 1)
    enc.setBuffer(iterBuffer, offset: 0, index: 2)
    enc.dispatchThreads(
      MTLSize(width: numThreads, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
  }
  return Double(numThreads) * Double(chains) * Double(iters) * 2.0 / time / 1e9
}

// MARK: - Memory Bandwidth Benchmark

func benchmarkMemory(_ device: MTLDevice, _ queue: MTLCommandQueue) {
  print("=== Memory Bandwidth (GB/s) ===")
  let floatSize = MemoryLayout<Float>.stride

  var fillPipelines: [MTLComputePipelineState] = []
  var copyPipelines: [MTLComputePipelineState] = []
  for n in 1...8 {
    fillPipelines.append(makePipeline(device, name: "fill_\(n)", source: shaderSource))
    copyPipelines.append(makePipeline(device, name: "copy_\(n)", source: shaderSource))
  }

  let sizes: [Int] = [1, 4, 16, 64, 128]

  for (label, pipelines, scale): (String, [MTLComputePipelineState], Int) in [
    ("Fill (write)", fillPipelines, 1), ("Copy (r+w)", copyPipelines, 2),
  ] {
    var header = "\(label.padding(toLength: 12, withPad: " ", startingAt: 0)) "
    for n in 1...8 { header += String(format: " %5dx32b", n) }
    print(header)
    print(String(repeating: "-", count: header.count))

    for sizeMB in sizes {
      let numBytes = sizeMB * 1024 * 1024
      let src = device.makeBuffer(length: numBytes, options: .storageModeShared)!
      let dst = device.makeBuffer(length: numBytes, options: .storageModeShared)!

      var line = String(format: " %4d MB     ", sizeMB)
      for n in 1...8 {
        let numThreads = numBytes / (floatSize * n)
        let pipeline = pipelines[n - 1]
        let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let isCopy = scale == 2

        let time = measureGPUTime(queue, label: "mem") { enc in
          enc.setComputePipelineState(pipeline)
          if isCopy { enc.setBuffer(src, offset: 0, index: 0) }
          enc.setBuffer(dst, offset: 0, index: isCopy ? 1 : 0)
          enc.dispatchThreads(
            MTLSize(width: numThreads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        }
        let bw = Double(scale * numBytes) / time / 1e9
        line += String(format: " %8.1f", bw)
      }
      print(line)
    }
    print()
  }

  // Find best copy chain count at 64MB (DRAM-bound, avoids cache effects)
  var bestChain = 1
  var bestBW = 0.0
  let probeBytes = 64 * 1024 * 1024
  for n in 1...8 {
    let numThreads = probeBytes / (floatSize * n)
    let pipeline = copyPipelines[n - 1]
    let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    let probeSrc = device.makeBuffer(length: probeBytes, options: .storageModeShared)!
    let probeDst = device.makeBuffer(length: probeBytes, options: .storageModeShared)!
    let time = measureGPUTime(queue, label: "probe", warmup: 2, iterations: 5) { enc in
      enc.setComputePipelineState(pipeline)
      enc.setBuffer(probeSrc, offset: 0, index: 0)
      enc.setBuffer(probeDst, offset: 0, index: 1)
      enc.dispatchThreads(
        MTLSize(width: numThreads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
    }
    let bw = Double(2 * probeBytes) / time / 1e9
    if bw > bestBW { bestBW = bw; bestChain = n }
  }

  // Cache sweep using best chain count
  print("Cache sweep (copy, \(bestChain)×float, \(bestChain * 32)b/thread)")
  print("   Size      GB/s")
  print(String(repeating: "-", count: 22))
  let sizesKB: [Int] = [
    32, 64, 128, 256, 512, 768,
    1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12288, 16384, 24576, 32768, 65536, 131072, 262144,
  ]
  let cachePipeline = copyPipelines[bestChain - 1]
  for sizeKB in sizesKB {
    let numBytes = sizeKB * 1024
    let numThreads = numBytes / (floatSize * bestChain)
    let src = device.makeBuffer(length: numBytes, options: .storageModeShared)!
    let dst = device.makeBuffer(length: numBytes, options: .storageModeShared)!
    let tpg = min(cachePipeline.maxTotalThreadsPerThreadgroup, 256)
    let time = measureGPUTime(queue, label: "copy", warmup: 3, iterations: 20) { enc in
      enc.setComputePipelineState(cachePipeline)
      enc.setBuffer(src, offset: 0, index: 0)
      enc.setBuffer(dst, offset: 0, index: 1)
      enc.dispatchThreads(
        MTLSize(width: numThreads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
    }
    let bw = Double(2 * numBytes) / time / 1e9
    let sizeStr = sizeKB < 1024 ? "\(sizeKB) KB" : "\(sizeKB / 1024) MB"
    print(
      "\(sizeStr.padding(toLength: 8, withPad: " ", startingAt: 0)) \(String(format: "%8.1f", bw))")
  }
  print()
}

// MARK: - Compute Benchmark

func benchmarkCompute(_ device: MTLDevice, _ queue: MTLCommandQueue) {
  let numThreads = 1024 * 1024
  var iters: UInt32 = 4096
  let iterBuffer = device.makeBuffer(
    bytes: &iters, length: MemoryLayout<UInt32>.stride,
    options: .storageModeShared)!

  // FP throughput: unified table with columns for each type
  let fpChains = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
  let fpTypes: [(name: String, prefix: String, elemSize: Int)] = [
    ("FP32", "fma_f32", MemoryLayout<Float>.stride),
    ("FP16", "fma_f16", MemoryLayout<UInt16>.stride),
    ("BF16", "fma_bf16", MemoryLayout<UInt16>.stride),
  ]

  print("=== FP Compute — GFLOPS (FMA, \(numThreads/1024/1024)M threads, \(iters) iters) ===")
  var header = "Chains "
  for t in fpTypes { header += t.name.padding(toLength: 10, withPad: " ", startingAt: 0) }
  print(header)
  print(String(repeating: "-", count: header.count))

  for chains in fpChains {
    var line = String(format: "%-6d", chains)
    for t in fpTypes {
      let gops = measureKernelGops(
        device, queue,
        kernelName: "\(t.prefix)_\(chains)", numThreads: numThreads,
        chains: chains, elemSize: t.elemSize, iters: iters, iterBuffer: iterBuffer)
      line += String(format: " %8.0f ", gops)
    }
    print(line)
  }
  print()

  // INT throughput: unified table
  let intChains = [1, 2, 3, 4, 5, 6, 7, 8]
  let intTypes: [(name: String, prefix: String, elemSize: Int)] = [
    ("INT16", "imad_i16", MemoryLayout<Int16>.stride),
    ("INT32", "imad_i32", MemoryLayout<Int32>.stride),
    ("INT64", "imad_i64", MemoryLayout<Int64>.stride),
  ]

  print("=== INT Compute — GIOPS (IMAD, \(numThreads/1024/1024)M threads, \(iters) iters) ===")
  header = "Chains"
  for t in intTypes { header += t.name.padding(toLength: 10, withPad: " ", startingAt: 0) }
  print(header)
  print(String(repeating: "-", count: header.count))

  for chains in intChains {
    var line = String(format: "%-6d", chains)
    for t in intTypes {
      let gops = measureKernelGops(
        device, queue,
        kernelName: "\(t.prefix)_\(chains)", numThreads: numThreads,
        chains: chains, elemSize: t.elemSize, iters: iters, iterBuffer: iterBuffer)
      line += String(format: " %8.0f ", gops)
    }
    print(line)
  }
  print()
}

// MARK: - MPP GEMM Benchmark (macOS 15+)

// TN layout: A[K,M] row-major × B[K,N] row-major → C[M,N] row-major
//            Logical: C = A_logical^T @ B,  A_logical[M,K]
// Templated over element type so a single shader covers float, half, bfloat.
let mppShaderSource = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TILE_M = 64;
constant constexpr int TILE_N = 32;

template<typename T>
kernel void gemm_tn_typed(
    device T*       A      [[buffer(0)]],
    device T*       B      [[buffer(1)]],
    device T*       C      [[buffer(2)]],
    constant uint4& sizes  [[buffer(3)]],
    uint2 tgid             [[threadgroup_position_in_grid]])
{
    const uint M = sizes.x, K = sizes.y, N = sizes.z;
    device T* A_tile = A + tgid.y * TILE_M;
    device T* B_tile = B + tgid.x * TILE_N;
    device T* C_tile = C + tgid.y * TILE_M * N + tgid.x * TILE_N;

    // A[K,M] row-major: col-major [TILE_M,K], strides {1,M} → transpose_left=true
    tensor<device T, extents<int32_t, TILE_M, dynamic_extent>, tensor_inline> mA(
        A_tile, extents<int32_t, TILE_M, dynamic_extent>(K), array<int32_t, 2>{1, (int)M});
    // B[K,N] row-major: col-major [TILE_N,K], strides {1,N} → transpose_right=false
    tensor<device T, extents<int32_t, TILE_N, dynamic_extent>, tensor_inline> mB(
        B_tile, extents<int32_t, TILE_N, dynamic_extent>(K), array<int32_t, 2>{1, (int)N});
    // C[M,N] row-major tile: col-major [TILE_N,TILE_M], strides {1,N}
    tensor<device T, extents<int32_t, TILE_N, TILE_M>, tensor_inline> mC(
        C_tile, extents<int32_t, TILE_N, TILE_M>(), array<int32_t, 2>{1, (int)N});

    constexpr auto desc = matmul2d_descriptor(TILE_M, TILE_N,
        static_cast<int>(dynamic_extent), /*transpose_left=*/true, /*transpose_right=*/false);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(mA, mB, mC);
}

template [[host_name("gemm_tn_float")]]
kernel void gemm_tn_typed<float>(device float*, device float*, device float*, constant uint4&, uint2);
template [[host_name("gemm_tn_half")]]
kernel void gemm_tn_typed<half>(device half*, device half*, device half*, constant uint4&, uint2);
template [[host_name("gemm_tn_bfloat")]]
kernel void gemm_tn_typed<bfloat>(device bfloat*, device bfloat*, device bfloat*, constant uint4&, uint2);
"""

struct MPPTypeConfig {
  let name: String
  let elementSize: Int
  let pipeline: MTLComputePipelineState
}


func benchmarkMPP(_ device: MTLDevice, _ queue: MTLCommandQueue) {
  print("=== MPP matmul2d — TN layout (A[K,M]ᵀ @ B[K,N] → C[M,N]) ===")

  // Compile the MPP shader; bail gracefully if MPP is unavailable.
  let lib: MTLLibrary
  do {
    lib = try device.makeLibrary(source: mppShaderSource, options: nil)
  } catch {
    print("MPP shader compilation failed: \(error)")
    print("Skipping MPP benchmark.")
    print()
    return
  }

  let configs: [MPPTypeConfig] = ["float", "half", "bfloat"].compactMap { name in
    guard let fn = lib.makeFunction(name: "gemm_tn_\(name)"),
          let ps = try? device.makeComputePipelineState(function: fn)
    else { print("  \(name): pipeline creation failed, skipping"); return nil }
    return MPPTypeConfig(name: name, elementSize: name == "float" ? 4 : 2, pipeline: ps)
  }
  guard !configs.isEmpty else { print(); return }
  print("  SIMD width: \(configs[0].pipeline.threadExecutionWidth)")
  print()

  // Each entry: (section header or nil, M, N, K).  M must be ×64, N ×32.
  let sweep: [(section: String?, M: Int, N: Int, K: Int)] = [
    ("Square (scaling)",      128,  128,  128),
    (nil,                     512,  512,  512),
    (nil,                    1024, 1024, 1024),
    (nil,                    2048, 2048, 2048),
    (nil,                    4096, 4096, 4096),
    ("Vary M  (N=K=4096)",     64, 4096, 4096),
    (nil,                     128, 4096, 4096),
    (nil,                     256, 4096, 4096),
    (nil,                     512, 4096, 4096),
    (nil,                    1024, 4096, 4096),
    (nil,                    2048, 4096, 4096),
    (nil,                    4096, 4096, 4096),
    ("Vary K  (M=N=1024)",   1024, 1024,   64),
    (nil,                    1024, 1024,  256),
    (nil,                    1024, 1024, 1024),
    (nil,                    1024, 1024, 4096),
    ("Vary N  (M=K=512)",     512,   32,  512),
    (nil,                     512,  256,  512),
    (nil,                     512, 1024,  512),
    (nil,                     512, 4096,  512),
  ]

  let TILE_M = 64, TILE_N = 32
  let w = 7
  let typeColW = 10
  let hdr =
    "M".padding(toLength: w, withPad: " ", startingAt: 0) +
    "N".padding(toLength: w, withPad: " ", startingAt: 0) +
    "K".padding(toLength: w, withPad: " ", startingAt: 0) +
    configs.map { $0.name.padding(toLength: typeColW, withPad: " ", startingAt: 0) }.joined()
  let sep = String(repeating: "-", count: hdr.count)

  var peaks = [String: Double]()
  for c in configs { peaks[c.name] = 0.0 }

  for s in sweep {
    if let label = s.section {
      print()
      print("  \(label)")
      print("  " + hdr)
      print("  " + sep)
    }
    let (M, N, K) = (s.M, s.N, s.K)
    var line =
      "\(M)".padding(toLength: w, withPad: " ", startingAt: 0) +
      "\(N)".padding(toLength: w, withPad: " ", startingAt: 0) +
      "\(K)".padding(toLength: w, withPad: " ", startingAt: 0)

    for c in configs {
      let eSize = c.elementSize
      let bufA = device.makeBuffer(length: K * M * eSize, options: .storageModeShared)!
      let bufB = device.makeBuffer(length: K * N * eSize, options: .storageModeShared)!
      let bufC = device.makeBuffer(length: M * N * eSize, options: .storageModeShared)!
      memset(bufA.contents(), 0x3C, K * M * eSize)
      memset(bufB.contents(), 0x3C, K * N * eSize)
      var sizes = SIMD4<UInt32>(UInt32(M), UInt32(K), UInt32(N), 0)
      let sizeBuf = device.makeBuffer(
        bytes: &sizes, length: MemoryLayout<SIMD4<UInt32>>.stride,
        options: .storageModeShared)!

      let simdW = c.pipeline.threadExecutionWidth
      let numTGX = (N + TILE_N - 1) / TILE_N
      let numTGY = (M + TILE_M - 1) / TILE_M

      let time = measureGPUTime(queue, label: "mpp_\(c.name)") { enc in
        enc.setComputePipelineState(c.pipeline)
        enc.setBuffer(bufA,   offset: 0, index: 0)
        enc.setBuffer(bufB,   offset: 0, index: 1)
        enc.setBuffer(bufC,   offset: 0, index: 2)
        enc.setBuffer(sizeBuf, offset: 0, index: 3)
        enc.dispatchThreadgroups(
          MTLSize(width: numTGX, height: numTGY, depth: 1),
          threadsPerThreadgroup: MTLSize(width: simdW * 4, height: 1, depth: 1))
      }
      let tflops = Double(2 * M * N * K) / time / 1e12
      if tflops > peaks[c.name]! { peaks[c.name] = tflops }
      line += String(format: "%-\(typeColW).2f", tflops)
    }
    print("  " + line)
  }

  print()
  print("  Peak TFLOPS:")
  for c in configs {
    print("    \(c.name.padding(toLength: 8, withPad: " ", startingAt: 0))\(String(format: "%.2f", peaks[c.name]!))")
  }
  print()
}

// MARK: - Main

let (device, queue) = makeDevice()

print("Metal Roofline Benchmark")
print("Device: \(device.name)")
print("Max Buffer: \(device.maxBufferLength / (1024*1024)) MB")
print("Max Threadgroup Memory: \(device.maxThreadgroupMemoryLength) bytes")
print("Working Set: \(device.recommendedMaxWorkingSetSize / (1024*1024)) MB")
print()

// Quick occupancy check
let deviceMax = device.maxThreadsPerThreadgroup.width
let allKernels: [String] = {
  var names: [String] = []
  for n in 1...8 {
    for p in ["fill", "copy"] { names.append("\(p)_\(n)") }
    for p in ["imad_i16", "imad_i32", "imad_i64"] { names.append("\(p)_\(n)") }
  }
  for n in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32] {
    for p in ["fma_f32", "fma_f16", "fma_bf16"] { names.append("\(p)_\(n)") }
  }
  return names
}()
let limited = allKernels.filter {
  makePipeline(device, name: $0, source: shaderSource).maxTotalThreadsPerThreadgroup < deviceMax
}
if limited.isEmpty {
  print("All kernels: full occupancy (\(deviceMax) threads/group)")
} else {
  print("Limited occupancy: \(limited.joined(separator: ", "))")
}
print()

benchmarkMemory(device, queue)
benchmarkCompute(device, queue)

if #available(macOS 26.0, *) {
  benchmarkMPP(device, queue)
} else {
  print("=== MPP matmul2d ===")
  print("Requires macOS 26.0+ (MetalPerformancePrimitives not available), skipping.")
  print()
}
