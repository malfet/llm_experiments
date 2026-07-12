// Metal compute-overlap probe: can two command buffers run concurrently on the GPU?
//
// The question is whether two independent command buffers, committed to two
// separate command queues, actually execute at the same time on the GPU — or
// whether the hardware serializes them regardless. One compute-heavy kernel (a
// dependent FMA chain — pure ALU, no memory traffic) is launched three ways, on
// its own buffers each time:
//   (a) single : one command buffer, one dispatch of N threads, on one queue.
//   (b) concur : two command buffers (one per queue), each a dispatch of N
//                threads, committed together and wall-clocked until both finish.
//   (c) serial : the same two dispatches encoded back-to-back in ONE command
//                buffer — a guaranteed-serial reference point.
//
// Reading the result:
//   serial / concur  > 1  ⇒ the two command buffers ran concurrently (their
//                           combined time beat forced serialization).
//   concur / single ≈ 1   ⇒ the second command buffer ran essentially for free
//                           (spare GPU capacity); ≈ 2 ⇒ no concurrency, N threads
//                           already saturate the GPU.
// Sweeping N shows the concurrency benefit shrinking as each command buffer
// approaches saturation on its own.
//
// Build: swiftc -O -o metal_compute_overlap metal_compute_overlap.swift -framework Metal -framework Foundation
// Run:   ./metal_compute_overlap

import Foundation
import Metal

let shaderSource = """
  #include <metal_stdlib>
  using namespace metal;

  kernel void spin(device const float* input  [[buffer(0)]],
                   device float*       output [[buffer(1)]],
                   constant uint&      iters  [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
      float a = input[gid];
      float c = a;
      for (uint i = 0; i < iters; i++) {
          c = fma(c, c, a);   // dependent chain -> not optimized away
      }
      output[gid] = c;
  }
  """

func makePipeline(_ device: MTLDevice) -> MTLComputePipelineState {
  let library = try! device.makeLibrary(source: shaderSource, options: nil)
  let function = library.makeFunction(name: "spin")!
  return try! device.makeComputePipelineState(function: function)
}

func makeFilledBuffer(_ device: MTLDevice, count: Int) -> MTLBuffer {
  let bytes = count * MemoryLayout<Float>.stride
  let buf = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)!
  let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
  for i in 0..<count { ptr[i] = 1.0000001 }
  return buf
}

// Median wall-clock seconds over `reps` runs of `body`.
func medianWall(reps: Int = 15, _ body: () -> Void) -> Double {
  var t: [Double] = []
  for _ in 0..<reps {
    let start = DispatchTime.now().uptimeNanoseconds
    body()
    t.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
  }
  t.sort()
  return t[t.count / 2]
}

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal not supported") }
let queue0 = device.makeCommandQueue()!
let queue1 = device.makeCommandQueue()!   // second independent stream
let pipeline = makePipeline(device)
let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, 256)

var iters: UInt32 = 200_000
let iterBuffer = device.makeBuffer(
  bytes: &iters, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

func encode(_ enc: MTLComputeCommandEncoder, _ inBuf: MTLBuffer, _ outBuf: MTLBuffer, _ n: Int) {
  enc.setComputePipelineState(pipeline)
  enc.setBuffer(inBuf, offset: 0, index: 0)
  enc.setBuffer(outBuf, offset: 0, index: 1)
  enc.setBuffer(iterBuffer, offset: 0, index: 2)
  enc.dispatchThreads(
    MTLSize(width: n, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
}

print("Metal Compute-Overlap Probe")
print("Device: \(device.name)")
print("Per-thread work: \(iters) FMA iterations, threadgroup \(tpg)")
print()
print(" N/stream    single     concur     serial    serial/concur   concur/single")
print(String(repeating: "-", count: 74))

// MARK: - Sweep per-stream thread count

var n = 256
while n <= 1 << 16 {   // 256 .. 65536 threads per stream
  let inA = makeFilledBuffer(device, count: n)
  let outA = device.makeBuffer(length: n * MemoryLayout<Float>.stride, options: .storageModeShared)!
  let inB = makeFilledBuffer(device, count: n)
  let outB = device.makeBuffer(length: n * MemoryLayout<Float>.stride, options: .storageModeShared)!

  func runSingle() {
    let cb = queue0.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    encode(e, inA, outA, n)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
  }
  func runConcur() {
    let cb0 = queue0.makeCommandBuffer()!
    let e0 = cb0.makeComputeCommandEncoder()!
    encode(e0, inA, outA, n)
    e0.endEncoding()
    let cb1 = queue1.makeCommandBuffer()!
    let e1 = cb1.makeComputeCommandEncoder()!
    encode(e1, inB, outB, n)
    e1.endEncoding()
    cb0.commit()
    cb1.commit()
    cb0.waitUntilCompleted()
    cb1.waitUntilCompleted()
  }
  func runSerial() {
    let cb = queue0.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    encode(e, inA, outA, n)
    encode(e, inB, outB, n)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
  }

  runConcur()  // warm up both queues
  let single = medianWall { runSingle() }
  let concur = medianWall { runConcur() }
  let serial = medianWall { runSerial() }

  print(String(
    format: " %8d  %8.3f   %8.3f   %8.3f      %6.2fx        %6.2fx",
    n, single * 1e3, concur * 1e3, serial * 1e3, serial / concur, concur / single))
  n *= 2
}
print()
print("serial/concur > 1  ⇒ command buffers ran concurrently;  concur/single ≈ 1 ⇒ 2nd ~free")
