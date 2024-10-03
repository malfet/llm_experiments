#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>

static at::native::mps::MetalShaderLibrary lib(R"SUM_SIN_COS(
#include <metal_stdlib>
using namespace metal;
template<typename T>
kernel void sum_sincos(constant T* x,
                       device   T* out,
                       uint index [[thread_position_in_grid]])
{
    out[index] = static_cast<T>(sin(x[index]) + cos(x[index]));
}

template [[host_name("sum_sincos_float")]] kernel void sum_sincos(constant float*, device float*, uint);
template [[host_name("sum_sincos_half")]] kernel void sum_sincos(constant half*, device half*, uint);
template [[host_name("sum_sincos_bfloat")]] kernel void sum_sincos(constant bfloat*, device bfloat*, uint);
)SUM_SIN_COS");

at::Tensor mps_sum_sincos(at::Tensor &in) {

  TORCH_CHECK(in.is_mps());
  using namespace at::native::mps;
  auto out = at::empty_like(in);

  @autoreleasepool {
    auto kernelPSO = lib.getPipelineStateForFunc("sum_sincos_" + scalarToMetalTypeString(in));
    MPSStream* mpsStream = getCurrentMPSStream();

    dispatch_sync(mpsStream->queue(), ^() {
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState: kernelPSO];
      mtl_setBuffer(computeEncoder, in, 0);
      mtl_setBuffer(computeEncoder, out, 1);
      mtl_dispatch1DJob(computeEncoder, kernelPSO, in.numel());
    });
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mps_sum_sincos", &mps_sum_sincos);
}
