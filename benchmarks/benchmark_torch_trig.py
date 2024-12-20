# Benchmark torch.sin + torch.cos performance on varios platforms/dtypes
# Against torch-2.5.0 for 4096x4096

import torch
from timeit import default_timer
from torch.utils.benchmark import Measurement, Timer

from torch._dynamo.device_interface import register_interface_for_device, DeviceInterface
from torch._inductor.codegen.common import register_backend_for_device, register_device_op_overrides, DeviceOpOverrides, Kernel, OpOverrides, CSEVariable, IndentedBuffer, DeferredLine
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.scheduler import BaseScheduling, Scheduler
from torch._inductor.codegen.simd import constant_repr, SIMDKernel, SIMDScheduling
from torch._inductor.utils import get_kernel_metadata
from torch._inductor.virtualized import V
from torch._inductor.ops_handler import StoreMode
import sympy

DTYPE_TO_METAL = {
   torch.float: "float",
   torch.half: "half",
   torch.bfloat16: "bfloat",
}

class MPSDeviceInterace(DeviceInterface):
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False):
        return torch.backends.mps.is_macos_or_newer(14, 0)

class MPSDeviceOpOverrides(DeviceOpOverrides):
    def device_guard(self, device_idx):
        assert device_idx == 0
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx):
        assert device_idx == 0
        return "# MPS set device"


class MPSOverrides(OpOverrides):
    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ):
        return f"mps.to_dtype(dtype)"

    @staticmethod
    def atan(x):
        return f"metal::atan({x})"

    @staticmethod
    def sin(x):
        return f"metal::sin({x})"

    @staticmethod
    def cos(x):
        return f"metal::cos({x})"


class MPSKernel(SIMDKernel):
    overrides = MPSOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        line = f"{var}[{index}]"
        return self.cse.generate(self.body, line)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        line = f"{var}[{index}] = static_cast<{dtype_str}>({value});"
        self.body.writeline(DeferredLine(name, line))

    def codegen_kernel(self, name=None):
        """Called at the end to generate a final kernel string"""
        code = IndentedBuffer()
        code.writeline('torch.mps._compile_shader("""')
        with code.indent():
           code.writeline("kernel void kernel_0(")
           with code.indent():
               for outer, inner in self.args.input_buffers.items():
                   dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                   code.writeline(f"constant {dtype_str}* {inner},")
               for outer, inner in self.args.output_buffers.items():
                   dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                   code.writeline(f"device {dtype_str}* {inner},")
               code.writeline("uint x0 [[thread_position_in_grid]]")
           code.writeline(") {")
           with code.indent():
               code.splice(self.body)
           code.writeline("}")
        code.writeline('""")')

        return code.getvalue()

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        wrapper.generate_kernel_call(
            name,
            self.args.python_argdefs()[1],
            gpu=False, # TODO: Fix me
            triton=False,
        )

class MPSScheduling(SIMDScheduling):
    kernel_type = MPSKernel

    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            kernel_name = f"mps_lib.kernel_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = kernel_name
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel("mps_lib", src_code, metadata_comment)

        return kernel_name


def bench_unary(
    m,
    n,
    unary_func,
    dtype=torch.float32,
    device: str = "cpu",
) -> Measurement:
    if device == "mps":
        sync_cmd = "torch.mps.synchronize()"
    elif device == "cuda":
        sync_cmd = "torch.cuda.synchronize()"
    else:
        sync_cmd = ""
    t = Timer(
        stmt=f"f(x);{sync_cmd}",
        setup=f"x=torch.rand(({m}, {n}), dtype={dtype}, device='{device}')",
        globals = {'f': unary_func},
        language="python", timer=default_timer
    )
    return t.blocked_autorange()



def run_bench_for_device(m, n, device, func, func_compiled):
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        # Validate correctness first
        inp = torch.rand(m, n, dtype=dtype, device=device)
        out = func(inp)
        out_compiled = func_compiled(inp)
        atol = 1e-2 if dtype is torch.bfloat16 else 1e-3 if dtype is torch.float16 else 1e-8
        if not torch.allclose(out, out_compiled, atol = atol):
            raise RuntimeError(f"out-out_compiled.abs().max() is {(out-out_compiled).abs().max().item()} for {dtype} and {device}")
        eager_t = bench_unary(m, n, func, dtype, device=device)
        comp_t = bench_unary(m, n, func_compiled, dtype, device=device)
        use_msec = eager_t.mean > 1e-4 or comp_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.sin+torch.cos({device}) {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname} {comp_t.mean*multiplier:>7.2f} {uname} {eager_t.mean/comp_t.mean:>7.2f}")

if __name__ == "__main__":
    def f(x):
        return torch.sin(x) + torch.cos(x)

    register_backend_for_device("mps", MPSScheduling, PythonWrapperCodegen, CppWrapperGpu)
    register_device_op_overrides("mps", MPSDeviceOpOverrides())
    register_interface_for_device("mps", MPSDeviceInterace)

    f_c=torch.compile(f)

    torch.set_num_threads(1)
    m, n = 8192, 16384
    # run_bench_for_device(m, n, "cpu", f, f_c)

    if torch.cuda.is_available():
        run_bench_for_device(m, n, "cuda", f, f_c)

    if torch.backends.mps.is_available():
        device = "mps"
        run_bench_for_device(m, n, "mps", f, f_c)
