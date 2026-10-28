# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for specialized decode.

Small-batch decode is bound by HBM bandwidth, and each kernel that starts on
weights not yet in L2 stalls on DRAM. Some kernels leave bandwidth idle, such
as the attention core and the all-reduces. At fixed points of a step, the
executor launches the prefetch kernel below on a side stream. It pulls the byte
ranges of the weights the next kernels read into L2, so that they start warm.

``PreparedWeightRanges`` owns its weights, an immutable table of (address,
bytes) ranges and the compiled kernel: the owner retains them across graph
replay. Activations and mutable caches may not be ranges before their producer
completes. The caller decides which weights to prefetch at which point, using
``prefetch_bulk``'s 16-byte-aligned ranges.
"""

import ctypes
import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from .host import _checked, fake, resolve_kernel, to_cute
from .primitives import launch, prefetch_bulk, thread


class L2Prefetch:
    """Each warp walks the device table, spreading 16-KiB chunks over the grid."""

    def __init__(self, grid, carveout):
        self.grid, self.carveout = grid, carveout

    @cute.jit
    def __call__(self, *args):
        options = dict(pdl=False, preferred_smem_carveout=self.carveout)
        launch(self.kernel, args, self.grid, 32, **options)

    @cute.kernel
    def kernel(self, args):
        (ranges,) = args
        cute.arch.griddepcontrol_launch_dependents()
        tid, _, _, block = thread()
        worker = cutlass.Int64(block * 32 + tid)
        workers = cutlass.Int64(self.grid * 32)
        phase, row = cutlass.Int64(0), cutlass.Int32(0)
        while row < ranges.shape[0]:
            base, nbytes = ranges[row, 0], ranges[row, 1]
            if (base != 0) & (nbytes > 0):
                chunks = (nbytes + 16384 - 1) // 16384
                index = worker - phase % workers
                if index < 0:
                    index += workers
                while index < chunks:
                    offset = index * 16384
                    size = nbytes - offset
                    if size > 16384:
                        size = cutlass.Int64(16384)
                    size = size & cutlass.Int64(-16)
                    if size > 0:
                        address = (base + offset) & cutlass.Int64(-16)
                        prefetch_bulk(address, size.to(cutlass.Int32))
                    index += workers
                phase += chunks
            row += 1
        # Early exit must not let a later grid bypass its producer.
        cute.arch.griddepcontrol_wait()


@functools.cache
def compile_kernel(count, grid, carveout, device_index):
    ranges = fake((count, 2), None, torch.int64)
    with torch.accelerator.device_index(device_index):
        return cute.compile(L2Prefetch(grid, carveout), ranges, make_fake_stream())


class PreparedWeightRanges:
    """One owner retains weights, immutable range table, executable and launch ABI."""

    def __init__(self, weights, *, byte_cap=0, grid=32, carveout=None, priority=False):
        self.weights = weights = tuple(weights)
        device = weights[0].device
        ranges, remaining = [], byte_cap
        for weight in weights:
            if weight.numel() == 0:
                continue
            if weight.data_ptr() % 16:  # the kernel walks 16-byte aligned chunks
                raise ValueError("L2 prefetch ranges must start 16-byte aligned")
            span = 1 + sum((n - 1) * s for n, s in zip(weight.shape, weight.stride()))
            size = span * weight.element_size()
            if byte_cap:
                size = min(size, remaining) // 16 * 16
                remaining -= size
            if size:
                ranges.append((weight.data_ptr(), size))
            if byte_cap and remaining < 16:
                break
        self.table = torch.tensor(ranges, dtype=torch.int64, device=device)
        self.compiled = compile_kernel(len(ranges), grid, carveout, device.index)
        self.config = None
        if not priority and carveout is None:
            self.views = to_cute(self.table, 16)
            return
        # Launch priority / carveout need a driver launch (CuTe drops both).
        self.executor, self.function = resolve_kernel(self.compiled, device.index)
        ids, self.attributes = cuda.CUlaunchAttributeID, []
        if priority:  # the greatest stream priority
            top = cuda.CUlaunchAttribute()
            top.id = ids.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
        config.sharedMemBytes, config.attrs = 0, self.attributes
        config.numAttrs = len(self.attributes)
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        self.address = ctypes.addressof(self.arguments)

    def launch(self, stream):
        if self.config is None:
            self.compiled(self.views, stream)
        else:
            self.config.hStream = stream
            _checked(cuda.cuLaunchKernelEx(self.config, self.function, self.address, 0))
