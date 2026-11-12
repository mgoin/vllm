// Weight-streaming grouped MoE GEMM for Rubin (SM107) LUT-B weights with
// MXFP8 activations.
//
// C[sid, n] = sum_k A[sid / top_k, k] * W[expert, n, k] (* topk_weight[sid])
//
// Same operands as lut_b_moe_sm107.cu, organized so the weight stream sets
// the speed when few tokens hit each expert:
//  * Persistent CTAs (one per SM) walk a static list of work items, each one
//    (32-row routing block, 256-column N tile, K split), and every role
//    prefetches the next item's routing. Routing uses blocks of 32 rows
//    (moe_align_block_size(..., 32)); the MMA is still M128, and its unused
//    rows are never stored.
//  * Warp-specialized: lane 0 of warp 0 streams weights with cp.async.bulk,
//    lane 0 of warp 1 issues MMAs, warp 2 stages the real activation rows
//    with cp.async, warps 3-6 run the epilogue, and warps 7-10 write each
//    stage's scales and codebooks into its own TMEM slots (tcgen05.st).
//  * A stage holds two K128 records: every stage costs each role a few
//    barrier waits, fences and a commit, which bounded one-record stages.
//  * Blocks with at most 8 real rows stage one 8-row core-matrix group of A
//    and point every M group of the descriptor at it (SBO = 0), so A costs
//    1 KB of shared memory per K128, and all four epilogue warps can drain
//    the accumulator (each lane quarter holds the same rows).
//  * Optional split-K (off by default) with an in-kernel reduction: each
//    split writes FP32 partials, and
//    the last split to finish a tile (per-tile counter) sums them and writes
//    the output. Counters are reset by the kernel, so they stay zero between
//    launches.
//
// Weights use the paired SM107 layout from `lut_b_sm107_pair_tiles`: per
// (expert, 256 N, 128 K) one 13824 B record that is the stage image,
//   [idx 0][idx 1][scales 0][scales 1][books 0][books 1]
// where 0 and 1 are the two 128-column halves in the lut_b_moe_sm107.cu tile
// format. Records are contiguous along K, so one cp.async.bulk fills a stage.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

namespace vllm {
namespace lut_b_moe_stream {

constexpr int kBlockM = 128;  // MMA M
constexpr int kRows = 32;     // routed rows per block (moe_align_block_size)
constexpr int kBlockN = 256;  // MMA N
constexpr int kTileN = 128;   // N of one global weight tile
constexpr int kBlockK = 128;

constexpr int kProducerWarp = 0;  // weights (lane 0)
constexpr int kMmaWarp = 1;       // MMAs (lane 0)
constexpr int kALoaderWarp = 2;   // activations
constexpr int kFirstEpilogueWarp = 3;
constexpr int kEpilogueThreads = 128;
// Four warps (one per TMEM lane quarter) write each stage's scales and
// codebooks into its own TMEM metadata slot with tcgen05.st, so neither the
// MMA thread nor the epilogue waits on metadata.
constexpr int kFirstMetaWarp = kFirstEpilogueWarp + kEpilogueThreads / 32;
constexpr int kMetaThreads = 128;
constexpr int kThreads = 96 + kEpilogueThreads + kMetaThreads;

// Global weight tile (one 128 x 128 block).
constexpr int kTileIdxBytes = kTileN * kBlockK * 3 / 8;  // 6144
constexpr int kTileScaleBytes = 512;
constexpr int kTileBookBytes = kTileN / 8 * 16;  // 256
constexpr int kTileBytes = kTileIdxBytes + kTileScaleBytes + kTileBookBytes;

// N256 weights within a shared-memory stage, after A and the A scales.
constexpr int kBIdxOff = 0;
constexpr int kBScaleOff = 2 * kTileIdxBytes;                // 12288
constexpr int kBBookOff = kBScaleOff + 2 * kTileScaleBytes;  // 13312
constexpr int kBStageBytes = kBBookOff + 2 * kTileBookBytes;  // 13824
constexpr int kAScaleBytes = 512;

// Leaves room for static shared memory under the 227 KB per-block limit.
constexpr int kSmemBudget = 227 * 1024 - 4096;

// K128 records per pipeline stage. Every stage costs the same handful of
// barrier waits, fences and a commit on each role's loop, which bounded the
// MMA thread at about 340 ns per stage; two records per stage halve that per
// byte.
constexpr int kRecs = 2;

// Stage: [A x kRecs][A scales x kRecs][weights x kRecs], where one A tile is
// kAGroups 8-row groups x K128. A block of up to 32 rows needs 4 A groups;
// the M128 MMA also reads groups 4-15, which land on the rest of the stage
// and only feed accumulator rows that are never stored. The weight records
// are contiguous so one bulk copy fills them.
template <int kAGroups>
struct Config {
  static constexpr int kABytes = kAGroups * 1024;
  static constexpr int kAOff = 0;
  static constexpr int kAScaleOff = kRecs * kABytes;
  static constexpr int kBOff = kAScaleOff + kRecs * kAScaleBytes;
  static constexpr int kStageBytes =
      (kBOff + kRecs * kBStageBytes + 1023) / 1024 * 1024;
  static constexpr int kStages = kSmemBudget / kStageBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes + 1024;
  static_assert(kBOff % 16 == 0);
  static_assert(kAGroups == 1 ||
                kStageBytes - (kRecs - 1) * kABytes >= kBlockM / 8 * 1024);
  static_assert(kAGroups * 8 <= kRows);
  static_assert(kStages >= 4);
};

// TMEM: 256 accumulator columns, then one 16-column metadata slot per record
// of each stage: A scales (4 columns), B scales (N256 = 8 columns),
// codebooks (4 columns).
constexpr int kTmemCols = 512;
constexpr int kMetaCols = 16;
constexpr int kMetaAScale = 0;
constexpr int kMetaBScale = 4;
constexpr int kMetaBook = 12;

constexpr int kMaxSplits = 16;

__device__ __forceinline__ uint32_t smem_addr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Shared memory matrix descriptor, no swizzle.
__device__ __forceinline__ uint64_t smem_desc(void const* p, int lbo, int sbo) {
  return uint64_t(smem_addr(p) >> 4) | (uint64_t(lbo >> 4) << 16) |
         (uint64_t(sbo >> 4) << 32) | (uint64_t(1) << 46);
}

__device__ __forceinline__ void cp_async_16(void* dst, void const* src,
                                            bool valid) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)),
      "l"(src), "r"(valid ? 16 : 0)
      : "memory");
}

__device__ __forceinline__ void cp_async_4(void* dst, void const* src,
                                           bool valid) {
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_addr(dst)),
      "l"(src), "r"(valid ? 4 : 0)
      : "memory");
}

// Arrive on `bar` once all of this thread's earlier cp.async copies land.
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile(
      "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
          smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void bulk_copy(void* dst, void const* src,
                                          int bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(smem_addr(dst)),
      "l"(src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar,
                                                      int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
          smem_addr(bar)),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar))
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
  asm volatile(
      "{.reg .pred p; LAB_WAIT: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@!p bra LAB_WAIT;}" ::"r"(smem_addr(bar)),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(
          smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void mma_lut_fill(uint32_t d, uint64_t a, uint64_t b,
                                             uint32_t book, uint32_t idesc,
                                             uint32_t sfa, uint32_t sfb,
                                             uint32_t accumulate) {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p, %7, 0; "
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b."
      "block32.collector::b::fill [%0], %1, %2, [%3], %4, [%5], [%6], p;}" ::
          "r"(d),
      "l"(a), "l"(b), "r"(book), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate)
      : "memory");
}

__device__ __forceinline__ void mma_lut_lastuse(uint32_t d, uint64_t a,
                                                uint64_t b, uint32_t book,
                                                uint32_t idesc, uint32_t sfa,
                                                uint32_t sfb) {
  asm volatile(
      "{.reg .pred p; setp.ne.b32 p, %7, 0; "
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b."
      "block32.collector::b::lastuse [%0], %1, %2, [%3], %4, [%5], [%6], p;}" ::
          "r"(d),
      "l"(a), "l"(b), "r"(book), "r"(idesc), "r"(sfa), "r"(sfb), "r"(1u)
      : "memory");
}

__device__ __forceinline__ void tmem_ld32(uint32_t addr, uint32_t (&v)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,"
      "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
      "%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]),
        "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]),
        "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
        "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]),
        "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]),
        "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]),
        "=r"(v[31])
      : "r"(addr)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Write 16 columns of this warp's lane quarter (complete after
// tmem_wait_st).
__device__ __forceinline__ void tmem_st16(uint32_t addr, uint4 const (&v)[4]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
      "%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "r"(v[0].x), "r"(v[0].y), "r"(v[0].z), "r"(v[0].w), "r"(v[1].x),
      "r"(v[1].y), "r"(v[1].z), "r"(v[1].w), "r"(v[2].x), "r"(v[2].y),
      "r"(v[2].z), "r"(v[2].w), "r"(v[3].x), "r"(v[3].y), "r"(v[3].z),
      "r"(v[3].w)
      : "memory");
}

__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void epilogue_bar() {
  asm volatile("bar.sync 1, %0;" ::"n"(kEpilogueThreads) : "memory");
}

template <typename OutT>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<OutT, __nv_bfloat16>) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  } else {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
  }
}

template <typename OutT>
__device__ __forceinline__ void store8(OutT* dst, float const* v, float w) {
  uint4 packed;
  packed.x = pack2<OutT>(v[0] * w, v[1] * w);
  packed.y = pack2<OutT>(v[2] * w, v[3] * w);
  packed.z = pack2<OutT>(v[4] * w, v[5] * w);
  packed.w = pack2<OutT>(v[6] * w, v[7] * w);
  *reinterpret_cast<uint4*>(dst) = packed;
}

struct Params {
  uint8_t const* a;
  uint8_t const* a_scales;
  uint8_t const* b;
  void* c;
  float* partials;    // [splits, num_valid_tokens, N] when splits > 1
  int32_t* counters;  // [num_m_blocks * n_tiles], zero between launches
  float const* topk_weights;
  int32_t const* sorted_token_ids;
  int32_t const* expert_ids;
  int32_t const* num_tokens_post_padded;
  int num_valid_tokens;
  int num_sorted;  // sorted_token_ids.size(0)
  int top_k;
  int N;
  int K;
  int splits;
};

// Work item i -> (M block, N256 tile, K split); splits are innermost so the
// CTAs that share a tile run at the same time.
struct Item {
  int m_block, n_tile, split, k_begin, k_end;
};

__device__ __forceinline__ Item decode_item(int i, int n_tiles, int splits,
                                            int k_tiles) {
  Item it;
  it.split = i % splits;
  int const t = i / splits;
  it.n_tile = t % n_tiles;
  it.m_block = t / n_tiles;
  it.k_begin = it.split * k_tiles / splits;
  it.k_end = (it.split + 1) * k_tiles / splits;
  return it;
}

template <typename OutT, int kAGroups>
__global__ void __launch_bounds__(kThreads, 1)
    lut_b_moe_stream_kernel(Params const p) {
  using Cfg = Config<kAGroups>;
  constexpr int kStages = Cfg::kStages;
  static_assert(kBlockN + kStages * kRecs * kMetaCols <= kTmemCols);

  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  __shared__ __align__(8) uint64_t full_bar[kStages];
  __shared__ __align__(8) uint64_t empty_bar[kStages];
  __shared__ __align__(8) uint64_t meta_bar[kStages];
  __shared__ __align__(8) uint64_t acc_full_bar;
  __shared__ __align__(8) uint64_t acc_empty_bar;
  __shared__ __align__(8) uint64_t epi_done_bar;
  __shared__ uint32_t tmem_base;
  __shared__ int last_split;
  __shared__ int32_t epi_sid[kBlockM];
  __shared__ float epi_weight[kBlockM];
  __shared__ int epi_rows[4];

  int const warp = threadIdx.x / 32;
  int const lane = threadIdx.x % 32;

  int const n_tiles = p.N / kBlockN;
  int const k_tiles = p.K / kBlockK;
  // The first item's M block does not depend on the routing length, so its
  // expert id is loaded alongside num_tokens_post_padded rather than after.
  Item const first = decode_item(blockIdx.x, n_tiles, p.splits, k_tiles);
  int const first_expert = p.expert_ids[first.m_block];
  int const num_items =
      (*p.num_tokens_post_padded / kRows) * n_tiles * p.splits;
  auto expert_of = [&](int i) {
    return i < num_items
               ? p.expert_ids[decode_item(i, n_tiles, p.splits, k_tiles)
                                  .m_block]
               : -1;
  };

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      // One expect_tx arrival plus one cp.async arrival per A-loader lane.
      mbar_init(&full_bar[s], 1 + 32);
      mbar_init(&empty_bar[s], 1);
      mbar_init(&meta_bar[s], kMetaThreads);
    }
    mbar_init(&acc_full_bar, 1);
    mbar_init(&acc_empty_bar, kEpilogueThreads);
    mbar_init(&epi_done_bar, kEpilogueThreads);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == kMmaWarp) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(smem_addr(&tmem_base)),
        "n"(kTmemCols)
        : "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  uint32_t const tmem = tmem_base;

  // Every role walks the same items and the same ring of stages; a stage
  // holds up to kRecs consecutive K128 records of one item.
  if (warp == kProducerWarp) {
    // -------------------------------------------------------------- weights
    if (lane == 0) {
      int stage = 0;
      uint32_t phase = 0;
      int expert_next = first_expert;
      for (int i = blockIdx.x; i < num_items; i += gridDim.x) {
        Item const it = decode_item(i, n_tiles, p.splits, k_tiles);
        int const expert = expert_next;
        expert_next = expert_of(i + gridDim.x);
        if (expert < 0) continue;
        // Paired records are the stage's weight image and are contiguous
        // along K, so one bulk copy fills a stage.
        uint8_t const* src =
            p.b + ((int64_t(expert) * n_tiles + it.n_tile) * k_tiles +
                   it.k_begin) *
                      kBStageBytes;
        for (int k = it.k_begin; k < it.k_end; k += kRecs) {
          int const bytes = min(kRecs, it.k_end - k) * kBStageBytes;
          mbar_wait(&empty_bar[stage], phase ^ 1);
          mbar_arrive_expect_tx(&full_bar[stage], bytes);
          bulk_copy(smem + stage * Cfg::kStageBytes + Cfg::kBOff, src, bytes,
                    &full_bar[stage]);
          src += bytes;
          if (++stage == kStages) stage = 0, phase ^= 1;
        }
      }
    }
    __syncwarp();
  } else if (warp == kALoaderWarp) {
    // ---------------------------------------------------------- activations
    int stage = 0;
    uint32_t phase = 0;
    int expert_next = first_expert;
    // The first block may be the array's last, partial one.
    int const first_row = first.m_block * kRows + lane;
    int32_t sid_next = first_row < p.num_sorted
                           ? p.sorted_token_ids[first_row]
                           : p.num_valid_tokens;
    for (int i = blockIdx.x; i < num_items; i += gridDim.x) {
      Item const it = decode_item(i, n_tiles, p.splits, k_tiles);
      int const expert = expert_next;
      int32_t const sid = sid_next;
      int const i_next = i + gridDim.x;
      expert_next = expert_of(i_next);
      if (i_next < num_items)
        sid_next = p.sorted_token_ids[decode_item(i_next, n_tiles, p.splits,
                                                  k_tiles)
                                              .m_block *
                                          kRows +
                                      lane];
      if (expert < 0) continue;
      // Real rows are a prefix of the block; lane r owns row r.
      bool const valid = sid < p.num_valid_tokens;
      int32_t const token = valid ? sid / p.top_k : -1;
      int const count = __popc(__ballot_sync(0xffffffffu, valid));
      int groups = count <= 8 ? 1 : kAGroups;

      // Per item, not per record: each lane copies 16-byte chunk lane % 8
      // of rows lane / 8 + 4j into the canonical K-major no-swizzle layout
      // (8-row x 16-byte core matrices, 128 bytes apart along K and 1024
      // bytes apart along M), so a record costs groups * 2 copies per lane.
      constexpr int kRowsPerLane = kAGroups * 2;
      int const chunk = lane % 8;
      uint8_t const* a_src[kRowsPerLane];
      int a_dst[kRowsPerLane];
      bool a_ok[kRowsPerLane];
#pragma unroll
      for (int j = 0; j < kRowsPerLane; ++j) {
        int const r = lane / 8 + 4 * j;
        int32_t const t = __shfl_sync(0xffffffffu, token, r);
        a_ok[j] = t >= 0;
        a_src[j] = p.a + int64_t(max(t, 0)) * p.K + chunk * 16;
        a_dst[j] = (r / 8) * 1024 + (r % 8) * 16 + chunk * 128;
      }
      // A scales: the MMA reads accumulator row R's scale word from record
      // R % 32, word R / 32. A small block (SBO 0) repeats rows 0-7 in every
      // lane quarter, so their words fill all four: lane l copies row l / 4
      // into word l % 4. Otherwise lane r copies row r into word 0.
      int32_t const s_token =
          groups == 1 ? __shfl_sync(0xffffffffu, token, lane / 4) : token;
      bool const s_copy = groups == 1 || (groups > 1 && lane < groups * 8);
      uint8_t const* const s_src =
          p.a_scales + int64_t(max(s_token, 0)) * (p.K / 32);
      int const s_dst = groups == 1 ? (lane / 4) * 16 + (lane % 4) * 4
                                    : lane * 16;

      for (int k = it.k_begin; k < it.k_end; k += kRecs) {
        int const recs = min(kRecs, it.k_end - k);
        mbar_wait(&empty_bar[stage], phase ^ 1);
        uint8_t* st = smem + stage * Cfg::kStageBytes;
        for (int r = 0; r < recs; ++r) {
          uint8_t* a_st = st + Cfg::kAOff + r * Cfg::kABytes;
          int const k_off = (k + r) * kBlockK;
#pragma unroll
          for (int j = 0; j < kRowsPerLane; ++j)
            if (j < groups * 2)
              cp_async_16(a_st + a_dst[j], a_src[j] + k_off, a_ok[j]);
          if (s_copy)
            cp_async_4(st + Cfg::kAScaleOff + r * kAScaleBytes + s_dst,
                       s_src + (k + r) * 4, s_token >= 0);
        }
        cp_async_arrive(&full_bar[stage]);
        if (++stage == kStages) stage = 0, phase ^= 1;
      }
    }
  } else if (warp == kMmaWarp) {
    // ----------------------------------------------------------------- MMAs
    if (lane == 0) {
      // E4M3 A and decoded B, FP32 accumulate, UE8M0 block32 scales,
      // M = 128, N = 256.
      uint32_t const idesc = (uint32_t(kBlockN / 8) << 17) | (1u << 23) |
                             (uint32_t(kBlockM / 16) << 24) | (1u << 31);
      // The second K64 half uses scale byte 2 of each 4-byte A/B scale word.
      uint32_t const idesc_hi = idesc | (2u << 29) | (2u << 4);
      int stage = 0, local = 0;
      uint32_t phase = 0;
      int expert_next = first_expert;
      int const first_row8 = first.m_block * kRows + 8;
      int32_t sid8_next = first_row8 < p.num_sorted
                              ? p.sorted_token_ids[first_row8]
                              : p.num_valid_tokens;
      for (int i = blockIdx.x; i < num_items; i += gridDim.x) {
        Item const it = decode_item(i, n_tiles, p.splits, k_tiles);
        int const expert = expert_next;
        int32_t const sid8 = sid8_next;
        int const i_next = i + gridDim.x;
        expert_next = expert_of(i_next);
        if (i_next < num_items)
          sid8_next = p.sorted_token_ids
              [decode_item(i_next, n_tiles, p.splits, k_tiles).m_block * kRows +
               8];
        if (expert < 0) continue;
        // With at most 8 real rows, SBO 0 points every M group at group 0,
        // so each TMEM lane quarter holds rows 0-7. (SBO 256, which would
        // spread larger blocks over the quarters, made the MMAs slower.)
        int const a_sbo = sid8 >= p.num_valid_tokens ? 0 : 1024;

        mbar_wait(&acc_empty_bar, (local & 1) ^ 1);
        tc_fence_after();
        for (int k = it.k_begin; k < it.k_end; k += kRecs) {
          int const recs = min(kRecs, it.k_end - k);
          // Metadata warps arrive once the stage has landed and its TMEM
          // slots are written.
          mbar_wait(&meta_bar[stage], phase);
          tc_fence_after();
          // Activations arrived through cp.async (generic proxy).
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          uint8_t* st = smem + stage * Cfg::kStageBytes;
          for (int r = 0; r < recs; ++r) {
            uint32_t const meta =
                tmem + kBlockN + (stage * kRecs + r) * kMetaCols;
            uint8_t* a_st = st + Cfg::kAOff + r * Cfg::kABytes;
            uint64_t const a_lo = smem_desc(a_st, 128, a_sbo);
            uint64_t const a_hi = smem_desc(a_st + 512, 128, a_sbo);
            // Compressed B rows are 48 bytes; descriptor bit 53 selects the
            // second K64 half of each row.
            uint64_t const b_lo = smem_desc(
                st + Cfg::kBOff + r * kBStageBytes + kBIdxOff, 128, 384);
            uint64_t const b_hi = b_lo | (uint64_t(1) << 53);
            mma_lut_fill(tmem, a_lo, b_lo, meta + kMetaBook, idesc,
                         meta + kMetaAScale, meta + kMetaBScale,
                         k + r != it.k_begin);
            mma_lut_lastuse(tmem, a_hi, b_hi, meta + kMetaBook + 2, idesc_hi,
                            meta + kMetaAScale, meta + kMetaBScale);
          }
          tc_commit(&empty_bar[stage]);
          if (++stage == kStages) stage = 0, phase ^= 1;
        }
        tc_commit(&acc_full_bar);
        ++local;
      }
    }
    __syncwarp();
  } else if (warp >= kFirstMetaWarp) {
    // -------------------------------------------------- TMEM metadata slots
    // Each warp writes its lane quarter, which gives the replication across
    // all four quarters that tcgen05.cp.32x128b.warpx4 would.
    uint32_t const lane_base = tmem + (uint32_t((warp % 4) * 32) << 16);
    int stage = 0;
    uint32_t phase = 0;
    int expert_next = first_expert;
    for (int i = blockIdx.x; i < num_items; i += gridDim.x) {
      Item const it = decode_item(i, n_tiles, p.splits, k_tiles);
      int const expert = expert_next;
      expert_next = expert_of(i + gridDim.x);
      if (expert < 0) continue;
      for (int k = it.k_begin; k < it.k_end; k += kRecs) {
        int const recs = min(kRecs, it.k_end - k);
        mbar_wait(&full_bar[stage], phase);
        uint8_t const* st = smem + stage * Cfg::kStageBytes;
        for (int r = 0; r < recs; ++r) {
          uint8_t const* b_st = st + Cfg::kBOff + r * kBStageBytes;
          uint4 v[4];
          v[0] = reinterpret_cast<uint4 const*>(st + Cfg::kAScaleOff +
                                                r * kAScaleBytes)[lane];
          v[1] = reinterpret_cast<uint4 const*>(b_st + kBScaleOff)[lane];
          v[2] = reinterpret_cast<uint4 const*>(b_st + kBScaleOff +
                                                kTileScaleBytes)[lane];
          v[3] = reinterpret_cast<uint4 const*>(b_st + kBBookOff)[lane];
          tmem_st16(lane_base + kBlockN + (stage * kRecs + r) * kMetaCols, v);
        }
        tmem_wait_st();
        tc_fence_before();
        mbar_arrive(&meta_bar[stage]);
        if (++stage == kStages) stage = 0, phase ^= 1;
      }
    }
  } else {
    // ------------------------------------------------------------- epilogue
    int const quarter = warp % 4;  // TMEM lanes this warp may access
    uint32_t const lane_addr = tmem + (uint32_t(quarter * 32) << 16);
    OutT* const c = static_cast<OutT*>(p.c);
    int local = 0;
    for (int i = blockIdx.x; i < num_items; i += gridDim.x) {
      Item const it = decode_item(i, n_tiles, p.splits, k_tiles);
      int32_t const* const blk = p.sorted_token_ids + it.m_block * kRows;
      int64_t const col0 = int64_t(it.n_tile) * kBlockN;
      int32_t const sid_lane = blk[lane];
      bool const small = blk[8] >= p.num_valid_tokens;

      if (p.expert_ids[it.m_block] < 0) {
        // Tokens routed to an expert that is not local to this rank.
        if (quarter == 0 && sid_lane < p.num_valid_tokens && it.split == 0) {
          uint4* dst =
              reinterpret_cast<uint4*>(c + sid_lane * int64_t(p.N) + col0);
          for (int j = 0; j < kBlockN * int(sizeof(OutT)) / 16; ++j)
            dst[j] = make_uint4(0, 0, 0, 0);
        }
        continue;
      }

      // A small block's rows 0-7 are in lanes 0-7 of every quarter, so the
      // four warps split its columns; a larger block's rows are lanes 0-31
      // of quarter 0.
      int const t = lane;
      int32_t const sid = (small ? lane < 8 : quarter == 0)
                              ? sid_lane
                              : p.num_valid_tokens;
      bool const valid = sid < p.num_valid_tokens;
      int const n_begin = small ? quarter * (kBlockN / 4) : 0;
      int const n_end = small ? n_begin + kBlockN / 4 : kBlockN;

      float const w = (p.topk_weights != nullptr && valid)
                          ? p.topk_weights[sid]
                          : 1.0f;
      bool const direct = p.splits == 1;
      mbar_wait(&acc_full_bar, local & 1);
      tc_fence_after();
      if (__any_sync(0xffffffffu, valid)) {
#pragma unroll 1
        for (int n = n_begin; n < n_end; n += 32) {
          uint32_t v[32];
          tmem_ld32(lane_addr + n, v);
          if (valid) {
            float f[32];
#pragma unroll
            for (int j = 0; j < 32; ++j) f[j] = __uint_as_float(v[j]);
            if (direct) {
              OutT* dst = c + sid * int64_t(p.N) + col0 + n;
#pragma unroll
              for (int j = 0; j < 32; j += 8) store8<OutT>(dst + j, f + j, w);
            } else {
              float4* dst = reinterpret_cast<float4*>(
                  p.partials +
                  (int64_t(it.split) * p.num_valid_tokens + sid) * p.N +
                  col0 + n);
#pragma unroll
              for (int j = 0; j < 8; ++j)
                dst[j] = make_float4(f[4 * j], f[4 * j + 1], f[4 * j + 2],
                                     f[4 * j + 3]);
            }
          }
        }
      }
      tc_fence_before();
      mbar_arrive(&acc_empty_bar);
      ++local;
      if (direct) continue;

      // Split-K: the last split to finish this tile sums the partials.
      // Real rows are a prefix of the block, so count them per warp.
      // epi_* only hold real rows: the reduction reads rows [0, rows).
      int const warp_rows = __popc(__ballot_sync(0xffffffffu, valid));
      // Small blocks repeat rows 0-7 in every quarter; count them once.
      if (lane == 0) epi_rows[quarter] = quarter > 0 ? 0 : warp_rows;
      if (valid) {
        epi_sid[t] = sid;
        epi_weight[t] = w;
      }
      __threadfence();
      epilogue_bar();
      int32_t* counter = p.counters + it.m_block * n_tiles + it.n_tile;
      if (threadIdx.x == kFirstEpilogueWarp * 32)
        last_split = atomicAdd(counter, 1) == p.splits - 1;
      epilogue_bar();
      if (!last_split) continue;
      __threadfence();
      {
        // Spread the tile over all epilogue threads and issue every split's
        // load before summing, so the reduction costs one L2 round trip.
        int const rows =
            epi_rows[0] + epi_rows[1] + epi_rows[2] + epi_rows[3];
        int const et = threadIdx.x - kFirstEpilogueWarp * 32;
        int64_t const split_stride = int64_t(p.num_valid_tokens) * p.N;
        constexpr int kVecs = kBlockN / 4;
        for (int v = et; v < rows * kVecs; v += kEpilogueThreads) {
          int const r = v / kVecs, col = (v % kVecs) * 4;
          int64_t const off = int64_t(epi_sid[r]) * p.N + col0 + col;
          float4 parts[kMaxSplits];
#pragma unroll
          for (int sp = 0; sp < kMaxSplits; ++sp)
            if (sp < p.splits)
              parts[sp] = __ldcg(reinterpret_cast<float4 const*>(
                  p.partials + sp * split_stride + off));
          float acc[4] = {parts[0].x, parts[0].y, parts[0].z, parts[0].w};
#pragma unroll
          for (int sp = 1; sp < kMaxSplits; ++sp)
            if (sp < p.splits) {
              acc[0] += parts[sp].x, acc[1] += parts[sp].y;
              acc[2] += parts[sp].z, acc[3] += parts[sp].w;
            }
          float const wr = epi_weight[r];
          uint2 packed;
          packed.x = pack2<OutT>(acc[0] * wr, acc[1] * wr);
          packed.y = pack2<OutT>(acc[2] * wr, acc[3] * wr);
          *reinterpret_cast<uint2*>(c + off) = packed;
        }
      }
      epilogue_bar();  // epi_* are rewritten for the next item
      if (threadIdx.x == kFirstEpilogueWarp * 32) *counter = 0;
    }
    // TMEM may be freed once every epilogue load is done.
    tc_fence_before();
    mbar_arrive(&epi_done_bar);
  }

  tc_fence_before();
  __syncthreads();
  if (warp == kMmaWarp) {
    mbar_wait(&epi_done_bar, 0);
    tc_fence_after();
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::"r"(
            tmem),
        "n"(kTmemCols)
        : "memory");
  }
}

}  // namespace lut_b_moe_stream
}  // namespace vllm

void lut_b_moe_gemm_stream(torch::stable::Tensor const& a,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b,
                           torch::stable::Tensor& c,
                           std::optional<torch::stable::Tensor> topk_weights,
                           torch::stable::Tensor const& sorted_token_ids,
                           torch::stable::Tensor const& expert_ids,
                           torch::stable::Tensor const& num_tokens_post_padded,
                           int64_t top_k, torch::stable::Tensor& split_counters,
                           int64_t split_k) {
  using namespace vllm::lut_b_moe_stream;
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(a.scalar_type() == ScalarType::Float8_e4m3fn,
                  "lut_b_moe_gemm_stream: a must be float8_e4m3fn");
  STD_TORCH_CHECK(a_scales.scalar_type() == ScalarType::Byte,
                  "lut_b_moe_gemm_stream: a_scales must be uint8 (UE8M0)");
  STD_TORCH_CHECK(b.scalar_type() == ScalarType::Byte,
                  "lut_b_moe_gemm_stream: b must be uint8");
  STD_TORCH_CHECK(a.dim() == 2 && a.is_contiguous(),
                  "lut_b_moe_gemm_stream: a must be a contiguous 2D tensor");
  STD_TORCH_CHECK(c.dim() == 2 && c.is_contiguous(),
                  "lut_b_moe_gemm_stream: c must be a contiguous 2D tensor");
  int64_t const K = a.size(1);
  int64_t const N = c.size(1);
  STD_TORCH_CHECK(K % kBlockK == 0 && N % kBlockN == 0,
                  "lut_b_moe_gemm_stream: N must be a multiple of 256 and K "
                  "of 128, got N=",
                  N, " K=", K);
  STD_TORCH_CHECK(a_scales.dim() == 2 && a_scales.is_contiguous() &&
                      a_scales.size(0) == a.size(0) &&
                      a_scales.size(1) == K / 32,
                  "lut_b_moe_gemm_stream: a_scales must be [M, K / 32]");
  STD_TORCH_CHECK(
      b.dim() == 4 && b.is_contiguous() && b.size(1) == N / kBlockN &&
          b.size(2) == K / kBlockK && b.size(3) == kBStageBytes,
      "lut_b_moe_gemm_stream: b must be paired tiles [E, N / 256, K / 128, ",
      kBStageBytes, "]");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(a.const_data_ptr()) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(b.const_data_ptr()) % 16 == 0,
                  "lut_b_moe_gemm_stream: a and b must be 16-byte aligned");
  STD_TORCH_CHECK(sorted_token_ids.scalar_type() == ScalarType::Int &&
                      expert_ids.scalar_type() == ScalarType::Int &&
                      num_tokens_post_padded.scalar_type() == ScalarType::Int &&
                      split_counters.scalar_type() == ScalarType::Int,
                  "lut_b_moe_gemm_stream: routing tensors and split_counters "
                  "must be int32");
  STD_TORCH_CHECK(expert_ids.size(0) * kRows >= sorted_token_ids.size(0),
                  "lut_b_moe_gemm_stream: routing must be aligned to block "
                  "size 32");
  float const* topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    STD_TORCH_CHECK(topk_weights->scalar_type() == ScalarType::Float &&
                        topk_weights->is_contiguous() &&
                        topk_weights->numel() == c.size(0),
                    "lut_b_moe_gemm_stream: topk_weights must be contiguous "
                    "float32 with one entry per row of c");
    topk_weights_ptr = topk_weights->const_data_ptr<float>();
  }

  int const device = a.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 10 && properties->minor == 7,
                  "lut_b_moe_gemm_stream requires an SM107 (Rubin) GPU");
  int64_t const num_m_blocks = expert_ids.size(0);
  int64_t const rows = c.size(0);
  if (num_m_blocks == 0 || rows == 0) return;

  int64_t const n_tiles = N / kBlockN;
  int64_t const k_tiles = K / kBlockK;
  int64_t const sms = properties->multiProcessorCount;

  // split_k <= 0 means no split. Each extra split adds a partials store, a
  // fence, an atomic and a reduction per item (~4 us), which cost more than
  // the extra weight streams gained at every decode size measured (5-15%
  // slower at 1-2 tokens on the GLM-4.7-Flash expert shapes).
  int64_t splits = split_k <= 0 ? 1 : split_k;
  splits = std::min<int64_t>({std::max<int64_t>(splits, 1), k_tiles,
                              int64_t(kMaxSplits)});

  float* partials = nullptr;
  std::optional<torch::stable::Tensor> partials_buf;
  if (splits > 1) {
    // Counters are indexed by real blocks, each holding at least one row.
    int64_t const counters_needed = std::min(num_m_blocks, rows) * n_tiles;
    STD_TORCH_CHECK(split_counters.is_contiguous() &&
                        split_counters.numel() >= counters_needed,
                    "lut_b_moe_gemm_stream: split_counters needs ",
                    counters_needed, " zero-initialized int32 entries");
    partials_buf = torch::stable::empty({splits, rows, N}, ScalarType::Float,
                                        std::nullopt, a.device());
    partials = partials_buf->mutable_data_ptr<float>();
  }

  Params params;
  params.a = static_cast<uint8_t const*>(a.const_data_ptr());
  params.a_scales = a_scales.const_data_ptr<uint8_t>();
  params.b = b.const_data_ptr<uint8_t>();
  params.c = c.mutable_data_ptr();
  params.partials = partials;
  params.counters = split_counters.mutable_data_ptr<int32_t>();
  params.topk_weights = topk_weights_ptr;
  params.sorted_token_ids = sorted_token_ids.const_data_ptr<int32_t>();
  params.expert_ids = expert_ids.const_data_ptr<int32_t>();
  params.num_tokens_post_padded =
      num_tokens_post_padded.const_data_ptr<int32_t>();
  params.num_valid_tokens = static_cast<int>(rows);
  params.num_sorted = static_cast<int>(sorted_token_ids.size(0));
  params.top_k = static_cast<int>(top_k);
  params.N = static_cast<int>(N);
  params.K = static_cast<int>(K);
  params.splits = static_cast<int>(splits);

  int64_t const max_items = num_m_blocks * n_tiles * splits;
  dim3 const grid(static_cast<unsigned>(std::min(sms, max_items)));
  cudaStream_t const stream = get_current_cuda_stream(device);
  auto launch = [&](auto* out, auto groups) {
    using OutT = std::remove_pointer_t<decltype(out)>;
    constexpr int kGroups = decltype(groups)::value;
    auto kernel = lut_b_moe_stream_kernel<OutT, kGroups>;
    int const smem_bytes = Config<kGroups>::kSmemBytes;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    kernel<<<grid, kThreads, smem_bytes, stream>>>(params);
  };
  // With at most 8 routed rows every block fits one 8-row A group, and the
  // smaller stage buys a deeper weight ring. Otherwise blocks hold up to 32.
  bool const tiny = rows <= 8;
  auto dispatch = [&](auto* out) {
    if (tiny)
      launch(out, std::integral_constant<int, 1>{});
    else
      launch(out, std::integral_constant<int, 4>{});
  };
  if (c.scalar_type() == ScalarType::BFloat16) {
    dispatch(reinterpret_cast<__nv_bfloat16*>(c.mutable_data_ptr()));
  } else if (c.scalar_type() == ScalarType::Half) {
    dispatch(reinterpret_cast<__half*>(c.mutable_data_ptr()));
  } else {
    STD_TORCH_CHECK(false,
                    "lut_b_moe_gemm_stream: c must be bfloat16 or float16");
  }
  cudaError_t const err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "lut_b_moe_gemm_stream launch failed: ",
                  cudaGetErrorString(err));
}
