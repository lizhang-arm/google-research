// Copyright 2026 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_UTILS_INTRINSICS_MEM_NEON_H_
#define SCANN_UTILS_INTRINSICS_MEM_NEON_H_

#ifdef __aarch64__

#include <arm_neon.h>
#include <stddef.h>

namespace research_scann {
namespace neon {

static inline void load_f32_4x4(const float* s, ptrdiff_t src_stride,
                                float32x4_t& s0, float32x4_t& s1,
                                float32x4_t& s2, float32x4_t& s3) {
  s0 = vld1q_f32(s);
  s += src_stride;
  s1 = vld1q_f32(s);
  s += src_stride;
  s2 = vld1q_f32(s);
  s += src_stride;
  s3 = vld1q_f32(s);
}

static inline void store_f32_4x4(float* s, ptrdiff_t dst_stride,
                                 const float32x4_t s0, const float32x4_t s1,
                                 const float32x4_t s2, const float32x4_t s3) {
  vst1q_f32(s, s0);
  s += dst_stride;
  vst1q_f32(s, s1);
  s += dst_stride;
  vst1q_f32(s, s2);
  s += dst_stride;
  vst1q_f32(s, s3);
}

}  // namespace neon
}  // namespace research_scann

#endif  // __aarch64__

#endif  // SCANN_UTILS_INTRINSICS_MEM_NEON_H_
