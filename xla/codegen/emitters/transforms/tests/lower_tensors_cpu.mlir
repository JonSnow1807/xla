// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// ==============================================================================
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

func.func @load_non_gep_from_args(%arg0: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.getelementptr inbounds %arg0[1]
    : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
  func.return %2 : !llvm.ptr
}

// CHECK-LABEL: @load_non_gep_from_args
// CHECK-NEXT:    %0 = llvm.getelementptr inbounds %arg0[1]
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    return %2 : !llvm.ptr

// -----

func.func @copy_slice(%src: tensor<12xf32>, %dst: tensor<20xf32>,
                      %src_idx: index, %dst_idx: index) -> tensor<20xf32> {
  %res = xla.copy_slice %src[%src_idx] into %dst[%dst_idx] count(6)
      : tensor<12xf32> into tensor<20xf32>
  func.return %res : tensor<20xf32>
}

// CHECK-LABEL: func.func @copy_slice(
// CHECK-SAME:      %[[SRC:.*]]: !llvm.ptr, %[[DST:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[SRC_IDX:.*]]: index, %[[DST_IDX:.*]]: index) {
// CHECK-DAG:     %[[BYTES:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-DAG:     %[[SRC_PTR:.*]] = llvm.getelementptr inbounds %[[SRC]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<12 x f32>
// CHECK-DAG:     %[[DST_PTR:.*]] = llvm.getelementptr inbounds %[[DST]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<20 x f32>
// CHECK:         "llvm.intr.memcpy"(%[[DST_PTR]], %[[SRC_PTR]], %[[BYTES]]) <{isVolatile = false}>
// CHECK:         return
