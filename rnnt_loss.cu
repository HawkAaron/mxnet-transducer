/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file rnnt_loss.cc
 * \brief
 * \author Mingkun Huang
*/

#include "./rnnt_loss-inl.h"
#include "./rnnt_include/detail/gpu_rnnt.h"

namespace mshadow {

template <typename DType>
void compute_rnnt_cost(const Tensor<gpu, 3, DType> trans_acts, 
                      const Tensor<gpu, 3, DType> pred_acts, DType *costs, 
                      DType *trans_grads, DType *pred_grads, int *labels,
                      int *label_lengths, int *data_lengths,
                      void *workspace, int train, int blank_label) {

  int minibatch = static_cast<int>(trans_acts.size(0));
  int maxT = static_cast<int>(trans_acts.size(1));
  int alphabet_size = static_cast<int>(trans_acts.size(2));
  int maxU = static_cast<int>(pred_acts.size(1));

  warp_rnnt::GpuRNNT<DType> rnnt(minibatch, maxT, maxU, alphabet_size, workspace, 
                    blank_label, trans_acts.stream_->stream_);
  if (train) {
    rnnt.cost_and_grad(trans_acts.dptr_, pred_acts.dptr_, 
                        trans_grads, pred_grads,
                        costs, 
                        labels, label_lengths, 
                        data_lengths);
  } else {
    rnnt.score_forward(trans_acts.dptr_, pred_acts.dptr_, 
                        costs, 
                        labels, label_lengths, 
                        data_lengths);
  }
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(RNNTLossParam param, int dtype) {
  return new RNNTLossOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
