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
 * \file rnnt_loss-inl.h
 * \brief
 * \author Mingkun Huang
*/

#ifndef MXNET_OPERATOR_CONTRIB_RNNT_LOSS_INL_H_
#define MXNET_OPERATOR_CONTRIB_RNNT_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../sequence_op_common.h"
#include "../mshadow_op.h"
#include "../nn/sequence_mask-inl.h"

namespace mxnet {
namespace op {

namespace rnnt_loss {
enum RNNTLossOpInputs { kTrans, kPred, kLabel, kInputLength, kLabelLength };
enum RNNTLossOpOutputs { kOut, kTransGrad, kPredGrad };
enum RNNTLossOpForwardResource { kTempSpace };
}

template <typename T>
inline void get_workspace_size(int maxT, int maxU,
                               int minibatch,
                               bool gpu,
                               size_t* size_bytes)
{
    if (minibatch <= 0 ||
        maxT <= 0 ||
        maxU <= 0 ||
        alphabet_size <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    *size_bytes = 0;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(float) * maxT * maxU * 2;

    // log_p, NOTE here just store the (denominator + maximum)
    // but for softmax, we need both maximum and denominator
    per_minibatch_bytes += sizeof(float) * maxT * maxU;

    if (gpu) {
        // forward-backward loglikelihood
        per_minibatch_bytes += sizeof(float) * 2;
        // labels
        per_minibatch_bytes += sizeof(int) * (maxU - 1);
        // length
        per_minibatch_bytes += sizeof(int) * 2;
    }

    *size_bytes = per_minibatch_bytes * minibatch;
}

// Takes a tensor of labels, and interprets 0-elements at the end of the vector
// as padding. The tensor is packed into an std::vector without padding
// characters. The label sequence lengths are also inferred from the padding chars.
// When cudnn is enabled, the return value signifies whether the cudnn length limit is exceeded.
template <typename DType, typename xpu>
inline bool LabelTensorToPackedVector(mshadow::Tensor<xpu, 2, DType> labels,
                                      int padding_mask,
                                      std::vector<int> *packed_labels,
                                      std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  bool exceed_limit = false;

  std::vector<int> cpu_labels(max_num_labels*batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data()+b*max_num_labels;
    auto res = std::find(start, start+max_num_labels, padding_mask);
    int len = std::distance(start, res);
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    exceed_limit = exceed_limit || len > CUDNN_LABEL_LENGTH_LIMIT;
#endif
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
    label_lengths->at(b) = len;
  }
  return exceed_limit;
}

// Takes a tensor of labels, and a vector which specifies the actual length of each label
// The tensor is packed into an std::vector without padding characters.
// The label length vector is copied into an std::vector.
// When cudnn is enabled, the return value signifies whether the cudnn length limit is exceeded.
template <typename DType, typename xpu>
inline bool PackLabelByLength(mshadow::Tensor<xpu, 2, DType> labels,
                              mshadow::Tensor<xpu, 1, DType> in_label_lengths,
                              std::vector<int> *packed_labels,
                              std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  bool exceed_limit = false;

  IndexTensorToVector(in_label_lengths, label_lengths);

  std::vector<int> cpu_labels(max_num_labels*batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data()+b*max_num_labels;
    int len = label_lengths->at(b);
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    exceed_limit = exceed_limit || len > CUDNN_LABEL_LENGTH_LIMIT;
#endif
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
  }
  return exceed_limit;
}

struct RNNTLossParam : public dmlc::Parameter<RNNTLossParam> {
  int blank_label;
  DMLC_DECLARE_PARAMETER(RNNTLossParam) {
    DMLC_DECLARE_FIELD(blank_label)
      .set_default(0)
      .describe("Set the label that is reserved for blank label.");
  }
};

template <typename xpu>
class RNNTLossOp : public Operator {
 public:
  explicit RNNTLossOp(RNNTLossParam p) {
    this->param_ = p;
    exceed_cudnn_limit = false;
  }

  ~RNNTLossOp() {

  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 5U);
    CHECK_EQ(out_data.size(), 3U);
    exceed_cudnn_limit = false; // not use now
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> trans_acts =
        in_data[rnnt_loss::kTrans].get<xpu, 3, real_t>(s);
    Tensor<xpu, 3, real_t> pred_acts = 
        in_data[rnnt_loss::kPred].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2, real_t> labels =
        in_data[rnnt_loss::kLabel].get<xpu, 2, real_t>(s);

    Tensor<xpu, 1, real_t> costs =
        out_data[rnnt_loss::kOut].get<xpu, 1, real_t>(s);
    Tensor<xpu, 3, real_t> trans_grad =
        out_data[rnnt_loss::kTransGrad].get<xpu, 3, real_t>(s);
    Tensor<xpu, 3, real_t> pred_grad = 
        out_data[rnnt_loss:kPredGrad].get<xpu, 3, real_t>(s);

    int batch_size = static_cast<int>(trans_acts.size(0));
    int maxT = static_cast<int>(trans_acts.size(1));
    int alphabet_size = static_cast<int>(trans_acts.size(2));
    int maxU = static_cast<int>(pred_acts.size(1));

    // data_lengths
    std::vector<int> data_lengths(batch_size, maxT);
    IndexTensorToVector(in_data[kInputLength].get<xpu, 1, real_t>(s), &data_lengths);

    // label_lengths
    std::vector<int> label_lengths(batch_size);
    IndexTensorToVector(in_data[kLabelLength].get<xpu, 1, real_t>(s), &label_lengths);

    // allocate temporary workspace
    size_t size_bytes;
    bool gpu = data.kDevCPU ? false : true;

    get_workspace_size<real_t>(maxT, maxU, batch_size, gpu, &size_bytes);

    // round-up so there are enough elems in memory
    int num_tmp_elems = (size_bytes + sizeof(real_t) - 1) / sizeof(real_t);
    Tensor<xpu, 1, real_t> workspace =
        ctx.requested[rnnt_loss::kTempSpace].get_space_typed<xpu, 1, real_t>(
            Shape1(num_tmp_elems), s);

    compute_rnnt_cost(trans_acts.dptr_, pred_acts.dptr_, costs.dptr_, 
                     trans_grad.dptr_, pred_acts.dptr_, labels.dptr_,
                     label_lengths.data(), data_lengths.data(),
                     workspace.dptr_, req[rnnt_loss::kGrad] != mxnet::kNullOp,
                     param_.blank_label, batch_size, maxT, maxU, alphabet_size);

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> trans_grad =
        in_grad[rnnt_loss::kTrans].get<xpu, 3, real_t>(s);
    Tensor<xpu, 3, real_t> pred_grad = 
        in_grad[rnnt_loss::kPred].get<xpu, 3, real_t>(s);
    Tensor<xpu, 1, real_t> output_grad =
        out_grad[rnnt_loss::kOut].get<xpu, 1, real_t>(s);

    Tensor<xpu, 3, real_t> trans_grad_computed =
        out_data[rnnt_loss::kTransGrad].get<xpu, 3, real_t>(s);
    Tensor<xpu, 3, real_t> pred_grad_computed = 
        out_data[rnnt_loss::kPredGrad].get<xpu, 3, real_t>(s);

    Assign(trans_grad, req[rnnt_loss::kTrans],
           mshadow::expr::broadcast<0>(output_grad, trans_grad.shape_) * trans_grad_computed);
    Assign(pred_grad, req[rnnt_loss::kPred],
           mshadow::expr::broadcast<0>(output_grad, pred_grad.shape_) * pred_grad_computed);
  }

 private:
  RNNTLossParam param_;
  bool exceed_cudnn_limit;

};  // class RNNTLossOp

template <typename xpu>
Operator *CreateOp(RNNTLossParam param, int dtype);

// #if DMLC_USE_CXX11
class RNNTLossProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 3; }

  std::vector<std::string> ListArguments() const override {
    return {"trans_acts", "pred_acts", "label", "data_lengths", "label_lengths"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "trans_grad", "pred_grad"};
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    index_t expected_inputs = 5;
    CHECK_EQ(in_shape->size(), expected_inputs)
        << "Expect " << expected_inputs << " inputs to the symbol.";

    const TShape &tshape = (*in_shape)[rnnt_loss::kTrans];
    const TShape &pshape = (*in_shape)[rnnt_loss::kPred];
    const TShape &lshape = (*in_shape)[rnnt_loss::kLabel];
    CHECK_EQ(tshape.ndim(), 3U) << "The trans_acts array must be of rank 3.";
    CHECK_EQ(pshape.ndim(), 3U) << "The pred_acts array must be of rank 3.";
    CHECK_EQ(lshape.ndim(), 2U) << "The labels array must be of rank 2.";
    CHECK_EQ(tshape[0], lshape[0])
        << "The batch size for the labels and data arrays must be the same.";

    CHECK_EQ(tshape[0], pshape[0])
        << "The batch size for the trans_acts and pred_acts must be the same.";
    CHECK_EQ(tshape[2], pshape[2])
        << "The alphabet size for the trans_acts and pred_acts must be the same.";

    const TShape &dlshape = (*in_shape)[kInputLength];
    CHECK_EQ(dlshape.ndim(), 1U) << "Data length array must be a vector.";
    CHECK_EQ(dlshape[0], tshape[0])
        << "The batch size for the data and data lengths must be the same.";

    const TShape &llshape = (*in_shape)[kLabelLength];
    CHECK_EQ(llshape.ndim(), 1U) << "Label length array must be a vector.";
    CHECK_EQ(llshape[0], lshape[0])
        << "The batch size for the labels and label lengths must be the same.";

    TShape oshape(1);
    oshape[0] = tshape[0];  // batch size
    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(tshape);  // trans_grad output
    out_shape->push_back(pshape);  // pred_grad output
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new RNNTLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "_contrib_RNNTLoss"; }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[rnnt_loss::kOut], out_data[rnnt_loss::kTransGrad],
            out_data[rnnt_loss:kPredGrad]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  RNNTLossParam param_;
};      // class RNNTLossProp
// #endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_RNNT_LOSS_INL_H_
