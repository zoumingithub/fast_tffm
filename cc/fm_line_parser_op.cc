#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

REGISTER_OP("FmLineParser")
    .Input("fea_str: string")
    .Output("ori_ids: int64")
    .Output("feature_ids: int32")
    .Output("feature_vals: float32")
    .Output("feature_poses: int32")
    .Attr("vocab_size: int")
    .Attr("hash_feature_id: bool = false");

#define MAX_FEATURE_ID_LENGTH 100

using namespace tensorflow;

class FmLineParserOp : public OpKernel {
 public:

  explicit FmLineParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hash_feature_id", &hash_feature_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* fea_str_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("fea_str", &fea_str_tensor));
    auto fea_str = fea_str_tensor->scalar<string>()();

    std::map<int64, int32> ori_id_map;
    std::vector<int32> feature_ids;
    std::vector<float> feature_vals;
    std::vector<int32> feature_poses;
    std::vector<string> data_lines;
    string buf;
    std::stringstream ss(fea_str);
    while (getline(ss, buf, ';')){
      data_lines.push_back(buf);
      std::cout << buf << std::endl;
    }

    feature_poses.push_back(0);
    for(size_t i = 0; i < data_lines.size(); i++) {
      ParseLine(ctx, data_lines[i], hash_feature_id_, vocab_size_, ori_id_map, feature_ids, feature_vals, feature_poses);
    }

    std::vector<int64> ori_ids(ori_id_map.size(), 0);
    for (auto it = ori_id_map.begin(); it != ori_id_map.end(); ++it) {
      ori_ids[it->second] = it->first;
    }

    AllocateTensorForVector<int64>(ctx, "ori_ids", ori_ids);
    AllocateTensorForVector<int32>(ctx, "feature_ids", feature_ids);
    AllocateTensorForVector<float>(ctx, "feature_vals", feature_vals);
    AllocateTensorForVector<int32>(ctx, "feature_poses", feature_poses);
  }

 private:
  int64 vocab_size_;
  bool hash_feature_id_;

  void ParseLine(OpKernelContext* ctx, const string& line, bool hash_feature_id, int64 vocab_size, std::map<int64, int32>& ori_id_map, std::vector<int32>& feature_ids, std::vector<float>& feature_vals, std::vector<int32>& feature_poses) {
    const char* p = line.c_str();
    int64 ori_id;
    int32 fid;
    float fv;
    int offset;

    size_t read_size;
    char ori_id_str[MAX_FEATURE_ID_LENGTH];
    char* err;
    while (true) {
      if (sscanf(p, " %[^: ]%n", ori_id_str, &offset) != 1) break;


      if (hash_feature_id) {
        ori_id = Hash64(ori_id_str, strlen(ori_id_str));
      } else {
        ori_id = strtol(ori_id_str, &err, 10);
        OP_REQUIRES(ctx, *err == 0, errors::InvalidArgument("Invalid feature id ", ori_id_str, ". Set hash_feature_id = True?"))
      }
      ori_id = labs(ori_id % vocab_size);
      p += offset;
      if (*p == ':') {
        OP_REQUIRES(ctx, sscanf(p, ":%f%n", &fv, &offset) == 1, errors::InvalidArgument("Invalid feature value: ", ori_id_str))
        p += offset;
      } else {
        fv = 1;
      }
      auto iter = ori_id_map.find(ori_id);
      if (iter == ori_id_map.end()) {
        fid = ori_id_map.size();
        ori_id_map[ori_id] = fid;
      } else {
        fid = iter->second;
      }
      feature_ids.push_back(fid);
      feature_vals.push_back(fv);
    }
    feature_poses.push_back(feature_ids.size());
  }

  template<typename T>
  void AllocateTensorForVector(OpKernelContext* ctx, const string& name, const std::vector<T>& data) {
    Tensor* tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, TensorShape({static_cast<int64>(data.size())}), &tensor));
    auto tensor_data = tensor->flat<T>();
    for (size_t i = 0; i < data.size(); ++i) {
      tensor_data(i) = data[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FmLineParser").Device(DEVICE_CPU), FmLineParserOp);
