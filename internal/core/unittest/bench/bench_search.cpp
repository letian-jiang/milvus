// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <cstdint>
#include <benchmark/benchmark.h>
#include <string>
#include "segcore/SegmentGrowing.h"
#include "segcore/SegmentSealed.h"
#include "test_utils/DataGen.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;

const int dim = 512;
const int N = 100000;
const int nq = 100;
const int64_t ts = MAX_TIMESTAMP;
const int seed = 66;

const auto schema = []() {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    auto i64_fid = schema->AddDebugField("age", DataType::INT64);
    schema->set_primary_field_id(i64_fid);
    return schema;
}();

const auto dataset_ = [] {
    auto dataset_ = DataGen(schema, N, seed);
    return dataset_;
}();

const auto plan = [] {
    std::string dsl = R"({
        "bool": {
                "vector": {
                    "fakevec": {
                        "metric_type": "L2",
                        "params": {
                            "nprobe": 10
                        },
                        "query": "$0",
                        "topk": 10,
                        "round_decimal": -1
                    }
                }
        }
    })";
    auto plan = CreatePlan(*schema, dsl);
    return plan;
}();

auto ph_group = [] {
    auto ph_group_raw = CreatePlaceholderGroup(nq, dim, seed);
    auto ph_group = ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());
    return ph_group;
}();

static void
Search_Sealed_Without_Index(benchmark::State& state) {
    auto segment = CreateSealedSegment(schema);
    SealedLoadFieldData(dataset_, *segment);

    for (auto _ : state) {
        auto qr = segment->Search(plan.get(), ph_group.get(), ts);
    }
}

static void
Search_Sealed_With_Index(benchmark::State& state) {
    auto segment = CreateSealedSegment(schema);
    SealedLoadFieldData(dataset_, *segment);
    segment->BuildVecIndex(milvus::FieldId(100));

    for (int i = 0; i < 100; i++) {
        auto qr = segment->Search(plan.get(), ph_group.get(), ts);
    }

    for (auto _ : state) {
        auto qr = segment->Search(plan.get(), ph_group.get(), ts);
    }
}

BENCHMARK(Search_Sealed_Without_Index);
BENCHMARK(Search_Sealed_With_Index);