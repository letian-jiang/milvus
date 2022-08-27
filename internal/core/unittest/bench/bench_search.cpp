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
#include "test_utils/BS_thread_pool.hpp"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;

const int ns = 64; // number of segments
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

auto segments = [] {
    std::vector<SegmentSealedPtr> segments(ns);
    for (int i = 0; i < ns; i++) {
        auto segment = CreateSealedSegment(schema);
        SealedLoadFieldData(dataset_, *segment);
        segment->BuildVecIndex(milvus::FieldId(100));
        segments[i] = move(segment);
    }
    return segments;
}();

void task(int segment_id) {
    auto qr = segments[segment_id]->Search(plan.get(), ph_group.get(), ts);
}

static void
SearchIndex(benchmark::State& state) {
    auto num_segments = state.range(0);
    auto pool_size = state.range(1);
    
    // init thread pool
    auto pool = BS::thread_pool(pool_size);
    // benchmark
    for (auto _ : state) {
        for (int i = 0; i < (int)num_segments; i++) {
            pool.push_task(task, i);
        }
        pool.wait_for_tasks();
    }
}

static void 
SegmentConcurrencyArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= ns; i <<= 1) { // segments
    for (int j = 1; j <= i; j <<= 1) { // concurrency
        // std::cout << "args: " << i << ' ' << j << std::endl;
        b->Args({i, j});
    }
  }
}

// BENCHMARK(SearchIndex)->Apply(SegmentConcurrencyArguments);
// BENCHMARK(SearchIndex)->Apply(SegmentConcurrencyArguments)->MeasureProcessCPUTime();
BENCHMARK(SearchIndex)->MinTime(10)->Apply(SegmentConcurrencyArguments)->MeasureProcessCPUTime()->UseRealTime();