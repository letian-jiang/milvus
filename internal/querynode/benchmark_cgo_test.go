package querynode

import (
	"context"
	"sync"
	"testing"

	"github.com/milvus-io/milvus/internal/log"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
	"go.uber.org/zap/zapcore"

	"github.com/stretchr/testify/assert"
)

const (
	NQ       = 10
	TOPK     = 10
	SEGMENTS = 64     // number of prepared segments
	RECORDS  = 100000 // 250K -> 512MB
	INDEX    = IndexFaissIVFFlat
)

var initOnce sync.Once
var segments []*Segment
var req *searchRequest

func prepareIndexedSegmentsAndSearchRequest() {
	qs, err := genSimpleQueryShard(context.TODO())
	if err != nil {
		panic(err)
	}
	collection, err := qs.metaReplica.getCollectionByID(defaultCollectionID)
	if err != nil {
		panic(err)
	}
	iReq, _ := genSearchRequest(NQ, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      nil,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	req, err = newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	if err != nil {
		panic(err)
	}

	for i := 0; i < SEGMENTS; i++ {
		segmentID := UniqueID(i + 999)
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, segmentID)
		if err != nil {
			panic(err)
		}
		segment.BuildVecIndex(100)
		segments = append(segments, segment)
	}
}

func benchmarkSearchIndexSerial(b *testing.B, n int) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	initOnce.Do(prepareIndexedSegmentsAndSearchRequest)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for i := 0; i < n; i++ {
			s := segments[i]
			_, err := s.search(req)
			assert.NoError(b, err)
		}
	}
}

func benchmarkSearchIndexConcurrent(b *testing.B, numSegments, concurrency int) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	initOnce.Do(prepareIndexedSegmentsAndSearchRequest)
	sem := make(chan int, concurrency)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg := sync.WaitGroup{}
		for i := 0; i < numSegments; i++ {
			s := segments[i]
			wg.Add(1)
			go func(s *Segment) {
				defer wg.Done()
				sem <- 1
				_, err := s.search(req)
				assert.NoError(b, err)
				<-sem
			}(s)
		}
		wg.Wait()
	}
}

func BenchmarkSearchIndexSegment1Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 1, 1)
}

func BenchmarkSearchIndexSegment2Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 2, 1)
}

func BenchmarkSearchIndexSegment2Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 2, 2)
}

func BenchmarkSearchIndexSegment4Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 4, 1)
}

func BenchmarkSearchIndexSegment4Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 4, 2)
}

func BenchmarkSearchIndexSegment4Concurrency4(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 4, 4)
}

func BenchmarkSearchIndexSegment8Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 8, 1)
}

func BenchmarkSearchIndexSegment8Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 8, 2)
}

func BenchmarkSearchIndexSegment8Concurrency4(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 8, 4)
}

func BenchmarkSearchIndexSegment8Concurrency8(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 8, 8)
}

func BenchmarkSearchIndexSegment16Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 16, 1)
}

func BenchmarkSearchIndexSegment16Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 16, 2)
}

func BenchmarkSearchIndexSegment16Concurrency4(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 16, 4)
}

func BenchmarkSearchIndexSegment16Concurrency8(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 16, 8)
}

func BenchmarkSearchIndexSegment16Concurrency16(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 16, 16)
}

func BenchmarkSearchIndexSegment32Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 1)
}

func BenchmarkSearchIndexSegment32Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 2)
}

func BenchmarkSearchIndexSegment32Concurrency4(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 4)
}

func BenchmarkSearchIndexSegment32Concurrency8(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 8)
}

func BenchmarkSearchIndexSegment32Concurrency16(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 16)
}

func BenchmarkSearchIndexSegment32Concurrency32(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 32, 32)
}

func BenchmarkSearchIndexSegment64Concurrency1(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 1)
}

func BenchmarkSearchIndexSegment64Concurrency2(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 2)
}

func BenchmarkSearchIndexSegment64Concurrency4(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 4)
}

func BenchmarkSearchIndexSegment64Concurrency8(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 8)
}

func BenchmarkSearchIndexSegment64Concurrency16(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 16)
}

func BenchmarkSearchIndexSegment64Concurrency32(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 32)
}

func BenchmarkSearchIndexSegment64Concurrency64(b *testing.B) {
	benchmarkSearchIndexConcurrent(b, 64, 64)
}
