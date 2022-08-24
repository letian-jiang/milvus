package querynode

import (
	"context"
	"sync"
	"testing"

	"github.com/milvus-io/milvus/internal/log"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/concurrency"
	"github.com/panjf2000/ants/v2"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/stretchr/testify/assert"
)

const (
	NQ       = 10
	TOPK     = 10
	SEGMENTS = 10
	RECORDS  = 100000            // 250K -> 512MB
	INDEX    = IndexFaissIVFFlat // not effective
)

func BenchmarkSearchWithoutIndex(b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())
	defer cancel()
	queryShardObj, err := genSimpleQueryShard(tx)
	assert.NoError(b, err)
	collection, err := queryShardObj.metaReplica.getCollectionByID(defaultCollectionID)
	assert.NoError(b, err)

	var segments []*Segment
	for i := 0; i < SEGMENTS; i++ {
		segmentID := UniqueID(i + 999)
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, segmentID)
		assert.NoError(b, err)
		// segment.BuildVecIndex(100)
		segments = append(segments, segment)
	}

	iReq, _ := genSearchRequest(NQ, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      nil,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	assert.NoError(b, err)

	// fmt.Println("finish prepare")
	// fmt.Println("start search b.N=", b.N)

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg := sync.WaitGroup{}
		for _, s := range segments {
			wg.Add(1)
			go func(s *Segment) {
				defer wg.Done()
				_, err = s.search(searchReq)
				assert.NoError(b, err)
			}(s)
		}
		wg.Wait()
	}
	// fmt.Println("finish search")
}

func BenchmarkSearchIndexSerial(b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())
	defer cancel()
	queryShardObj, err := genSimpleQueryShard(tx)
	assert.NoError(b, err)
	collection, err := queryShardObj.metaReplica.getCollectionByID(defaultCollectionID)
	assert.NoError(b, err)

	var segments []*Segment
	for i := 0; i < SEGMENTS; i++ {
		segmentID := UniqueID(i + 999)
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, segmentID)
		assert.NoError(b, err)
		segment.BuildVecIndex(100)
		segments = append(segments, segment)
	}

	iReq, _ := genSearchRequest(NQ, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      nil,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	assert.NoError(b, err)

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range segments {
			_, err = s.search(searchReq)
			assert.NoError(b, err)
		}
	}
}

func BenchmarkSearchIndexConcurrent(b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())
	defer cancel()
	queryShardObj, err := genSimpleQueryShard(tx)
	assert.NoError(b, err)
	collection, err := queryShardObj.metaReplica.getCollectionByID(defaultCollectionID)
	assert.NoError(b, err)

	var segments []*Segment
	for i := 0; i < SEGMENTS; i++ {
		segmentID := UniqueID(i + 999)
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, segmentID)
		assert.NoError(b, err)
		segment.BuildVecIndex(100)
		segments = append(segments, segment)
	}

	iReq, _ := genSearchRequest(NQ, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      nil,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	assert.NoError(b, err)

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg := sync.WaitGroup{}
		for _, s := range segments {
			wg.Add(1)
			go func(s *Segment) {
				defer wg.Done()
				_, err = s.search(searchReq)
				assert.NoError(b, err)
			}(s)
		}
		wg.Wait()
	}
}

func BenchmarkSearchIndexPool(b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())
	defer cancel()
	queryShardObj, err := genSimpleQueryShard(tx)
	assert.NoError(b, err)
	collection, err := queryShardObj.metaReplica.getCollectionByID(defaultCollectionID)
	assert.NoError(b, err)

	var segments []*Segment
	for i := 0; i < SEGMENTS; i++ {
		segmentID := UniqueID(i + 999)
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, segmentID)
		assert.NoError(b, err)
		segment.BuildVecIndex(100)
		segments = append(segments, segment)
	}

	iReq, _ := genSearchRequest(NQ, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      nil,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	assert.NoError(b, err)

	pool, err := concurrency.NewPool(getNumCPU(), ants.WithPreAlloc(true))
	if err != nil {
		log.Error("failed to create goroutine pool for segment loader",
			zap.Error(err))
		panic(err)
	}

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg := sync.WaitGroup{}
		for _, s := range segments {
			wg.Add(1)
			go func(s *Segment) {
				defer wg.Done()
				pool.Submit(func() (interface{}, error) {
					_, err = s.search(searchReq)
					assert.NoError(b, err)
					return nil, nil
				}).Await()
			}(s)
		}
		wg.Wait()
	}
}
