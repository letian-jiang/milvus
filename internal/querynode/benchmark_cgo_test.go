package querynode

import (
	"context"
	"testing"

	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"

	"github.com/milvus-io/milvus/internal/log"
)

const (
	NQ       = 10
	TOPK     = 10
	SEGMENTS = 100
	RECORDS  = 10
	WARMUP   = 10
	COUNT    = 10
)

func benchmarkSearch(nq int64, b *testing.B) {
	log.SetLevel(zapcore.ErrorLevel)
	defer log.SetLevel(zapcore.DebugLevel)

	tx, cancel := context.WithCancel(context.Background())
	defer cancel()
	queryShardObj, err := genSimpleQueryShard(tx)
	assert.NoError(b, err)

	assert.Equal(b, 0, queryShardObj.metaReplica.getSegmentNum(segmentTypeSealed))
	assert.Equal(b, 0, queryShardObj.metaReplica.getSegmentNum(segmentTypeGrowing))

	var segmentIDs []UniqueID
	for i := 0; i < SEGMENTS; i++ {
		segment, err := genSimpleSealedSegmentWithSegmentID(RECORDS, i)
		assert.NoError(b, err)
		err = queryShardObj.metaReplica.setSegment(segment)
		assert.NoError(b, err)
		segmentIDs = append(segmentIDs, UniqueID(i))
	}

	assert.Equal(b, SEGMENTS, queryShardObj.metaReplica.getSegmentNum(segmentTypeSealed))
	assert.Equal(b, 0, queryShardObj.metaReplica.getSegmentNum(segmentTypeGrowing))

	collection, err := queryShardObj.metaReplica.getCollectionByID(defaultCollectionID)
	assert.NoError(b, err)

	iReq, _ := genSearchRequest(nq, IndexFaissIDMap, collection.schema)
	queryReq := &queryPb.SearchRequest{
		Req:             iReq,
		DmlChannels:     []string{defaultDMLChannel},
		SegmentIDs:      segmentIDs,
		FromShardLeader: true,
		Scope:           queryPb.DataScope_Historical,
	}
	searchReq, err := newSearchRequest(collection, queryReq, queryReq.Req.GetPlaceholderGroup())
	assert.NoError(b, err)

	for i := 0; i < WARMUP; i++ {
		_, _, _, err := searchHistorical(queryShardObj.metaReplica, searchReq, defaultCollectionID, nil, queryReq.GetSegmentIDs())
		assert.NoError(b, err)
	}

	// f, err := os.Create("nq_" + strconv.Itoa(int(nq)) + ".perf")
	// if err != nil {
	// 	panic(err)
	// }
	// if err = pprof.StartCPUProfile(f); err != nil {
	// 	panic(err)
	// }
	// defer pprof.StopCPUProfile()

	// start benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := int64(0); j < COUNT; j++ {
			_, _, _, err := searchHistorical(queryShardObj.metaReplica, searchReq, defaultCollectionID, nil, queryReq.GetSegmentIDs())
			assert.NoError(b, err)
		}
	}
}

func BenchmarkSearch(b *testing.B) { benchmarkSearch(1, b) }
