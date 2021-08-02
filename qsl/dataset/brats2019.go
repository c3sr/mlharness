package dataset

import (
	"fmt"
	"strconv"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/dlframework/steps"
	"github.com/c3sr/pipeline"
)

type BraTS2019 struct {
	names             []string
	dataPath          string
	dataInMemory      map[int]interface{}
	preprocessOptions common.PreprocessOptions
	preprocessMethod  string
}

func NewBraTS2019(dataPath string, dataList string, count int, preprocessOptions common.PreprocessOptions, preprocessMethod string) (*BraTS2019, error) {

	res := &BraTS2019{
		dataPath:          dataPath,
		preprocessOptions: preprocessOptions,
		preprocessMethod:  preprocessMethod,
	}

	return res, nil
}

func (s *BraTS2019) LoadQuerySamples(sampleList []int) error {
	s.dataInMemory = make(map[int]interface{})

	input := make(chan interface{}, defaultChannelBuffer)
	opts := []pipeline.Option{pipeline.ChannelBuffer(defaultChannelBuffer)}
	output := pipeline.New(opts...).
		Then(steps.NewPreprocessGeneral(s.preprocessOptions, s.preprocessMethod)).
		Run(input)

	for _, sample := range sampleList {
		input <- strconv.Itoa(sample)
	}

	close(input)

	for _, sample := range sampleList {
		s.dataInMemory[sample] = <-output
	}
	return nil
}

func (s *BraTS2019) UnloadQuerySamples(sampleList []int) error {
	if s.dataInMemory == nil {
		return fmt.Errorf("Data map is nil.")
	}
	if len(sampleList) == 0 {
		s.dataInMemory = nil
	} else {
		for _, sample := range sampleList {
			delete(s.dataInMemory, sample)
		}
	}
	return nil
}

func (s *BraTS2019) GetItemCount() int {
	return -1
}

func (s *BraTS2019) GetSamples(sampleList []int) (map[int]interface{}, error) {
	for _, sample := range sampleList {
		if _, exist := s.dataInMemory[sample]; !exist {
			return nil, fmt.Errorf("sample id %d not loaded.", sample)
		}
	}
	return s.dataInMemory, nil
}
