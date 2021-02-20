package mlcommonsmlmomodelscope

import (
	"context"
	"fmt"
	"os"

	dl "github.com/c3sr/dlframework"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/mlcommons-mlmodelscope/sut"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
)

var (
	mlmodelscopeSUT *sut.SUT
	mlmodelscopeQSL dataset.Dataset
	rootSpan        opentracing.Span
	ctx             context.Context
)

// This needs to be call once from the python side in the start
func Initialize(backendName string, modelName string, modelVersion string,
	datasetName string, imageList string, count int, useGPU bool, traceLevel string) error {

	var err error

	rootSpan, ctx = tracer.StartSpanFromContext(
		context.Background(),
		tracer.APPLICATION_TRACE,
		"MLCommons-MLModelScope",
	)
	if rootSpan == nil {
		panic("invalid span")
	}

	fmt.Println("Start initializing SUT...")

	mlmodelscopeSUT, err = sut.NewSUT(ctx, backendName, modelName, modelVersion, useGPU, traceLevel)
	if err != nil {
		return err
	}

	fmt.Println("Finish initializing SUT...")

	opt, err := mlmodelscopeSUT.GetPreprocessOptions()
	if err != nil {
		return err
	}

	path := os.Getenv("DATA_DIR")

	fmt.Println("Start initializing QSL...")

	mlmodelscopeQSL, err = qsl.NewQSL(ctx, datasetName, path, imageList, count, opt)
	if err != nil {
		return err
	}

	fmt.Println("Finish initializing QSL...")

	return nil
}

func Warmup() error {
	wamupSpan, issueCtx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"Warmup Span",
	)
	if wamupSpan == nil {
		panic("invalid issue query span")
	}
	defer wamupSpan.Finish()

	fmt.Println("Start warmup...")

	if err := LoadQuerySamples([]int{0}); err != nil {
		return err
	}

	data, err := mlmodelscopeQSL.GetSamples([]int{0})
	if err != nil {
		return err
	}

	for ii := 0; ii < 5; ii++ {
		if _, err := mlmodelscopeSUT.ProcessQuery(issueCtx, data); err != nil {
			return err
		}
	}

	if err := UnloadQuerySamples([]int{}); err != nil {
		return err
	}

	fmt.Println("Finish warmup...")
	return nil
}

// TODO: What do we want to return to python?
func IssueQuery(sampleList []int) ([]dl.Features, error) {
	issueSpan, issueCtx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"IssueQuery Span",
	)
	if issueSpan == nil {
		panic("invalid issue query span")
	}
	defer issueSpan.Finish()

	data, err := mlmodelscopeQSL.GetSamples(sampleList)
	if err != nil {
		return nil, err
	}

	return mlmodelscopeSUT.ProcessQuery(issueCtx, data)
}

func LoadQuerySamples(sampleList []int) error {
	return mlmodelscopeQSL.LoadQuerySamples(sampleList)
}

func UnloadQuerySamples(sampleList []int) error {
	return mlmodelscopeQSL.UnloadQuerySamples(sampleList)
}

func InfoModels(backendName string) error {
	return sut.InfoModels(backendName)
}

// This needs to be call once from the python side in the end
func Finalize() error {
	mlmodelscopeSUT.Close()
	rootSpan.Finish()
	tracer.Close()
	return nil
}
