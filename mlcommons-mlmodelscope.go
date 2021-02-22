package mlcommonsmlmomodelscope

import (
	"context"
	"fmt"
	"os"

	"github.com/c3sr/mlcommons-mlmodelscope/qsl"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/mlcommons-mlmodelscope/sut"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
)

var (
	mlmodelscopeSUT     *sut.SUT
	mlmodelscopeQSL     dataset.Dataset
	rootSpan            opentracing.Span
	ctx                 context.Context
	supportedTraceLevel = map[string]int{
		"NO_TRACE":             0,
		"APPLICATION_TRACE":    1,
		"MODEL_TRACE":          2,
		"FRAMEWORK_TRACE":      3,
		"ML_LIBRARY_TRACE":     4,
		"SYSTEM_LIBRARY_TRACE": 5,
		"HARDWARE_TRACE":       6,
		"FULL_TRACE":           7,
	}
)

// This needs to be call once from the python side in the start
func Initialize(backendName string, modelName string, modelVersion string,
	datasetName string, imageList string, count int, useGPU bool, traceLevel string, batchSize int) (int, error) {

	if _, ok := supportedTraceLevel[traceLevel]; !ok {
		return 0, fmt.Errorf("%s is not a supported trace level", traceLevel)
	}

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

	mlmodelscopeSUT, err = sut.NewSUT(ctx, backendName, modelName, modelVersion, useGPU, traceLevel, batchSize)
	if err != nil {
		return 0, err
	}

	fmt.Println("Finish initializing SUT...")

	opt, err := mlmodelscopeSUT.GetPreprocessOptions()
	if err != nil {
		return 0, err
	}

	path := os.Getenv("DATA_DIR")

	fmt.Println("Start initializing QSL...")

	mlmodelscopeQSL, err = qsl.NewQSL(ctx, datasetName, path, imageList, count, opt)
	if err != nil {
		return 0, err
	}

	fmt.Println("Finish initializing QSL...")

	if batchSize < 1 || batchSize > 128 {
		return 0, fmt.Errorf("Please give a batchsize between 1 and 128, right now is %d.", batchSize)
	}

	if err := warmup(); err != nil {
		return 0, nil
	}

	return mlmodelscopeQSL.GetItemCount(), nil
}

func warmup() error {
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
		mlmodelscopeSUT.ProcessQuery(issueCtx, data, []int{0})
	}

	if err := UnloadQuerySamples([]int{}); err != nil {
		return err
	}

	fmt.Println("Finish warmup...")
	return nil
}

// TODO: What do we want to return to python?
func IssueQuery(sampleList []int) string {
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
		return "[[]]"
	}

	return mlmodelscopeSUT.ProcessQuery(issueCtx, data, sampleList)
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

// This needs to be called once from the python side in the end
func Finalize() error {
	mlmodelscopeSUT.Close()
	rootSpan.Finish()
	tracer.Close()
	return nil
}
