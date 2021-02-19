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
	mlmodelscopeSUT sut.SUT
	mlmodelscopeQSL dataset.Dataset
	rootSpan        opentracing.Span
	ctx             context.Context
)

// This needs to be call once from the python side in the start
func Initialize(backendName string, modelName string, modelVersion string,
	datasetName string, imageList string, count int, useGPU bool, traceLevel string) error {
	rootSpan, ctx = tracer.StartSpanFromContext(
		context.Background(),
		tracer.APPLICATION_TRACE,
		"MLCommons-MLModelScope",
	)
	if rootSpan == nil {
		panic("invalid span")
	}

	fmt.Println("Start initializing SUT...")

	mlmodelscopeSUT, err := sut.NewSUT(ctx, backendName, modelName, modelVersion, useGPU, traceLevel)
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

// This needs to be call once from the python side in the end
func Finalize() error {
	mlmodelscopeSUT.Close()
	rootSpan.Finish()
	return nil
}

func InfoModels(backendName string) error {
	return sut.InfoModels(backendName)
}
