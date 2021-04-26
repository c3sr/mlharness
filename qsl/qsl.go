package qsl

import (
	"context"
	"fmt"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/tracer"
)

func NewQSL(ctx context.Context, datasetName string, dataPath string, imageList string, count int, preprocessOptions common.PreprocessOptions, preprocessMethod string) (dataset.Dataset, error) {
	initQSLSpan, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"new_qsl",
	)
	if initQSLSpan == nil {
		panic("invalid span")
	}
	defer initQSLSpan.Finish()

	switch datasetName {
	case "imagenet":
		return dataset.NewImageNet(dataPath, imageList, count, preprocessOptions, preprocessMethod)
	case "coco":
		return dataset.NewCoco(dataPath, imageList, count, preprocessOptions, preprocessMethod)
	}

	return nil, fmt.Errorf("%s dataset is not implemented", datasetName)
}
