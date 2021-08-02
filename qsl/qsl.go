package qsl

import (
	"context"
	"fmt"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/tracer"
)

func NewQSL(ctx context.Context, datasetName string, dataPath string, dataList string, count int, preprocessOptions common.PreprocessOptions, preprocessMethod string) (dataset.Dataset, error) {
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
		return dataset.NewImageNet(dataPath, dataList, count, preprocessOptions, preprocessMethod)
	case "coco":
		return dataset.NewCoco(dataPath, dataList, count, preprocessOptions, preprocessMethod)
	case "squad":
		return dataset.NewSQuAD(dataPath, dataList, count, preprocessOptions, preprocessMethod)
	case "brats2019":
		return dataset.NewBraTS2019(dataPath, dataList, count, preprocessOptions, preprocessMethod)
	}

	return nil, fmt.Errorf("%s dataset is not implemented", datasetName)
}
