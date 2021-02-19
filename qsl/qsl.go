package qsl

import (
	"context"
	"fmt"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/tracer"
)

func NewQSL(ctx context.Context, datasetName string, dataPath string, imageList string, count int, preprocessOptions common.PreprocessOptions) (dataset.Dataset, error) {
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
		return dataset.NewImageNet(dataPath, imageList, count, preprocessOptions)
		// case "coco":
		//   return &QSL{
		//     Dataset: dataset.NewCoco(preprocessOptions),
		//   }, nil

	}

	return nil, fmt.Errorf("%s dataset is not implemented", datasetName)
}
