package qsl

import (
	"context"
	"fmt"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/tracer"
)

type QSL struct {
	dataset.Dataset
}

func NewQSL(ctx context.Context, datasetName string, dataPath string, imageList string, count int, preprocessOptions common.PreprocessOptions) (*QSL, error) {
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
		res, err := dataset.NewImageNet(dataPath, imageList, count, preprocessOptions)
		if err != nil {
			return nil, err
		}
		return &QSL{
			Dataset: res,
		}, nil
		// case "coco":
		//   return &QSL{
		//     Dataset: dataset.NewCoco(preprocessOptions),
		//   }, nil
	}

  fmt.Printf("Successfully initialized QSL with dataset = %s.\n\n", datasetName)

	return nil, fmt.Errorf("%s dataset is not implemented", datasetName)
}
