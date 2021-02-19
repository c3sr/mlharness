package sut

import (
	"context"
	"fmt"

	dl "github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/pytorch"
	_ "github.com/c3sr/pytorch/predictor"
	"github.com/c3sr/tracer"
	"github.com/pkg/errors"
)

type SUT struct {
	predictor common.Predictor
}

type backend struct {
	frameworkRegisterFunc func()
	frameworkManifest     dl.FrameworkManifest
}

var (
	supportedBackend = map[string]backend{
		"pytorch": {pytorch.Register, pytorch.FrameworkManifest},
	}
)

// NewSUT ...
func NewSUT(ctx context.Context, backendName string, modelName string, modelVersion string, useGPU bool, traceLevel tracer.Level) (*SUT, error) {

	initSUTSpan, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"new_sut",
	)
	if initSUTSpan == nil {
		panic("invalid span")
	}
	defer initSUTSpan.Finish()

	// get backend
	value, ok := supportedBackend[backendName]
	if !ok {
		return nil, fmt.Errorf("The backend %s is not supported.", backendName)
	}

	frameworkRegister, framework := value.frameworkRegisterFunc, value.frameworkManifest

	frameworkRegister()

	fmt.Printf("Use %s as backend.\n", framework.MustCanonicalName())

	if modelVersion == "" {
		fmt.Printf("Model version is empty, default to 1.0 ...\n")
		modelVersion = "1.0"
	}

	model, err := framework.FindModel(modelName + ":" + modelVersion)

	if err != nil {
		return nil, err
	}

	fmt.Printf("Found %s.\n", modelName+":"+modelVersion)

	// Use the same method in c3sr/dlframework to get predictors
	predictors, err := agent.GetPredictors(framework)
	if err != nil {
		return nil, errors.Wrapf(err,
			"⚠️ failed to get predictor for %s. make sure you have "+
				"imported the framework's predictor package",
			framework.MustCanonicalName(),
		)
	}

	// Use the same method in c3sr/dlframework to get modalities
	var predictorHandle common.Predictor
	for _, pred := range predictors {
		predModality, err := pred.Modality()
		if err != nil {
			continue
		}
		modelModality, err := model.Modality()
		if err != nil {
			continue
		}
		if predModality == modelModality {
			predictorHandle = pred
			break
		}
	}
	if predictorHandle == nil {
		return nil, errors.New("unable to find predictor for requested modality")
	}

	// setup options as we did in c3sr/dlframework
	var dc map[string]int32
	if useGPU {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("not gpu found")
		}
		dc = map[string]int32{"GPU": 0}
		log.WithField("gpu = ", nvidiasmi.Info.GPUS[0].ProductName).Info("Running evaluation on GPU")
	} else {
		dc = map[string]int32{"CPU": 0}
	}
	execOpts := &dl.ExecutionOptions{
		TraceLevel: dl.ExecutionOptions_TraceLevel(
			dl.ExecutionOptions_TraceLevel_value[traceLevel.String()],
		),
		DeviceCount: dc,
	}
	predOpts := &dl.PredictionOptions{
		ExecutionOptions: execOpts,
	}

	// Load predictor
	predictor, err := predictorHandle.Load(
		ctx,
		*model,
		options.PredictorOptions(predOpts),
	)
	if err != nil {
		return nil, err
	}

  fmt.Printf("Successfully initialized SUT with backend/model = %s.\n\n", model.MustCanonicalName())

	return &SUT{
		predictor: predictor,
	}, nil
}

func (s *SUT) GetPreprocessOptions() (common.PreprocessOptions, error) {
  return s.predictor.GetPreprocessOptions()
}
