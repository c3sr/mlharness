package sut

import (
	"context"
	"fmt"
	"os"

	dl "github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/dlframework/steps"
	"github.com/c3sr/mxnet"
	_ "github.com/c3sr/mxnet/predictor"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/onnxruntime"
	_ "github.com/c3sr/onnxruntime/predictor"
	"github.com/c3sr/pipeline"
	"github.com/c3sr/pytorch"
	_ "github.com/c3sr/pytorch/predictor"
	"github.com/c3sr/tensorflow"
	_ "github.com/c3sr/tensorflow/predictor"
	"github.com/c3sr/tracer"
	"github.com/olekukonko/tablewriter"
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
		"pytorch":     {pytorch.Register, pytorch.FrameworkManifest},
		"onnxruntime": {onnxruntime.Register, onnxruntime.FrameworkManifest},
		"tensorflow":  {tensorflow.Register, tensorflow.FrameworkManifest},
		"mxnet":       {mxnet.Register, mxnet.FrameworkManifest},
	}
	defaultChannelBuffer = 100000
)

// NewSUT ...
func NewSUT(ctx context.Context, backendName string, modelName string, modelVersion string, useGPU bool, traceLevel string) (*SUT, error) {

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
			dl.ExecutionOptions_TraceLevel_value[traceLevel],
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

	fmt.Printf("Successfully initialized SUT with backend/model = %s.\n", model.MustCanonicalName())

	return &SUT{
		predictor: predictor,
	}, nil
}

func (s *SUT) GetPreprocessOptions() (common.PreprocessOptions, error) {
	return s.predictor.GetPreprocessOptions()
}

func (s *SUT) Close() {
	s.predictor.Close()
}

func InfoModels(backendName string) error {
	// get backend
	value, ok := supportedBackend[backendName]
	if !ok {
		return fmt.Errorf("The backend %s is not supported.", backendName)
	}

	frameworkRegister, framework := value.frameworkRegisterFunc, value.frameworkManifest

	frameworkRegister()

	models := framework.Models()
	if len(models) == 0 {
		fmt.Println("No Models")
		return nil
	}

	tbl := tablewriter.NewWriter(os.Stdout)
	tbl.SetHeader([]string{"Name", "Version", "Cannonical Name"})
	for _, model := range models {
		tbl.Append([]string{
			model.Name,
			model.Version,
			model.MustCanonicalName(),
		})
	}
	tbl.Render()

	return nil
}

func (s *SUT) ProcessQuery(ctx context.Context, data []interface{}) ([]dl.Features, error) {
	input := make(chan interface{}, defaultChannelBuffer)

	go func() {
		defer close(input)
		for _, d := range data {
			input <- []interface{}{d}
		}
	}()

	output := pipeline.New(pipeline.Context(ctx), pipeline.ChannelBuffer(defaultChannelBuffer)).
		Then(steps.NewPredict(s.predictor)).
		Run(input)

	res := make([]dl.Features, len(data))

	for ii, _ := range data {
		out0 := <-output

		out, ok := out0.(steps.IDer)
		if !ok {
			return nil, fmt.Errorf("expecting steps.IDer, but got %v", out0)
		}

		res[ii] = out.GetData().(dl.Features)
	}

	return res, nil
}
