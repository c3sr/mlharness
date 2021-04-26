package sut

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"

	dl "github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/dlframework/steps"
	// "github.com/c3sr/mxnet"
	// _ "github.com/c3sr/mxnet/predictor"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/onnxruntime"
	_ "github.com/c3sr/onnxruntime/predictor"
	"github.com/c3sr/pipeline"
	// "github.com/c3sr/pytorch"
	// _ "github.com/c3sr/pytorch/predictor"
	// "github.com/c3sr/tensorflow"
	// _ "github.com/c3sr/tensorflow/predictor"
	"github.com/c3sr/tracer"
	"github.com/olekukonko/tablewriter"
	"github.com/pkg/errors"
)

type SUT struct {
	predictor common.Predictor
	batchSize int
}

type backend struct {
	frameworkRegisterFunc func()
	frameworkManifest     dl.FrameworkManifest
}

var (
	supportedBackend = map[string]backend{
		// "pytorch":     {pytorch.Register, pytorch.FrameworkManifest},
		"onnxruntime": {onnxruntime.Register, onnxruntime.FrameworkManifest},
		// "tensorflow":  {tensorflow.Register, tensorflow.FrameworkManifest},
		// "mxnet":       {mxnet.Register, mxnet.FrameworkManifest},
	}
	defaultChannelBuffer = 100000
)

// NewSUT ...
func NewSUT(ctx context.Context, backendName string, modelName string,
	modelVersion string, useGPU bool, traceLevel string, batchSize int) (*SUT, error) {

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
		BatchSize:        int32(batchSize),
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

	if batchSize > 128 || batchSize < 1 {
		fmt.Printf("Batchsize = %d is not supported, default to 1.\n", batchSize)
		batchSize = 1
	}

	return &SUT{
		predictor: predictor,
		batchSize: batchSize,
	}, nil
}

func (s *SUT) GetPreprocessOptions() (common.PreprocessOptions, error) {
	return s.predictor.GetPreprocessOptions()
}

func (s *SUT) GetPreprocessMethod() (string, error) {
	_, modelManifest, err := s.predictor.Info()
	if err != nil {
		return "", fmt.Errorf("Unable to get preprocess method")
	}

	return modelManifest.GetPreprocess(), nil
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

func (s *SUT) ProcessQuery(ctx context.Context, data []interface{}, sampleList []int) string {
	input := make(chan interface{}, defaultChannelBuffer)

	_, modelManifest, err := s.predictor.Info()
	if err != nil {
		return "[[]]"
	}

	output := pipeline.New(pipeline.Context(ctx), pipeline.ChannelBuffer(defaultChannelBuffer)).
		Then(steps.NewPredictGeneral(s.predictor, modelManifest.GetPostprocess())).
		Run(input)

	imageParts := dl.Partition(data, s.batchSize)
	res := make([]dl.Features, len(data))

	for i, d := range imageParts {

		reflect.ValueOf(s.predictor).Convert(reflect.TypeOf(s.predictor)).Elem().FieldByName("Options").Interface().(*options.Options).SetBatchSize(len(d))

		input <- d
		for j := 0; j < len(d); j++ {
			out0 := <-output

			out, ok := out0.(steps.IDer)
			if !ok {
				return "[[]]"
			}
			res[i*s.batchSize+j] = out.GetData().(dl.Features)
		}
	}

	close(input)

	modelModality, _ := s.predictor.Modality()

	switch modelModality {
	case "image_classification":
		resSlice := make([][]float32, len(data))
		for i := 0; i < len(data); i++ {
			resSlice[i] = []float32{float32(res[i][0].GetClassification().GetIndex())}
		}
		resJSON, _ := json.Marshal(resSlice)
		return string(resJSON)
	case "image_object_detection":
		resSlice := make([][][]float32, len(data))
		for i := 0; i < len(data); i++ {
			for _, f := range res[i] {
				resSlice[i] = append(resSlice[i], []float32{float32(sampleList[i]), f.GetBoundingBox().GetYmin(), f.GetBoundingBox().GetXmin(),
					f.GetBoundingBox().GetYmax(), f.GetBoundingBox().GetXmax(), f.GetProbability(), float32(f.GetBoundingBox().GetIndex())})
			}
		}
		resJSON, _ := json.Marshal(resSlice)
		return string(resJSON)
	}

	return ""
}
