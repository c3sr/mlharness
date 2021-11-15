package sut

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"reflect"

	dl "github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/dlframework/steps"
	dlmodel "github.com/c3sr/dlmodel/cmd"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/pipeline"
	"github.com/c3sr/tracer"
	"github.com/pkg/errors"
	"gopkg.in/yaml.v2"
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
	defaultChannelBuffer = 100000
)

// NewSUT ...
func NewSUT(ctx context.Context, backendName string, modelPath string,
	useGPU bool, GPUID int, traceLevel string, batchSize int) (*SUT, error) {

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

	if err := dlmodel.DownloadPattern(modelPath); err != nil {
		return nil, err
	}

	bts, err := ioutil.ReadFile(modelPath)
	if err != nil {
		return nil, err
	}

	var modelManifest dl.ModelManifest
	if err := yaml.Unmarshal(bts, &modelManifest); err != nil {
		return nil, err
	}

	model, err := framework.FindModel(modelManifest.GetName() + ":" + modelManifest.GetVersion())
	if err != nil {
		return nil, err
	}

	fmt.Printf("Found %s.\n", modelManifest.GetName()+":"+modelManifest.GetVersion())

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
		if predModality != "general" {
			continue
		}
		predictorHandle = pred
	}
	if predictorHandle == nil {
		return nil, errors.New("unable to find predictor for requested modality")
	}

	// setup options as we did in c3sr/dlframework
	var dc map[string]int32
	if useGPU {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no gpu found")
		}
		dc = map[string]int32{"GPU": int32(GPUID)}
		log.Info("Running evaluation on GPU")
	} else {
		dc = map[string]int32{"CPU": 0}
		log.Info("Running evaluation on CPU")
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

	fmt.Printf("Successfully initialized SUT with %s.\n", model.MustCanonicalName())

	if batchSize < 1 {
		fmt.Printf("Batchsize = %d is not supported, default to 1.\n", batchSize)
		batchSize = 1
	}

	return &SUT{
		predictor: predictor,
		batchSize: batchSize,
	}, nil
}

func (s *SUT) GetModelManifest() (dl.ModelManifest, error) {
	_, modelManifest, err := s.predictor.Info()
	return modelManifest, err
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

func (s *SUT) imageObjectDetection(ctx context.Context, data map[int]interface{}, sampleList []int) string {
	_, modelManifest, err := s.predictor.Info()
	if err != nil {
		return "[[]]"
	}
	input := make(chan interface{}, defaultChannelBuffer)

	output := pipeline.New(pipeline.Context(ctx), pipeline.ChannelBuffer(defaultChannelBuffer)).
		Then(steps.NewPredictGeneral(s.predictor, modelManifest.GetPostprocess())).
		Run(input)

	resSlice := make([][][]float32, len(sampleList))

	for st := 0; st < len(sampleList); st += s.batchSize {
		ed := st + s.batchSize
		if ed > len(sampleList) {
			ed = len(sampleList)
		}
		cur := make([]interface{}, ed-st)
		for i := 0; i < len(cur); i++ {
			cur[i] = data[sampleList[st+i]]
		}
		reflect.ValueOf(s.predictor).Convert(reflect.TypeOf(s.predictor)).Elem().FieldByName("Options").Interface().(*options.Options).SetBatchSize(ed - st)
		input <- cur
		for j := 0; j < ed-st; j++ {
			out0 := <-output

			out, ok := out0.(steps.IDer)
			if !ok {
				return "[[]]"
			}
			for _, f := range out.GetData().(dl.Features) {
				resSlice[st+j] = append(resSlice[st+j], []float32{float32(sampleList[st+j]), f.GetBoundingBox().GetYmin(), f.GetBoundingBox().GetXmin(),
					f.GetBoundingBox().GetYmax(), f.GetBoundingBox().GetXmax(), f.GetProbability(), float32(f.GetBoundingBox().GetIndex())})
			}
		}
	}

	close(input)

	resJSON, _ := json.Marshal(resSlice)
	return string(resJSON)
}

func (s *SUT) generalTask(ctx context.Context, data map[int]interface{}, sampleList []int) string {
	_, modelManifest, err := s.predictor.Info()
	if err != nil {
		return "[[]]"
	}
	input := make(chan interface{}, defaultChannelBuffer)

	output := pipeline.New(pipeline.Context(ctx), pipeline.ChannelBuffer(defaultChannelBuffer)).
		Then(steps.NewPredictGeneral(s.predictor, modelManifest.GetPostprocess())).
		Run(input)

	resJSON := []byte{'['}

	for st := 0; st < len(sampleList); st += s.batchSize {
		ed := st + s.batchSize
		if ed > len(sampleList) {
			ed = len(sampleList)
		}
		cur := make([]interface{}, ed-st)
		for i := 0; i < len(cur); i++ {
			cur[i] = data[sampleList[st+i]]
		}
		reflect.ValueOf(s.predictor).Convert(reflect.TypeOf(s.predictor)).Elem().FieldByName("Options").Interface().(*options.Options).SetBatchSize(ed - st)
		input <- cur
		for j := 0; j < ed-st; j++ {
			out0 := <-output

			out, ok := out0.(steps.IDer)
			if !ok {
				return "[[]]"
			}
			for _, f := range out.GetData().(dl.Features) {
				resJSON = append(resJSON, f.GetText().GetData()...)
				resJSON = append(resJSON, ',')
			}
		}
	}

	close(input)

	resJSON[len(resJSON)-1] = ']'
	return string(resJSON)
}

func (s *SUT) ProcessQuery(ctx context.Context, data map[int]interface{}, sampleList []int) string {
	_, modelManifest, err := s.predictor.Info()
	if err != nil {
		return "[[]]"
	}
	switch strings.ToLower(modelManifest.GetModality()) {
	case "image_object_detection":
		return s.imageObjectDetection(ctx, data, sampleList)
	default:
		return s.generalTask(ctx, data, sampleList)
	}
	return "[[]]"
}
