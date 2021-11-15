package mlharness

import (
	"context"
	"fmt"
	"runtime"
	"strconv"

	"github.com/c3sr/dldataset"
	"github.com/c3sr/dlframework/steps"
	"github.com/c3sr/go-python3"
	"github.com/c3sr/mlharness/sut"
	"github.com/c3sr/pipeline"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
)

var (
	mlharnessSUT        *sut.SUT
	mlharnessDataset    *dldataset.Dataset
	rootSpan            opentracing.Span
	ctx                 context.Context
	dataCount           int
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
	defaultChannelBuffer = 100000
)

// This needs to be call once from the python side in the beginning
func Initialize(backendName string, modelPath string, datasetPath string, count int,
	useGPU bool, GPUID int, traceLevel string, batchSize int) (int, error) {

	// Not from shared library
	if !python3.Py_IsInitialized() {
		python3.Py_Initialize()
		if !python3.Py_IsInitialized() {
			return 0, fmt.Errorf("Error initializing the python interpreter")
		}
		python3.PyEval_SaveThread()
	}

	if _, ok := supportedTraceLevel[traceLevel]; !ok {
		return 0, fmt.Errorf("%s is not a supported trace level", traceLevel)
	}

	var err error

	rootSpan, ctx = tracer.StartSpanFromContext(
		context.Background(),
		tracer.APPLICATION_TRACE,
		"MLHarness",
	)
	if rootSpan == nil {
		panic("invalid span")
	}

	fmt.Println("Start initializing SUT...")
	mlharnessSUT, err = sut.NewSUT(ctx, backendName, modelPath, useGPU, GPUID, traceLevel, batchSize)
	if err != nil {
		return 0, err
	}
	fmt.Println("Finish initializing SUT...")

	fmt.Println("Start initializing QSL...")

	mlharnessDataset, err := dldataset.NewDataset(datasetPath, count)
	if err != nil {
		return 0, err
	}

	fmt.Println("Finish initializing QSL...")

	tracer.SetLevel(tracer.NO_TRACE)

	modelManifest, err := mlharnessSUT.GetModelManifest()
	if err != nil {
		return 0, err
	}

	if modelManifest.GetBeforePreprocess() != "" {
		runtime.LockOSThread()
		pyState := python3.PyGILState_Ensure()
		python3.PyRun_SimpleString(modelManifest.GetBeforePreprocess())
		pyMain := python3.PyImport_AddModule("__main__")
		pyDict := python3.PyModule_GetDict(pyMain)
		pyBeforePreprocess := python3.PyDict_GetItemString(pyDict, "before_preprocess")
		pyCnt := pyBeforePreprocess.CallFunctionObjArgs()
		if python3.PyLong_Check(pyCnt) {
			dataCount = python3.PyLong_AsLong(pyCnt)
		}
		pyCnt.DecRef()
		python3.PyGILState_Release(pyState)
		runtime.UnlockOSThread()
	}
	if modelManifest.GetBeforePostprocess() != "" {
		runtime.LockOSThread()
		pyState := python3.PyGILState_Ensure()
		python3.PyRun_SimpleString(modelManifest.GetBeforePostprocess())
		pyMain := python3.PyImport_AddModule("__main__")
		pyDict := python3.PyModule_GetDict(pyMain)
		pyBeforePostprocess := python3.PyDict_GetItemString(pyDict, "before_postprocess")
		pyBeforePostprocess.CallFunctionObjArgs().DecRef()
		python3.PyGILState_Release(pyState)
		runtime.UnlockOSThread()
	}

	if err := warmup(); err != nil {
		return 0, err
	}

	tracer.SetLevel(tracer.LevelFromName(traceLevel))

	return mlharnessDataset.Count(), nil
}

func warmup() error {
	warmupSpan, issueCtx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"Warmup Span",
	)
	if warmupSpan == nil {
		panic("invalid issue query span")
	}
	defer warmupSpan.Finish()

	fmt.Println("Start warmup...")

	if err := mlharnessDataset.Load([]int{0}); err != nil {
		return err
	}

	input := make(chan interface{}, defaultChannelBuffer)
	opts := []pipeline.Option{pipeline.ChannelBuffer(defaultChannelBuffer)}

	preProcessMethod, err := mlharnessSUT.GetPreprocessMethod()
	if err != nil {
		return err
	}

	output := pipeline.New(opts...).
		Then(steps.NewPreprocessGeneral(preProcessMethod)).
		Run(input)

	tensors := make(map[int]interface{})
	input <- strconv.Itoa(0)
	close(input)

	for out := range output {
		if err, ok := out.(error); ok {
			return err
		}
		tensors[0] = out
	}

	for i := 0; i < 5; i++ {
		mlharnessSUT.ProcessQuery(issueCtx, tensors, []int{0})
	}

	if err := mlharnessDataset.Unload([]int{}); err != nil {
		return err
	}

	fmt.Println("Finish warmup...")
	return nil
}

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

	data, err := mlharnessDataset.GetAll()
	if err != nil {
		return "[[]]"
	}

	return mlharnessSUT.ProcessQuery(issueCtx, data, sampleList)
}

func LoadQuerySamples(sampleList []int) error {
	if err := mlharnessDataset.Load(sampleList); err != nil {
		return err
	}

	input := make(chan interface{}, defaultChannelBuffer)
	opts := []pipeline.Option{pipeline.ChannelBuffer(defaultChannelBuffer)}

	preProcessMethod, err := mlharnessSUT.GetPreprocessMethod()
	if err != nil {
		return err
	}

	output := pipeline.New(opts...).
		Then(steps.NewPreprocessGeneral(preProcessMethod)).
		Run(input)

	tensors := make(map[int]interface{})
	for _, sample := range sampleList {
		input <- strconv.Itoa(sample)
	}
	close(input)
	idx := 0
	for out := range output {
		if err, ok := out.(error); ok {
			return err
		}
		tensors[sampleList[idx]] = out
		idx++
	}

	return mlharnessDataset.Set(tensors)
}

func UnloadQuerySamples(sampleList []int) error {
	return mlharnessDataset.Unload(sampleList)
}

// This needs to be called once from the python side in the end
func Finalize() error {
	modelManifest, err := mlharnessSUT.GetModelManifest()
	if err != nil {
		return err
	}

	if modelManifest.GetAfterPreprocess() != "" {
		runtime.LockOSThread()
		pyState := python3.PyGILState_Ensure()
		python3.PyRun_SimpleString(modelManifest.GetAfterPreprocess())
		pyMain := python3.PyImport_AddModule("__main__")
		pyDict := python3.PyModule_GetDict(pyMain)
		pyAfterPreprocess := python3.PyDict_GetItemString(pyDict, "after_preprocess")
		pyAfterPreprocess.CallFunctionObjArgs().DecRef()
		python3.PyGILState_Release(pyState)
		runtime.UnlockOSThread()
	}
	if modelManifest.GetAfterPostprocess() != "" {
		runtime.LockOSThread()
		pyState := python3.PyGILState_Ensure()
		python3.PyRun_SimpleString(modelManifest.GetAfterPostprocess())
		pyMain := python3.PyImport_AddModule("__main__")
		pyDict := python3.PyModule_GetDict(pyMain)
		pyAfterPostprocess := python3.PyDict_GetItemString(pyDict, "after_postprocess")
		pyAfterPostprocess.CallFunctionObjArgs().DecRef()
		python3.PyGILState_Release(pyState)
		runtime.UnlockOSThread()
	}

	mlharnessSUT.Close()
	rootSpan.Finish()
	tracer.Close()

	return nil
}
