package mlcommonsmlmomodelscope

import (
	"context"
	"fmt"
	"os"
  "runtime"

	"github.com/c3sr/go-python3"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl"
	"github.com/c3sr/mlcommons-mlmodelscope/qsl/dataset"
	"github.com/c3sr/mlcommons-mlmodelscope/sut"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
)

var (
	mlmodelscopeSUT     *sut.SUT
	mlmodelscopeQSL     dataset.Dataset
	rootSpan            opentracing.Span
	ctx                 context.Context
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
)

// This needs to be call once from the python side in the start
func Initialize(backendName string, modelName string, modelVersion string,
	datasetName string, imageList string, count int, useGPU bool, traceLevel string, batchSize int) (int, error) {

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
		"MLCommons-MLModelScope",
	)
	if rootSpan == nil {
		panic("invalid span")
	}

	fmt.Println("Start initializing SUT...")

	mlmodelscopeSUT, err = sut.NewSUT(ctx, backendName, modelName, modelVersion, useGPU, traceLevel, batchSize)
	if err != nil {
		return 0, err
	}

	fmt.Println("Finish initializing SUT...")

	opt, err := mlmodelscopeSUT.GetPreprocessOptions()
	if err != nil {
		return 0, err
	}

	preprocessMethod, err := mlmodelscopeSUT.GetPreprocessMethod()
	if err != nil {
		return 0, err
	}

	path := os.Getenv("DATA_DIR")

	fmt.Println("Start initializing QSL...")

	mlmodelscopeQSL, err = qsl.NewQSL(ctx, datasetName, path, imageList, count, opt, preprocessMethod)
	if err != nil {
		return 0, err
	}

	fmt.Println("Finish initializing QSL...")

	tracer.SetLevel(tracer.NO_TRACE)

  modelManifest, err := mlmodelscopeSUT.GetModelManifest()
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
    pyBeforePreprocess.CallFunctionObjArgs().DecRef()
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

	return mlmodelscopeQSL.GetItemCount(), nil
}

func warmup() error {
	wamupSpan, issueCtx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"Warmup Span",
	)
	if wamupSpan == nil {
		panic("invalid issue query span")
	}
	defer wamupSpan.Finish()

	fmt.Println("Start warmup...")

	if err := LoadQuerySamples([]int{0}); err != nil {
		return err
	}

	data, err := mlmodelscopeQSL.GetSamples([]int{0})
	if err != nil {
		return err
	}

	for ii := 0; ii < 5; ii++ {
		mlmodelscopeSUT.ProcessQuery(issueCtx, data, []int{0})
	}

	if err := UnloadQuerySamples([]int{}); err != nil {
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

	data, err := mlmodelscopeQSL.GetSamples(sampleList)
	if err != nil {
		return "[[]]"
	}

	return mlmodelscopeSUT.ProcessQuery(issueCtx, data, sampleList)
}

func LoadQuerySamples(sampleList []int) error {
	return mlmodelscopeQSL.LoadQuerySamples(sampleList)
}

func UnloadQuerySamples(sampleList []int) error {
	return mlmodelscopeQSL.UnloadQuerySamples(sampleList)
}

func InfoModels(backendName string) error {
	return sut.InfoModels(backendName)
}

// This needs to be called once from the python side in the end
func Finalize() error {
  modelManifest, err := mlmodelscopeSUT.GetModelManifest()
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

	mlmodelscopeSUT.Close()
	rootSpan.Finish()
	tracer.Close()

	return nil
}
