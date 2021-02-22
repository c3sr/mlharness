package main

import (
	"fmt"
	"unsafe"

	base "github.com/c3sr/mlcommons-mlmodelscope"
)

// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/tensorflow/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/libtorch/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/mxnet/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/onnxruntime/lib
import "C"

//export Initialize
func Initialize(cBackendName *C.char, cModelName *C.char, cModelVersion *C.char, cDatasetName *C.char,
	cImageList *C.char, cCount C.int, cUseGPU C.int, cTraceLevel *C.char, cBatchSize C.int) *C.char {

	sz, err := base.Initialize(C.GoString(cBackendName), C.GoString(cModelName), C.GoString(cModelVersion), C.GoString(cDatasetName),
		C.GoString(cImageList), int(cCount), int(cUseGPU) != 0, C.GoString(cTraceLevel), int(cBatchSize))
	if err != nil {
		return C.CString(fmt.Sprintf("0, %s", err.Error()))
	}
	return C.CString(fmt.Sprintf("%d, nil", sz))
}

//export IssueQuery
func IssueQuery(cLen C.int, cSampleList *C.int) *C.char {
	len := int(cLen)
	slice := (*[1 << 30]C.int)(unsafe.Pointer(cSampleList))[:len:len]
	sampleList := make([]int, len)

	for i := 0; i < len; i++ {
		sampleList[i] = int(slice[i])
	}

	return C.CString(base.IssueQuery(sampleList))
}

//export LoadQuerySamples
func LoadQuerySamples(cLen C.int, cSampleList *C.int) *C.char {
	len := int(cLen)
	slice := (*[1 << 30]C.int)(unsafe.Pointer(cSampleList))[:len:len]
	sampleList := make([]int, len)

	for i := 0; i < len; i++ {
		sampleList[i] = int(slice[i])
	}

	if err := base.LoadQuerySamples(sampleList); err != nil {
		return C.CString(err.Error())
	}

	return C.CString("")
}

//export UnloadQuerySamples
func UnloadQuerySamples(cLen C.int, cSampleList *C.int) *C.char {
	len := int(cLen)
	slice := (*[1 << 30]C.int)(unsafe.Pointer(cSampleList))[:len:len]
	sampleList := make([]int, len)

	for i := 0; i < len; i++ {
		sampleList[i] = int(slice[i])
	}

	if err := base.UnloadQuerySamples(sampleList); err != nil {
		return C.CString(err.Error())
	}

	return C.CString("")
}

//export Finalize
func Finalize() *C.char {
	if err := base.Finalize(); err != nil {
		return C.CString(err.Error())
	}

	return C.CString("")
}

//export InfoModels
func InfoModels(cBackendName *C.char) *C.char {
	if err := base.InfoModels(C.GoString(cBackendName)); err != nil {
		return C.CString(err.Error())
	}

	return C.CString("")
}

func main() {}
