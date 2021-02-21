package main

import (
	"fmt"

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

func main() {}
