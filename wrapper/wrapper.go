package main

import (
	"fmt"
  "runtime"
	"unsafe"

	base "github.com/c3sr/mlharness"
)

// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/tensorflow/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/libtorch/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/mxnet/lib
// #cgo LDFLAGS: -Wl,-rpath -Wl,/opt/onnxruntime/lib
import "C"

//export Initialize
func Initialize(cBackendName *C.char, cModelPath *C.char, cDatasetPath *C.char, cCount C.int,
	cUseGPU C.int, cGPUID C.int, cTraceLevel *C.char, cBatchSize C.int) *C.char {
  runtime.LockOSThread()
  defer runtime.UnlockOSThread()

	sz, err := base.Initialize(C.GoString(cBackendName), C.GoString(cModelPath), C.GoString(cDatasetPath), int(cCount),
		int(cUseGPU) != 0, int(cGPUID), C.GoString(cTraceLevel), int(cBatchSize))
	if err != nil {
		return C.CString(fmt.Sprintf("0, %s", err.Error()))
	}
	return C.CString(fmt.Sprintf("%d, nil", sz))
}

//export IssueQuery
func IssueQuery(cLen C.int, cSampleList *C.int) *C.char {
  runtime.LockOSThread()
  defer runtime.UnlockOSThread()
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
  runtime.LockOSThread()
  defer runtime.UnlockOSThread()
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
  runtime.LockOSThread()
  defer runtime.UnlockOSThread()
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
  runtime.LockOSThread()
  defer runtime.UnlockOSThread()
	if err := base.Finalize(); err != nil {
		return C.CString(err.Error())
	}

	return C.CString("")
}

func main() {}
