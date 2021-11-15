// +build onnxruntime

package sut

import (
	"github.com/c3sr/onnxruntime"
	_ "github.com/c3sr/onnxruntime/predictor"
)

var (
	supportedBackend = map[string]backend{
		"onnxruntime": {onnxruntime.Register, onnxruntime.FrameworkManifest},
	}
)
