// +build tensorflow

package sut

import (
	"github.com/c3sr/tensorflow"
	_ "github.com/c3sr/tensorflow/predictor"
)

var (
	supportedBackend = map[string]backend{
		"tensorflow": {tensorflow.Register, tensorflow.FrameworkManifest},
	}
)
