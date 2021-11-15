// +build mxnet

package sut

import (
	"github.com/c3sr/mxnet"
	_ "github.com/c3sr/mxnet/predictor"
)

var (
	supportedBackend = map[string]backend{
		"mxnet": {mxnet.Register, mxnet.FrameworkManifest},
	}
)
