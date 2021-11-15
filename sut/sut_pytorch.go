// +build pytorch

package sut

import (
	"github.com/c3sr/pytorch"
	_ "github.com/c3sr/pytorch/predictor"
)

var (
	supportedBackend = map[string]backend{
		"pytorch": {pytorch.Register, pytorch.FrameworkManifest},
	}
)
