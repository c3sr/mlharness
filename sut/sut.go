package sut

import (
	"fmt"

	"github.com/c3sr/dlframework"
	"github.com/c3sr/pytorch"
)

type backend struct {
	frameworkRegisterFunc func()
	frameworkManifest     dlframework.FrameworkManifest
}

var (
	supportedBackend = map[string]backend{
		"pytorch": {pytorch.Register, pytorch.FrameworkManifest},
	}
)

// InitializeSUT ...
func InitializeSUT(backendName string, modelName string, modelVersion string) error {
  // get backend
  value, ok := supportedBackend[backendName]
  if !ok {
    return fmt.Errorf("The backend %s is not supported.", backendName)
  }

  fmt.Printf("Use %s as backend.\n\n", backendName)

	frameworkRegister, framework := value.frameworkRegisterFunc, value.frameworkManifest

  frameworkRegister()

  if modelVersion == "" {
    fmt.Printf("Model version is empty, default to 1.0 ...\n")
    modelVersion = "1.0"
  }

  _, err := framework.FindModel(modelName + ":" + modelVersion)

  if err != nil {
    return err
  }

  return nil
}
