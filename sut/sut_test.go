package sut

import (
	"testing"
)

func TestInitializeSUT(t *testing.T) {
  err := InitializeSUT("pytorch", "torchvision_alexnet", "")
  if err != nil {
    t.Error(err)
  }
}
