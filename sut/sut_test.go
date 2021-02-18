package sut

import (
  "fmt"
	"context"
	"testing"

	"github.com/c3sr/tracer"
)

func TestSUT(t *testing.T) {
	sut, err := NewSUT(context.Background(), "pytorch", "torchvision_alexnet", "", false, tracer.FULL_TRACE)
	if err != nil {
		t.Error(err)
	}
  opt, err := sut.GetPreprocessOptions()
  if err != nil {
    t.Error(err)
  }
  fmt.Println(opt)
}
