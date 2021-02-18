package sut

import (
	"context"
	"testing"

	"github.com/c3sr/tracer"
)

func TestNewSUT(t *testing.T) {
	_, err := NewSUT(context.Background(), "pytorch", "torchvision_alexnet", "", false, tracer.FULL_TRACE)
	if err != nil {
		t.Error(err)
	}
}
