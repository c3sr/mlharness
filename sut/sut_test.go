package sut

import (
	"context"
	"testing"
)

func TestSUT(t *testing.T) {
	sut, err := NewSUT(context.Background(), "pytorch", "torchvision_alexnet", "", false, "FULL_TRACE")
	if err != nil {
		t.Error(err)
	}

	_, err = sut.GetPreprocessOptions()
	if err != nil {
		t.Error(err)
	}
}
