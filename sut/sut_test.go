package sut

import (
	"context"
	"testing"
)

func TestSUT(t *testing.T) {
	sut, err := NewSUT(context.Background(), "onnxruntime", "MLPerf_ResNet50_v1.5", "", false, "FULL_TRACE", 1)
	if err != nil {
		t.Error(err)
	}

	_, err = sut.GetPreprocessOptions()
	if err != nil {
		t.Error(err)
	}
}
