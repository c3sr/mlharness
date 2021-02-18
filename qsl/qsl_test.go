package qsl

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/c3sr/mlcommons-mlmodelscope/sut"
	"github.com/c3sr/tracer"
)

func TestQSL(t *testing.T) {
	sut, err := sut.NewSUT(context.Background(), "pytorch", "torchvision_alexnet", "", false, tracer.FULL_TRACE)
	if err != nil {
		t.Error(err)
	}

	opt, err := sut.GetPreprocessOptions()
	if err != nil {
		t.Error(err)
	}

	path, _ := os.Getwd()
	path = filepath.Join(path, "dataset/_fixture/fake_imagenet")

	_, err = NewQSL(context.Background(), "imagenet", path, "", 5, opt)
	if err != nil {
		t.Error(err)
	}

}
