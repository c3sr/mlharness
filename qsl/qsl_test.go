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

	qsl, err := NewQSL(context.Background(), "imagenet", path, "", 5, opt)
	if err != nil {
		t.Error(err)
	}

	if qsl.GetItemCount() != 5 {
		t.Errorf("item count doesn't match.")
	}

	err = qsl.LoadQuerySamples([]int{0, 1, 2, 3, 4})
	if err != nil {
		t.Error(err)
	}

	err = qsl.UnloadQuerySamples([]int{1, 3})
	if err != nil {
		t.Error(err)
	}

	_, _, err = qsl.GetSamples([]int{1})
	if err == nil {
		t.Errorf("found a sample that is unloaded.")
	}

	data, label, err := qsl.GetSamples([]int{4, 2, 0})
	if err != nil {
		t.Error(err)
	}

	if len(data) != 3 || len(label) != 3 || label[0] != 817 || label[1] != 13 || label[2] != 817 {
		t.Errorf("incorrect labels.")
	}
}
