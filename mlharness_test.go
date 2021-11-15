package mlharness

import (
	"fmt"
	"testing"

	_ "github.com/c3sr/tracer/all"
)

func TestImageNet(t *testing.T) {
	if _, err := Initialize("onnxruntime", "/Users/user/MLModelScope/c3sr/dlmodel/models/vision/onnxruntime/torchvision/TorchVision_AlexNet.yml", "/Users/user/MLModelScope/c3sr/dldataset/datasets/image_url.yml",
		2, false, 0, "FULL_TRACE", 1); err != nil {
		t.Error(err)
	}

	if err := LoadQuerySamples([]int{0, 1}); err != nil {
		t.Error(err)
	}

	res := IssueQuery([]int{0, 1})
	fmt.Println(res)

	if err := UnloadQuerySamples([]int{}); err != nil {
		t.Error(err)
	}

	if err := Finalize(); err != nil {
		t.Error(err)
	}
}
