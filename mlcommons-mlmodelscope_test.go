package mlcommonsmlmomodelscope

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/c3sr/tracer/jaeger"
)

func TestInfoModels(t *testing.T) {
	// if err := InfoModels("onnxruntime"); err != nil {
	// 	t.Error(err)
	// }

	// if err := InfoModels("mxnet"); err != nil {
	// 	t.Error(err)
	// }

	// if err := InfoModels("tensorflow"); err != nil {
	// 	t.Error(err)
	// }

	// if err := InfoModels("unimplemented"); err == nil {
	// 	t.Errorf("backend not implemented but found.")
	// }
}

func TestImageNet(t *testing.T) {
	pwd, _ := os.Getwd()
	os.Setenv("DATA_DIR", filepath.Join(pwd, "qsl/dataset/_fixtures/fake_imagenet"))
	if _, err := Initialize("onnxruntime", "MLPerf_ResNet50_v1.5", "1.0",
		"imagenet", "", 10, false, "FULL_TRACE", 3); err != nil {
		t.Error(err)
	}

	if err := LoadQuerySamples([]int{0, 1, 2, 3, 4, 5, 6, 7}); err != nil {
		t.Error(err)
	}

	if err := UnloadQuerySamples([]int{1, 3, 5, 7}); err != nil {
		t.Error(err)
	}

	res := IssueQuery([]int{0, 2, 4, 6})
	fmt.Println(res)

	if err := UnloadQuerySamples([]int{}); err != nil {
		t.Error(err)
	}

	if err := Finalize(); err != nil {
		t.Error(err)
	}
}

func TestCoco(t *testing.T) {
	pwd, _ := os.Getwd()
	os.Setenv("DATA_DIR", filepath.Join(pwd, "qsl/dataset/_fixtures/fake_coco"))
	if _, err := Initialize("onnxruntime", "onnxvision_ssd", "1.0",
		"coco", "", 0, false, "FULL_TRACE", 1); err != nil {
		t.Error(err)
	}

	if err := LoadQuerySamples([]int{0, 1, 2, 3, 4, 5, 6, 7}); err != nil {
		t.Error(err)
	}

	if err := UnloadQuerySamples([]int{1, 3, 5, 7}); err != nil {
		t.Error(err)
	}

	res := IssueQuery([]int{0, 2, 4, 6})
	fmt.Println(res)

	if err := UnloadQuerySamples([]int{}); err != nil {
		t.Error(err)
	}

	if err := Finalize(); err != nil {
		t.Error(err)
	}
}
