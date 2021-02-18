package dataset

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/dlframework/steps"
	"github.com/c3sr/pipeline"
)

type ImageNet struct {
	data   []interface{}
	labels []int
}

var (
	defaultChannelBuffer = 100000
)

func NewImageNet(dataPath string, imageList string, count int, preprocessOptions common.PreprocessOptions) (Dataset, error) {

	start := time.Now()

	res := &ImageNet{}

	if imageList == "" {
		imageList = filepath.Join(dataPath, "val_map.txt")
	}
	file, err := os.Open(imageList)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	notFound := 0

  input := make(chan interface{}, defaultChannelBuffer)
  opts := []pipeline.Option{pipeline.ChannelBuffer(defaultChannelBuffer)}
  output := pipeline.New(opts...).
    Then(steps.NewReadImage(preprocessOptions)).
    Then(steps.NewPreprocessImage(preprocessOptions)).
    Run(input)

	for scanner.Scan() {
		val := strings.Split(scanner.Text(), " ")
		if len(val) != 2 {
			return nil, fmt.Errorf("The format of the image list is not correct.")
		}

		imageName, label := val[0], val[1]

		src := filepath.Join(dataPath, imageName)
		if _, err := os.Stat(src); err != nil {
			notFound++
			continue
		}

		// preprocess image
		imageBytes, err := ioutil.ReadFile(src)
		if err != nil {
			return nil, fmt.Errorf("Cannot read %s", src)
		}

		input <- bytes.NewBuffer(imageBytes)

		labelInt, err := strconv.Atoi(label)
		if err != nil {
			return nil, fmt.Errorf("Can't convert %s into an interger.", label)
		}

		res.labels = append(res.labels, labelInt)

		if count != 0 && len(res.labels) == count {
			break
		}
	}

  close(input)
  for out := range output {
    res.data = append(res.data, out)
  }

	elapsed := time.Now().Sub(start)

	if len(res.data) == 0 {
		return nil, fmt.Errorf("no images in image list found.")
	}
	if notFound > 0 {
		fmt.Printf("reduced image list, %d images not found", notFound)
	}

	fmt.Printf("loaded %d images, took %.1f seconds.\n", len(res.data), elapsed.Seconds())

	return res, nil

}
