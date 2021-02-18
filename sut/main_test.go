package sut

import (
	"os"
	"testing"

	"github.com/c3sr/config"
)

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("mlcommons-mlmodelscope"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
	os.Exit(m.Run())
}
