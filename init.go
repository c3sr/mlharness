package mlharness

import (
	"github.com/c3sr/config"
	"github.com/c3sr/logger"
	_ "github.com/c3sr/tracer/all"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "mlharness")
	})
	config.Init(
		config.AppName("mlharness"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
}
