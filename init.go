package mlcommonsmlmomodelscope

import (
	"github.com/c3sr/config"
	"github.com/c3sr/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "mlcommonsmlmomodelscope")
	})
	config.Init(
		config.AppName("mlcommons-mlmodelscope"),
		config.DebugMode(true),
		config.VerboseMode(true),
	)
}
