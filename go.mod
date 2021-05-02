module github.com/c3sr/mlcommons-mlmodelscope

go 1.15

replace (
	github.com/coreos/bbolt => go.etcd.io/bbolt v1.3.5
	github.com/jaegertracing/jaeger => github.com/uber/jaeger v1.22.0
	github.com/uber/jaeger => github.com/jaegertracing/jaeger v1.22.0
	google.golang.org/grpc => google.golang.org/grpc v1.29.1
)

require (
	github.com/c3sr/config v1.0.1
	github.com/c3sr/dlframework v1.2.2
	github.com/c3sr/go-python3 v0.0.0-20210424014611-ae173b2e6908
	github.com/c3sr/logger v1.0.1
	github.com/c3sr/mxnet v1.0.0
	github.com/c3sr/nvidia-smi v1.0.0
	github.com/c3sr/onnxruntime v1.0.2-0.20210426041342-2cebad7eb8bb
	github.com/c3sr/pipeline v1.0.0
	github.com/c3sr/pytorch v1.0.0
	github.com/c3sr/tensorflow v1.0.1
	github.com/c3sr/tracer v1.0.0
	github.com/olekukonko/tablewriter v0.0.5
	github.com/opentracing/opentracing-go v1.2.0
	github.com/pkg/errors v0.9.1
	github.com/sirupsen/logrus v1.8.1
)
