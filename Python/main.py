"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from queue import Queue
import subprocess
import mlperf_loadgen as lg
import numpy as np

import ctypes
from ctypes import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

last_timeing = []
result_timeing = []
last_loaded = -1

TRACE_LEVEL = ( "NO_TRACE",
                "APPLICATION_TRACE",
                "MODEL_TRACE",          # pipelines within model
                "FRAMEWORK_TRACE",      # layers within framework
                "ML_LIBRARY_TRACE",     # cudnn, ...
                "SYSTEM_LIBRARY_TRACE", # cupti
                "HARDWARE_TRACE",       # perf, papi, ...
                "FULL_TRACE")           # includes all of the above)
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['coco', 'imagenet'], help="dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    # in MLPerf the default max-batchsize value is 128, but in Onnxruntime lots of model can only support size of 1
    parser.add_argument("--max-batchsize", type=int, default=1, help="max batch size in a single inference")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--model-name", help="name of the mlperf model, ie. resnet50")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="../inference/mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="../inference/vision/classification_and_detection/user.conf", help="user config for user LoadGen settings such as target QPS")
    # log path for loadgen
    parser.add_argument("--log_dir", default='../logs')

    # unused arguments from MLPerf
    # parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    # parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    # parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    # parser.add_argument("--model", required=True, help="model file")
    # parser.add_argument("--inputs", help="model inputs")
    # parser.add_argument("--outputs", help="model outputs")
    # parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    # parser.add_argument("--cache", type=int, default=0, help="use cache")
    # parser.add_argument("--output", help="test results")


    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", type=int, help="mlperf multi-stream sample per query")


    # MLModelScope Parameters
    parser.add_argument("--use-gpu", type=int, default=0, help="enable gpu for inference")
    parser.add_argument("--trace-level", choices=TRACE_LEVEL, default="NO_TRACE", help="MLModelScope Trace Level")
    parser.add_argument("--model-version", help="version of the model used in MLModelScope")
    parser.add_argument("--info-models", action="store_true", help="list the models under the specified backend")

    args = parser.parse_args()

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))

    return args

def get_backend(backend):
    if backend == "tensorflow":
        return backend
    elif backend == "onnxruntime":
        return backend
    elif backend == "pytorch":
        return backend
    elif backend == "mxnet":
        return backend
    else:
        raise ValueError("unknown backend: " + backend)

def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100. * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))

def parse_ret_msg(ret_msg):
    count, err = ret_msg.split(',', 1)
    count = int(count)
    return count, err.lstrip()


def initialize_sut(dataset, dataset_list, backend, model_name, model_version, count, use_gpu, trace_level, max_batchsize):
    # (dataset, backend, use_gpu, max_batchsize) won't be None, checked by main()
    global so
    if dataset_list is None:
        dataset_list = ""
    if model_name is None:
        model_name = ""
    if model_version is None:
        model_version = ""

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    # Jake Pu: I have no clue where they limit it to 200.
    if count is None:
        count = 0

    # ensure encoding of all strings
    backend = backend.encode('utf-8')
    model_name = model_name.encode('utf-8')
    model_version = str(model_version).encode('utf-8')
    dataset = dataset.encode('utf-8')
    dataset_list = dataset_list.encode('utf-8')
    trace_level = trace_level.encode('utf-8')

    ret_msg = ctypes.string_at(so.Initialize(c_char_p(backend), c_char_p(model_name), c_char_p(model_version),
                                                c_char_p(dataset), c_char_p(dataset_list),
                                                c_int(count), c_int(use_gpu), c_char_p(trace_level), c_int(max_batchsize)))
    count, err = parse_ret_msg(ret_msg.decode('utf-8'))
    return count, err

def go_load_query_samples(sample_list, so):
    sample_list = np.array(sample_list).astype(np.int32)
    ret_msg = so.LoadQuerySamples(len(sample_list), sample_list)
    return ret_msg.decode('utf-8')

def go_unload_query_samples(sample_list, so):
    if sample_list is None:
        sample_list = []
    sample_list = np.array(sample_list).astype(np.int32)
    ret_msg = so.UnloadQuerySamples(len(sample_list), sample_list)
    return ret_msg.decode('utf-8')

def go_finalize(so):
  ret_msg = so.Finalize()
  return ret_msg.decode('utf-8')

def go_info_models(backend, so):
  backend = backend.encode('utf-8')
  ret_msg = so.InfoModels(backend)
  return ret_msg.decode('utf-8')

def load_go_shared_library():

    so = ctypes.cdll.LoadLibrary('../wrapper/_wrapper.so')

    """
    Go Function Signature
    func Initialize(cBackendName *C.char, cModelName *C.char, cModelVersion *C.char,
	  cDatasetName *C.char, cImageList *C.char, cCount C.int, cUseGPU C.int, cTraceLevel *C.char, cMaxBatchsize C.int) *C.char
    """
    so.Initialize.restype = c_char_p
    so.Initialize.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p,
                                c_char_p, c_int, c_int, c_char_p, c_int]

    """
    Have to use numpy ndarray to pass the integer list
    https://nesi.github.io/perf-training/python-scatter/ctypes#learn-the-basics
    https://numpy.org/doc/stable/reference/routines.ctypeslib.html#module-numpy.ctypeslib
    https://numpy.org/devdocs/user/basics.types.html
    """
    so.IssueQuery.restype = c_char_p
    so.IssueQuery.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.int32)]

    """
    Go Function Signature
    func Finalize() *C.char
    """
    so.Finalize.restype = c_char_p
    so.Finalize.argtypes = []

    """
    Go Function Signature
    func InfoModels(cBackendName *C.char) *C.char
    """
    so.InfoModels.restype = c_char_p
    so.InfoModels.argtypes = [c_char_p]

    """
    Go Function Signature
    func LoadQuerySamples(cLen C.int, cSampleList *C.int) *C.char
    """
    so.LoadQuerySamples.restype = c_char_p
    so.LoadQuerySamples.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.int32)]

    """
    Go Function Signature
    func UnloadQuerySamples(cLen C.int, cSampleList *C.int) *C.char
    """
    so.UnloadQuerySamples.restype = c_char_p
    so.UnloadQuerySamples.argtypes = [c_int, np.ctypeslib.ndpointer(dtype=np.int32)]

    return so

def main():

    global so
    global last_timeing
    global last_loaded
    global result_timeing

    args = get_args()

    log.info(args)

    # find backend
    backend = get_backend(args.backend)

    if args.info_models:
      err = go_info_models(backend, so)
      if (err != ''):
        print(err)
      return

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    if count:
        count_override = True

    final_results = {
        "runtime": args.model_name,
        "version": args.model_version,
        "time": int(time.time()),
        "cmdline": str(args),
    }

    """
    Python signature
    go_initialize(dataset, dataset_list, backend, model_name, model_version, count, use_gpu, trace_level, max_batchsize, so)
    """

    # initialize_sut('imagenet', '', 'pytorch', 'torchvision_alexnet', '1.0', 0, 0, 'FULL_TRACE')
    count, err = initialize_sut(args.dataset, args.dataset_list, backend, args.model_name, 
                    args.model_version, args.count, args.use_gpu, args.trace_level, args.max_batchsize)


    if (err != 'nil'):
        print(err)
        raise RuntimeError('initialization in go failed')

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    # if args.output:
    #     output_dir = os.path.abspath(args.output)
    #     os.makedirs(output_dir, exist_ok=True)
    #     os.chdir(output_dir)

    scenario = SCENARIO_MAP[args.scenario]

    def issue_queries(query_samples):
        global so
        global last_timeing
        global result_timeing
        idx = np.array([q.index for q in query_samples]).astype(np.int32)
        query_id = [q.id for q in query_samples]
        start = time.time()
        processed_results = so.IssueQuery(len(idx), idx)
        result_timeing.append(time.time() - start)
        processed_results = json.loads(processed_results.decode('utf-8'))
        response_array_refs = []
        response = []
        for idx, qid in enumerate(query_id):
            response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)

    def flush_queries():
        pass

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    def load_query_samples(sample_list):
        global so
        global last_loaded
        err = go_load_query_samples(sample_list, so)
        last_loaded = time.time()
        if (err != ''):
            print(err)
            raise RuntimeError('load query samples failed')

    def unload_query_samples(sample_list):
        global so
        err = go_unload_query_samples(sample_list, so)
        if (err != ''):
            print(err)
            raise RuntimeError('unload query samples failed')

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(count, min(count, 500), load_query_samples, unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"good": 0, "total": 0, "scenario": str(scenario)}

    log_path = os.path.realpath(args.log_dir)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    # log_settings.enable_trace = True
    # lg.StartTest(sut, qsl, settings)
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    
    if not last_timeing:
        last_timeing = result_timeing


    if args.accuracy:
        accuracy_script_paths = {'coco': os.path.realpath('../inference/vision/classification_and_detection/tools/accuracy-coco.py'),
                        'imagenet': os.path.realpath('../inference/vision/classification_and_detection/tools/accuracy-imagenet.py')}
        accuracy_script_path = accuracy_script_paths[args.dataset]
        if args.dataset == 'coco':
            subprocess.check_call(['python3', accuracy_script_path, '--mlperf-accuracy-file', 'mlperf_log_accuracy.json',
                                    '--coco-dir', '$DATA_DIR', '--verbose'], shell=True)
        else:   # imagenet
            subprocess.check_call(['python3', accuracy_script_path, '--mlperf-accuracy-file', 'mlperf_log_accuracy.json',
                                    '--imagenet-val-file', '$DATA_DIR/val_map.txt'], shell=True)
    # runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    """
    Python signature
    go_finalize(so)
    """
    err = go_finalize(so)
    if (err != ''):
        print(err)
        raise RuntimeError('finialize in go failed')




# load MLModelScope go wrapper shared libraby
so = load_go_shared_library()

if __name__ == "__main__":
    main()
