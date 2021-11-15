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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
BACKENDS = ("pytorch", "onnxruntime", "tensorflow", "mxnet")
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['coco', 'imagenet', 'squad', 'brats2019'], required = True, help="select accuracy script for dataset")
    parser.add_argument("--dataset_path", required = True, help="path to dataset yaml file")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlcommons inference benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    # in MLPerf the default max-batchsize value is 128, but in Onnxruntime some models can only support size of 1
    parser.add_argument("--max_batchsize", type=int, default=1, help="max batch size in a single inference")
    parser.add_argument("--backend", choices=BACKENDS, required = True, help="runtime to use")
    parser.add_argument("--model_path", required = True, help="path to model yaml file")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find_peak_performance", action="store_true", help="enable finding peak performance pass")
    parser.add_argument("--model_name", default = "", help="provide model name to match configurations")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="../inference/mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="../inference/vision/classification_and_detection/user.conf", help="user config for user LoadGen settings such as target QPS")
    # log path for loadgen
    parser.add_argument("--log_dir", default='../logs')

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max_latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples_per_query", type=int, help="mlperf multi-stream sample per query")


    # MLHarness Parameters
    parser.add_argument("--use_gpu", type=int, default=0, help="enable gpu for inference")
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU")
    parser.add_argument("--trace_level", choices=TRACE_LEVEL, default="NO_TRACE", help="MLModelScope Trace Level")
    # Modality Specific
    # inv_map for object detection
    parser.add_argument("--use_inv_map", action="store_true", help="use inv_map for object detection")

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

def parse_ret_msg(ret_msg):
    count, err = ret_msg.split(',', 1)
    count = int(count)
    return count, err.lstrip()

def go_initialize(backend, model_path, dataset_path, count, use_gpu, gpu_id, trace_level, max_batchsize):
    global so

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    # Jake Pu: I have no clue where they limit it to 200.
    if count is None:
        count = 0

    # ensure encoding of all strings
    backend = backend.encode('utf-8')
    model_path = model_path.encode('utf-8')
    dataset_path = dataset_path.encode('utf-8')
    trace_level = trace_level.encode('utf-8')

    ret_msg = ctypes.string_at(so.Initialize(c_char_p(backend), c_char_p(model_path), c_char_p(dataset_path), c_int(count),
                                             c_int(use_gpu), c_int(gpu_id), c_char_p(trace_level), c_int(max_batchsize)))
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

def load_go_shared_library():

    so = ctypes.cdll.LoadLibrary('../wrapper/_wrapper.so')

    """
    Go Function Signature
    func Initialize(cBackendName *C.char, cModelPath *C.char, cDatasetPath *C.char, cCount C.int,
    cUseGPU C.int, cGPUID C.int, cTraceLevel *C.char, cBatchSize C.int) *C.char
    """
    so.Initialize.restype = c_char_p
    so.Initialize.argtypes = [c_char_p, c_char_p, c_char_p, c_int,
                                c_int, c_int, c_char_p, c_int]

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

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    if count:
        count_override = True

    """
    Python signature
    go_initialize(backend, model_path, dataset_path, count, use_gpu, gpu_id, trace_level, max_batchsize)
    """

    count, err = go_initialize(backend, args.model_path, args.dataset_path, count,
                    args.use_gpu, args.gpu_id, args.trace_level, args.max_batchsize)


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

    log_dir = None

    if args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
        os.makedirs(log_dir, exist_ok=True)

    scenario = SCENARIO_MAP[args.scenario]

    def issue_queries(query_samples):
        global so
        global last_timeing
        global result_timeing
        idx = np.array([q.index for q in query_samples]).astype(np.int32)
        query_id = [q.id for q in query_samples]
        if args.dataset == 'brats2019':
            start = time.time()
            response_array_refs = []
            response = []
            for i, qid in enumerate(query_id):
                processed_results = so.IssueQuery(1, idx[i][np.newaxis])
                processed_results = json.loads(processed_results.decode('utf-8'))
                response_array = array.array("B", np.array(processed_results[0], np.float16).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(qid, bi[0], bi[1]))
            result_timeing.append(time.time() - start)
            lg.QuerySamplesComplete(response)
        else:
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
    if args.model_name != "":
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
                        'imagenet': os.path.realpath('../inference/vision/classification_and_detection/tools/accuracy-imagenet.py'),
                        'squad': os.path.realpath('../inference/language/bert/accuracy-squad.py'),
                        'brats2019': os.path.realpath('../inference/vision/medical_imaging/3d-unet/accuracy-brats.py'),}
        accuracy_script_path = accuracy_script_paths[args.dataset]
        accuracy_file_path = os.path.join(log_dir, 'mlperf_log_accuracy.json')
        data_dir = os.environ['DATA_DIR']
        if args.dataset == 'coco':
            if args.use_inv_map:
                subprocess.check_call('python3 {} --mlperf-accuracy-file {} --coco-dir {} --use-inv-map'.format(accuracy_script_path, accuracy_file_path, data_dir), shell=True)
            else:
                subprocess.check_call('python3 {} --mlperf-accuracy-file {} --coco-dir {}'.format(accuracy_script_path, accuracy_file_path, data_dir), shell=True)
        elif args.dataset == 'imagenet':   # imagenet
            subprocess.check_call('python3 {} --mlperf-accuracy-file {} --imagenet-val-file {}'.format(accuracy_script_path, accuracy_file_path, os.path.join(data_dir, 'val_map.txt')), shell=True)
        elif args.dataset == 'squad':   # squad
            vocab_path = os.path.join(data_dir, 'vocab.txt')
            val_path = os.path.join(data_dir, 'dev-v1.1.json')
            out_path = os.path.join(log_dir, 'predictions.json')
            cache_path = os.path.join(data_dir, 'eval_features.pickle')
            subprocess.check_call('python3 {} --vocab_file {} --val_data {} --log_file {} --out_file {} --features_cache_file {} --max_examples {}'.
            format(accuracy_script_path, vocab_path, val_path, accuracy_file_path, out_path, cache_path, count), shell=True)
        elif args.dataset == 'brats2019':   # brats2019
            base_dir = os.path.realpath('../inference/vision/medical_imaging/3d-unet/build')
            post_dir = os.path.join(base_dir, 'postprocessed_data')
            label_dir = os.path.join(base_dir, 'raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr')
            os.makedirs(post_dir, exist_ok=True)
            subprocess.check_call('python3 {} --log_file {} --preprocessed_data_dir {} --postprocessed_data_dir {} --label_data_dir {}'.
            format(accuracy_script_path, accuracy_file_path, data_dir, post_dir, label_dir), shell=True)
        else:
            raise RuntimeError('Dataset not Implemented.')

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
