#!/usr/bin/env python3
""" Model Benchmark Script

An inference and train step benchmark script for timm models.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import csv
import json
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import hydra
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.parallel
from hydra import compose, initialize
from omegaconf import OmegaConf
from timm.data import resolve_data_config
from timm.layers import set_fast_norm
from timm.models import create_model, is_model, list_models
from timm.optim import create_optimizer_v2
from timm.utils import set_jit_fuser, decay_batch_step, check_batch_size_retry

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from deepspeed.profiling.flops_profiler import get_model_profile

    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    has_fvcore_profiling = False

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


def timestamp(sync=False):
    return time.perf_counter()


def cuda_timestamp(sync=False, device=None):
    if sync:
        torch.cuda.synchronize(device=device)
    return time.perf_counter()


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def resolve_precision(precision: str):
    assert precision in ('amp', 'amp_bfloat16', 'float16', 'bfloat16', 'float32')
    amp_dtype = None  # amp disabled
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16':
        amp_dtype = torch.bfloat16
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return amp_dtype, model_dtype, data_dtype


def profile_deepspeed(model, input_size=(3, 224, 224), batch_size=1, detailed=False):
    _, macs, _ = get_model_profile(
        model=model,
        input_shape=(batch_size,) + input_size,  # input shape/resolution
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), batch_size=1, detailed=False, force_cpu=False):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


class BenchmarkRunner:
    def __init__(
            self,
            model_name,
            detail=False,
            device='cuda',
            torchscript=False,
            torchcompile=None,
            aot_autograd=False,
            precision='float32',
            fuser='',
            num_warm_iter=10,
            num_bench_iter=50,
            use_train_size=False,
            **kwargs
    ):
        self.detail = detail
        self.device = device
        self.amp_dtype, self.model_dtype, self.data_dtype = resolve_precision(precision)
        self.channels_last = kwargs.pop('channels_last', False)
        if self.amp_dtype is not None:
            self.amp_autocast = partial(torch.cuda.amp.autocast, dtype=self.amp_dtype)
        else:
            self.amp_autocast = suppress

        if fuser:
            set_jit_fuser(fuser)
        if isinstance(model_name, torch.nn.Module):
            self.model = model_name
            self.model_name = model_name.__class__.__name__
        else:
            self.model = create_model(
                model_name,
                num_classes=kwargs.pop('num_classes', None),
                in_chans=3,
                global_pool=kwargs.pop('gp', 'fast'),
                scriptable=torchscript,
                drop_rate=kwargs.pop('drop', 0.),
                drop_path_rate=kwargs.pop('drop_path', None),
                drop_block_rate=kwargs.pop('drop_block', None),
                **kwargs.pop('model_kwargs', {}),
            )
        self.model.to(device=self.device, dtype=self.model_dtype,
                      memory_format=torch.channels_last if self.channels_last else None)
        self.num_classes = self.model.num_classes
        self.param_count = count_params(self.model)
        # _logger.info('Model %s created, param count: %d' % (model_name, self.param_count))

        data_config = resolve_data_config(kwargs, model=self.model, use_test_size=not use_train_size)
        self.input_size = data_config['input_size']
        self.batch_size = kwargs.pop('batch_size', 256)

        self.compiled = False
        if torchscript:
            self.model = torch.jit.script(self.model)
            self.compiled = True
        elif torchcompile:
            assert has_compile, 'A version of torch w/ torch.compile() is required, possibly a nightly.'
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend=torchcompile)
            self.compiled = True
        elif aot_autograd:
            assert has_functorch, "functorch is needed for --aot-autograd"
            self.model = memory_efficient_fusion(self.model)
            self.compiled = True

        self.example_inputs = None
        self.num_warm_iter = num_warm_iter
        self.num_bench_iter = num_bench_iter
        self.log_freq = num_bench_iter // 5
        if 'cuda' in self.device:
            self.time_fn = partial(cuda_timestamp, device=self.device)
        else:
            self.time_fn = timestamp

    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, device=self.device, dtype=self.data_dtype)
        if self.channels_last:
            self.example_inputs = self.example_inputs.contiguous(memory_format=torch.channels_last)


class InferenceBenchmarkRunner(BenchmarkRunner):

    def __init__(
            self,
            model_name,
            device='cuda',
            torchscript=False,
            **kwargs
    ):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.eval()

    def run(self):
        def _step():
            t_step_start = self.time_fn()
            with self.amp_autocast():
                output = self.model(self.example_inputs)
            t_step_end = self.time_fn(True)
            return t_step_end - t_step_start

        _logger.info(
            f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        with torch.no_grad():
            self._init_input()

            for _ in range(self.num_warm_iter):
                _step()

            total_step = 0.
            num_samples = 0
            t_run_start = self.time_fn()
            for i in range(self.num_bench_iter):
                delta_fwd = _step()
                total_step += delta_fwd
                num_samples += self.batch_size
                num_steps = i + 1
                if num_steps % self.log_freq == 0:
                    _logger.info(
                        f"Infer [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start

        results = dict(
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / self.num_bench_iter, 3),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        retries = 0 if self.compiled else 2  # skip profiling if model is scripted
        while retries:
            retries -= 1
            try:
                if has_deepspeed_profiling:
                    macs, _ = profile_deepspeed(self.model, self.input_size)
                    results['gmacs'] = round(macs / 1e9, 2)
                elif has_fvcore_profiling:
                    macs, activations = profile_fvcore(self.model, self.input_size, force_cpu=not retries)
                    results['gmacs'] = round(macs / 1e9, 2)
                    results['macts'] = round(activations / 1e6, 2)
            except RuntimeError as e:
                pass

        _logger.info(
            f"Inference benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step")

        return results


class TrainBenchmarkRunner(BenchmarkRunner):

    def __init__(
            self,
            model_name,
            device='cuda',
            torchscript=False,
            **kwargs
    ):
        super().__init__(model_name=model_name, device=device, torchscript=torchscript, **kwargs)
        self.model.train()

        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()

        self.optimizer = create_optimizer_v2(
            self.model,
            opt=kwargs.pop('opt', 'sgd'),
            lr=kwargs.pop('lr', 1e-4))

        if kwargs.pop('grad_checkpointing', False):
            self.model.set_grad_checkpointing()

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.target_shape, device=self.device, dtype=torch.long).random_(self.num_classes)

    def run(self):
        def _step(detail=False):
            self.optimizer.zero_grad()  # can this be ignored?
            t_start = self.time_fn()
            t_fwd_end = t_start
            t_bwd_end = t_start
            with self.amp_autocast():
                output = self.model(self.example_inputs)
                if isinstance(output, tuple):
                    output = output[0]
                if detail:
                    t_fwd_end = self.time_fn(True)
                target = self._gen_target(output.shape[0])
                self.loss(output, target).backward()
                if detail:
                    t_bwd_end = self.time_fn(True)
            self.optimizer.step()
            t_end = self.time_fn(True)
            if detail:
                delta_fwd = t_fwd_end - t_start
                delta_bwd = t_bwd_end - t_fwd_end
                delta_opt = t_end - t_bwd_end
                return delta_fwd, delta_bwd, delta_opt
            else:
                delta_step = t_end - t_start
                return delta_step

        _logger.info(
            f'Running train benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        self._init_input()

        for _ in range(self.num_warm_iter):
            _step()

        t_run_start = self.time_fn()
        if self.detail:
            total_fwd = 0.
            total_bwd = 0.
            total_opt = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_fwd, delta_bwd, delta_opt = _step(True)
                num_samples += self.batch_size
                total_fwd += delta_fwd
                total_bwd += delta_bwd
                total_opt += delta_opt
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    total_step = total_fwd + total_bwd + total_opt
                    _logger.info(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_fwd / num_steps:0.3f} ms/step fwd,"
                        f" {1000 * total_bwd / num_steps:0.3f} ms/step bwd,"
                        f" {1000 * total_opt / num_steps:0.3f} ms/step opt."
                    )
            total_step = total_fwd + total_bwd + total_opt
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                fwd_time=round(1000 * total_fwd / self.num_bench_iter, 3),
                bwd_time=round(1000 * total_bwd / self.num_bench_iter, 3),
                opt_time=round(1000 * total_opt / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )
        else:
            total_step = 0.
            num_samples = 0
            for i in range(self.num_bench_iter):
                delta_step = _step(False)
                num_samples += self.batch_size
                total_step += delta_step
                num_steps = (i + 1)
                if num_steps % self.log_freq == 0:
                    _logger.info(
                        f"Train [{num_steps}/{self.num_bench_iter}]."
                        f" {num_samples / total_step:0.2f} samples/sec."
                        f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_elapsed = self.time_fn() - t_run_start
            results = dict(
                samples_per_sec=round(num_samples / t_run_elapsed, 2),
                step_time=round(1000 * total_step / self.num_bench_iter, 3),
                batch_size=self.batch_size,
                img_size=self.input_size[-1],
                param_count=round(self.param_count / 1e6, 2),
            )

        _logger.info(
            f"Train benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/sample")

        return results


class ProfileRunner(BenchmarkRunner):

    def __init__(self, model_name, device='cuda', profiler='', **kwargs):
        super().__init__(model_name=model_name, device=device, **kwargs)
        if not profiler:
            if has_deepspeed_profiling:
                profiler = 'deepspeed'
            elif has_fvcore_profiling:
                profiler = 'fvcore'
        assert profiler, "One of deepspeed or fvcore needs to be installed for profiling to work."
        self.profiler = profiler
        self.model.eval()

    def run(self):
        _logger.info(
            f'Running profiler on {self.model_name} w/ '
            f'input size {self.input_size} and batch size {self.batch_size}.')

        macs = 0
        activations = 0
        if self.profiler == 'deepspeed':
            macs, _ = profile_deepspeed(self.model, self.input_size, batch_size=self.batch_size, detailed=False)
        elif self.profiler == 'fvcore':
            macs, activations = profile_fvcore(self.model, self.input_size, batch_size=self.batch_size, detailed=False)

        results = dict(
            gmacs=round(macs / 1e9, 2),
            macts=round(activations / 1e6, 2),
            batch_size=self.batch_size,
            img_size=self.input_size[-1],
            param_count=round(self.param_count / 1e6, 2),
        )

        _logger.info(
            f"Profile of {self.model_name} done. "
            f"{results['gmacs']:.2f} GMACs, {results['param_count']:.2f} M params.")

        return results


def _try_run(
        model_name,
        bench_fn,
        bench_kwargs,
        initial_batch_size,
        no_batch_size_retry=False
):
    batch_size = initial_batch_size
    results = dict()
    error_str = 'Unknown'
    while batch_size:
        try:
            torch.cuda.empty_cache()
            bench = bench_fn(model_name=model_name, batch_size=batch_size, **bench_kwargs)
            results = bench.run()
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running benchmark.')
            if not check_batch_size_retry(error_str):
                _logger.error(f'Unrecoverable error encountered while benchmarking {model_name}, skipping.')
                break
            if no_batch_size_retry:
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    return results


def benchmark(args, nn_model=None):
    if args.amp:
        # _logger.warning("Overriding precision to 'amp' since --amp flag set.")
        args.precision = 'amp' if args.amp_dtype == 'float16' else '_'.join(['amp', args.amp_dtype])
    _logger.info(f'Benchmarking in {args.precision} precision. '
                 f'{"NHWC" if args.channels_last else "NCHW"} layout. '
                 f'torchscript {"enabled" if args.torchscript else "disabled"}')

    bench_kwargs = OmegaConf.to_container(args)
    bench_kwargs.pop('amp')
    model = bench_kwargs.pop('model')
    batch_size = bench_kwargs.pop('batch_size')

    bench_fns = (InferenceBenchmarkRunner,)
    prefixes = ('infer',)
    if args.bench == 'all':
        bench_fns = (
            InferenceBenchmarkRunner,
            TrainBenchmarkRunner,
            ProfileRunner,
        )
        prefixes = ('infer', 'train', 'profile')
    elif args.bench == 'both':
        bench_fns = (
            InferenceBenchmarkRunner,
            TrainBenchmarkRunner
        )
        prefixes = ('infer', 'train')
    elif args.bench == 'inf+pro':
        bench_fns = (
            InferenceBenchmarkRunner,
            ProfileRunner
        )
        prefixes = ('infer', 'profile')
    elif args.bench == 'train':
        bench_fns = TrainBenchmarkRunner,
        prefixes = 'train',
    elif args.bench.startswith('profile'):
        # specific profiler used if included in bench mode string, otherwise default to deepspeed, fallback to fvcore
        if 'deepspeed' in args.bench:
            assert has_deepspeed_profiling, "deepspeed must be installed to use deepspeed flop counter"
            bench_kwargs['profiler'] = 'deepspeed'
        elif 'fvcore' in args.bench:
            assert has_fvcore_profiling, "fvcore must be installed to use fvcore flop counter"
            bench_kwargs['profiler'] = 'fvcore'
        bench_fns = ProfileRunner,
        batch_size = 1

    model_results = OrderedDict(model=model)
    for prefix, bench_fn in zip(prefixes, bench_fns):
        run_results = _try_run(
            nn_model if nn_model else model,
            bench_fn,
            bench_kwargs=bench_kwargs,
            initial_batch_size=batch_size,
            no_batch_size_retry=args.no_retry,
        )
        if prefix and 'error' not in run_results:
            run_results = {'_'.join([prefix, k]): v for k, v in run_results.items()}
        model_results.update(run_results)
        if 'error' in run_results:
            break
    if 'error' not in model_results:
        param_count = model_results.pop('infer_param_count', model_results.pop('train_param_count', 0))
        model_results.setdefault('param_count', param_count)
        model_results.pop('train_param_count', 0)
    return model_results


@hydra.main(config_path="configs/benchmark", config_name="benchmark", version_base="1.3")
def main():
    model_cfgs = []
    model_names = []

    if cfg.fast_norm:
        set_fast_norm()

    if cfg.model_list:
        cfg.model = ''
        with open(cfg.model_list) as f:
            model_names = [line.rstrip() for line in f]
        model_cfgs = [(n, None) for n in model_names]

    elif not is_model(cfg.model):
        # model name doesn't exist, try as wildcard filter
        model_names = list_models(cfg.model)
        model_cfgs = [(n, None) for n in model_names]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            for m, _ in model_cfgs:
                if not m:
                    continue
                cfg.model = m
                r = benchmark(cfg)
                if r:
                    results.append(r)
                time.sleep(10)
        except KeyboardInterrupt as e:
            pass
        sort_key = 'infer_samples_per_sec'
        if 'train' in cfg.bench:
            sort_key = 'train_samples_per_sec'
        elif 'profile' in cfg.bench:
            sort_key = 'infer_gmacs'
        results = filter(lambda x: sort_key in x, results)
        results = sorted(results, key=lambda x: x[sort_key], reverse=True)
    else:
        results = benchmark(cfg)

    if cfg.results_file:
        write_results(cfg.results_file, results)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def benchmark_with_model(cfg, model):
    if isinstance(cfg.dataset.size, int):
        cfg.benchmark.input_size = (cfg.dataset.in_channels, cfg.dataset.size, cfg.dataset.size)
    else:
        cfg.benchmark.input_size = cfg.dataset.size

    cfg.benchmark.num_classes = cfg.dataset.num_classes
    result = benchmark(cfg.benchmark, model)

    result.pop('model', None)
    result.pop('infer_img_size', None)

    print(f'Benchmark result\n{json.dumps(result, indent=4)}')
    return result


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


if __name__ == '__main__':
    with initialize('../../configs'):
        cfg = compose('config')

    model = create_model('resnet50')
    results = benchmark_with_model(cfg, model)
    print(results)
