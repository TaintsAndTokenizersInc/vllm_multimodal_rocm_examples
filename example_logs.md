### vLLM API Server Startup Log (Example)

```bash
vllm-1  | INFO 09-25 13:38:11 [__init__.py:216] Automatically detected platform rocm.
vllm-1  | (APIServer pid=8) INFO 09-25 13:38:14 [api_server.py:1822] vLLM API server version 0.11.0rc2.dev98+g393de22d2
vllm-1  | (APIServer pid=8) INFO 09-25 13:38:14 [utils.py:233] non-default args: {'model_tag': '/google/gemma-3-27b-it', 'model': '/google/gemma-3-27b-it', 'max_model_len': 4096, 'limit_mm_per_prompt': {'image': 2}}
vllm-1  | (APIServer pid=8) `torch_dtype` is deprecated! Use `dtype` instead!
vllm-1  | (APIServer pid=8) INFO 09-25 13:38:20 [model.py:545] Resolved architecture: Gemma3ForConditionalGeneration
vllm-1  | (APIServer pid=8) INFO 09-25 13:38:20 [model.py:1564] Using max model len 4096
vllm-1  | (APIServer pid=8) INFO 09-25 13:38:20 [scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=8192.
vllm-1  | INFO 09-25 13:38:25 [__init__.py:216] Automatically detected platform rocm.
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:28 [core.py:644] Waiting for init message from front-end.
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:28 [core.py:77] Initializing a V1 LLM engine (v0.11.0rc2.dev98+g393de22d2) with config: model='/google/gemma-3-27b-it', speculative_config=None, tokenizer='/google/gemma-3-27b-it', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/google/gemma-3-27b-it, enable_prefix_caching=True, chunked_prefill_enabled=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer","vllm.gdn_attention"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":[2,1],"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"use_inductor_graph_partition":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
vllm-1  | [W925 13:38:30.973452569 ProcessGroupNCCL.cpp:981] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:31 [parallel_state.py:1201] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
vllm-1  | (EngineCore_DP0 pid=278) Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:36 [gpu_model_runner.py:2547] Starting to load model /google/gemma-3-27b-it...
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:36 [gpu_model_runner.py:2579] Loading model from scratch...
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:36 [layer.py:424] MultiHeadAttention attn_backend: _Backend.TORCH_SDPA, use_upstream_fa: False
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:38:37 [rocm.py:251] Using Triton Attention backend on V1 engine.
Loading safetensors checkpoint shards:   0% Completed | 0/12 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   8% Completed | 1/12 [00:02<00:26,  2.42s/it]
Loading safetensors checkpoint shards:  17% Completed | 2/12 [00:05<00:25,  2.57s/it]
Loading safetensors checkpoint shards:  25% Completed | 3/12 [00:07<00:23,  2.61s/it]
Loading safetensors checkpoint shards:  33% Completed | 4/12 [00:10<00:21,  2.64s/it]
Loading safetensors checkpoint shards:  42% Completed | 5/12 [00:13<00:18,  2.65s/it]
Loading safetensors checkpoint shards:  50% Completed | 6/12 [00:15<00:16,  2.73s/it]
Loading safetensors checkpoint shards:  58% Completed | 7/12 [00:18<00:13,  2.71s/it]
Loading safetensors checkpoint shards:  67% Completed | 8/12 [00:21<00:10,  2.70s/it]
Loading safetensors checkpoint shards:  75% Completed | 9/12 [00:24<00:08,  2.70s/it]
Loading safetensors checkpoint shards:  83% Completed | 10/12 [00:26<00:05,  2.64s/it]
Loading safetensors checkpoint shards:  92% Completed | 11/12 [00:27<00:01,  1.98s/it]
Loading safetensors checkpoint shards: 100% Completed | 12/12 [00:29<00:00,  2.01s/it]
Loading safetensors checkpoint shards: 100% Completed | 12/12 [00:29<00:00,  2.43s/it]
vllm-1  | (EngineCore_DP0 pid=278) 
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:06 [default_loader.py:267] Loading weights took 29.38 seconds
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:07 [gpu_model_runner.py:2598] Model loading took 51.7891 GiB and 29.929735 seconds
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:07 [gpu_model_runner.py:3269] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 31 image items of the maximum feature size.
vllm-1  | (EngineCore_DP0 pid=278) WARNING 09-25 13:39:34 [cudagraph_dispatcher.py:106] cudagraph dispatching keys are not initialized. No cudagraph will be used.
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:42 [backends.py:548] Using cache directory: /root/.cache/vllm/torch_compile_cache/f789f0b4ce/rank_0_0/backbone for vLLM's torch.compile
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:42 [backends.py:559] Dynamo bytecode transform time: 8.49 s
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:39:48 [backends.py:197] Cache the graph for dynamic shape for later use
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:22 [backends.py:218] Compiling a graph for dynamic shape takes 39.70 s
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:25 [monitor.py:34] torch.compile takes 48.19 s in total
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:27 [gpu_worker.py:306] Available KV cache memory: 109.86 GiB
vllm-1  | (EngineCore_DP0 pid=278) WARNING 09-25 13:40:28 [kv_cache_utils.py:982] Add 8 padding layers, may waste at most 15.38% KV cache memory
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:28 [kv_cache_utils.py:1087] GPU KV cache size: 205,696 tokens
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:28 [kv_cache_utils.py:1091] Maximum concurrency for 4,096 tokens per request: 50.05x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 67/67 [00:07<00:00,  9.40it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████| 67/67 [00:13<00:00,  5.05it/s]
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:49 [gpu_model_runner.py:3388] Graph capturing finished in 22 secs, took 1.02 GiB
vllm-1  | (EngineCore_DP0 pid=278) INFO 09-25 13:40:50 [core.py:210] init engine (profile, create kv cache, warmup model) took 102.52 seconds
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [loggers.py:148] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 89994
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [api_server.py:1618] Supported_tasks: ['generate']
vllm-1  | (APIServer pid=8) WARNING 09-25 13:40:51 [model.py:1443] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [serving_responses.py:137] Using default chat sampling params from model: {'top_k': 64, 'top_p': 0.95}
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [serving_chat.py:137] Using default chat sampling params from model: {'top_k': 64, 'top_p': 0.95}
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [serving_completion.py:76] Using default completion sampling params from model: {'top_k': 64, 'top_p': 0.95}
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [api_server.py:1895] Starting vLLM API server 0 on http://0.0.0.0:8000
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:34] Available routes are:
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /openapi.json, Methods: HEAD, GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /docs, Methods: HEAD, GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /docs/oauth2-redirect, Methods: HEAD, GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /redoc, Methods: HEAD, GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /health, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /load, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /ping, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /ping, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /tokenize, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /detokenize, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/models, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /version, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/responses, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/responses/{response_id}, Methods: GET
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/responses/{response_id}/cancel, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/chat/completions, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/completions, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/embeddings, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /pooling, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /classify, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /score, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/score, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/audio/transcriptions, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/audio/translations, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /rerank, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v1/rerank, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /v2/rerank, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /scale_elastic_ep, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /is_scaling_elastic_ep, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /invocations, Methods: POST
vllm-1  | (APIServer pid=8) INFO 09-25 13:40:51 [launcher.py:42] Route: /metrics, Methods: GET
vllm-1  | (APIServer pid=8) INFO:     Started server process [8]
vllm-1  | (APIServer pid=8) INFO:     Waiting for application startup.
vllm-1  | (APIServer pid=8) INFO:     Application startup complete.
vllm-1  | (APIServer pid=8) INFO:     172.19.0.1:43934 - "GET /v1/models HTTP/1.1" 200 OK
```