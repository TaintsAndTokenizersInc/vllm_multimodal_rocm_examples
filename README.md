# vLLM Multimodal Inference Demo

A complete example for running multimodal inference with **Gemma 3 27B IT** using **vLLM** on ROCm-enabled hardware. This repository demonstrates how to serve a multimodal model with support for text and images, and interact with it via the OpenAI-compatible API.

---

## 🚀 Features

- Serve **Gemma 3 27B IT** with vLLM
- Support for **text**,  **image URLs** and **base64-encoded content**
- Health checks and robust error handling
- Configurable via `.env` and CLI arguments
- Compatible with OpenAI client library

---

## 📦 Prerequisites

- **ROCm-enabled Instinct GPU** (e.g., AMD MI300X, MI325X)
- **Docker** and ROCm-based vLLM
- **Docker Compose**
- **Python 3.10+**
- `pip` and `virtualenv` (optional)
- `huggingface-cli` installed (`pip install huggingface-cli`)

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/TaintsAndTokenizersInc/vllm-multimodal-demo.git
cd vllm-multimodal-demo
```

### 2. Download the Model

> ⚠️ Use `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads with multi-threading.

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/gemma-3-27b-it --local-dir ~/google/gemma-3-27b-it
```

> This will download the model to `~/google/gemma-3-27b-it`. Ensure the directory exists and is accessible.

### 3. Build and Run the vLLM Server

```bash
docker compose up -d
```

> 🔍 **Check server logs** for startup status and any errors.  
> View example output in [`example_logs.md`](example_logs.md).

> ⚠️ Ensure your system has ROCm drivers installed and the `docker` daemon is running.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Run Examples

### Run a Specific Example

```bash
python multimodal_vllm_example.py --chat-type single-image
```

Available options:

- `text-only`
- `single-image`
- `multi-image`

---

## 📁 File Structure

```
.
├── compose.yaml               # Docker setup for vLLM with ROCm 
├── multimodal_vllm_example.py # Demo script with OpenAI client
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
└── README.md                  # This README file
```

---

## ⚙️ Configuration

### `.env` File

```env
OPENAI_MODEL="/google/gemma-3-27b-it"
OPENAI_API_KEY="EMPTY"
OPENAI_BASE_URL="http://localhost:8000/v1"
```

> `OPENAI_API_KEY="EMPTY"` is required for vLLM compatibility (no real key needed).

---

## 📌 Notes

- **Gemma 3 27B IT** is unstable in `float16` → uses `bfloat16` for stability.
- `limit-mm-per-prompt` restricts to 2 images per prompt, if removed vLLM defaults to a single image per API call
- Ensure the model path in `compose.yaml` matches your local directory (`~/google/gemma-3-27b-it`).

---

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Gemma 3 Model Card](https://deepmind.google/models/gemma/)
- [Hugging Face Model Hub](https://huggingface.co/google/gemma-3-27b-it)
- [OpenAI API Specification (vLLM-compatible)](https://platform.openai.com/docs/api-reference)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
