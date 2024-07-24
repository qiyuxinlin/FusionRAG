<div align="center">
  <h1>KTransformers</h1>
  <h3>A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations</h3>
  <strong><a href="#show-cases">🔥 Show Cases</a> | <a href="#quick-start">🚀 Quick Start</a></strong>
</div>


<h2 id="overview">🎉 Overview</h2>
KTransformers, or Quick Transformers, is designed to enhance your 🤗 <a href="https://github.com/huggingface/transformers">Transformers</a> experience with advanced kernel optimizations and placement/parallelism strategies. With KTransformers, you can run state-of-the-art models like the 236B DeepSeek-V2 on a single 24GB consumer GPU and 136GB DRAM by adding just one line of code. Experience up to 6X speedup on prefill and 3.4X speedup for generation compared to llama.cpp. More features, including support for extremely long contexts, are coming soon.
<br/><br/>
KTransformers is a flexible, Python-centric framework designed to inject cutting-edge LLM inference optimizations into production-ready, user-friendly inference engines. Implement an optimized operator/kernel, inject it via KTransformers, and gain a Transformers-compatible interface, OpenAI/ollama-compatible API, and even a simple ChatGPT-like web UI.
<br/><br/>
We aim to develop KTransformers as a user-friendly platform for experiencing and experimenting with cutting-edge LLM inference optimizations. Please let us know if you need any other features.


<h2 id="show-cases">🔥 Show Cases</h2>
<h3>GPT-4-level Local VSCode Copilot on a Desktop with only 24GB VRAM</h3>

- **Local 236B DeepSeek-Coder-V2:** Running its Q4_K_M version using only 21GB VRAM and 136GB DRAM, attainable on a local desktop machine.
- **Faster Speed:** Achieving 126 tokens/s for 2K prompt prefill and 13.6 tokens/s for generation through MoE offloading and injecting advanced kernels from [Llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main) and [Marlin](https://github.com/IST-DASLab/marlin).
- **Long Context:** Further improved to XXX tokens/s for long 120K prompt via layer-wise GPU prefill, suitable for long context understanding.
- **VSCode Integration:** Wrapped into an OpenAI and Ollama compatible API for seamless integration as a backend for Tabby and various other frontends.

<strong>More advanced features will coming soon, so stay tuned!</strong>

<h2 id="quick-start">🚀 Quick Start</h2>
