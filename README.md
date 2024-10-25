# ArkVale: Efficient Gener<ins>a</ins>tive LLM Inference with <ins>R</ins>ecallable <ins>K</ins>ey-<ins>Val</ins>ue <ins>E</ins>viction 

## Download

```bash
git clone https://github.com/pku-liang/ArkVale.git --recursive 
```

or 

```bash
git clone https://github.com/pku-liang/ArkVale.git
cd ArkVale
git submodule update --init --recursive --depth 1 
```

## Install 

```bash
pip install -r requirements.txt
cd source && python3 setup.py [develop|install]
```

## Usage 

```python
from transformers import AutoModelForCausalLM
from arkvale import adapter
path = ...
dev = torch.device("cuda:0")
dtype = torch.float16
model = (
    AutoModelForCausalLM
    .from_pretrained(path, torch_dtype=dtype, device_map=dev)
    .eval()
)
adapter.enable_arkvale(
    model, 
    dtype=dtype, 
    device=dev, 
    page_size=32,
    # page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
    page_budgets=4096 // 32,
    page_topks=32,
    n_max_bytes=40 * (1 << 30),
    n_max_cpu_bytes=80 * (1 << 30),
)
...
```
