# LongReward: Improving Long-context Large Language Models with AI Feedback

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongReward-10k" target="_blank">HuggingFace</a> • 📃 <a href="https://arxiv.org/abs/2410.21252" target="_blank">Paper</a>
</p>

## 🔍 Table of Contents

- [🤖️ LongReward](#longreward)
- [⚙️ Datasets & Models](#released-datasets--models)
- [🌟 Quick Start](#quickstart)
- [📊 Evaluation](#evaluation)
- [📝 Citation](#citation)
- [📍 License](#license)

## LongReward

![cof](https://github.com/user-attachments/assets/a9b06ba1-23ca-44b4-be98-dc2b59b5b84c)

We open-source **LongReward** under `long_reward/auto_scorer.py`, a novel method that utilize an off-the-shelf LLM to
automatically provide rewards for model responses in long-context scenarios, considering four human-valued dimensions:
helpfulness, logicality, faithfulness, and completeness. Given a long-context-based model response, LongReward assigns a
score ranging from 0 to 10 for each dimension, and takes their average as the final reward.

Please configure your API key in the `utils/llm_api.py` before running the code. You can also run the following scripts
under `batch_inference/` for large-scale inference: `1_get_chunk_info.py`, `2_get_score.py`. `3_get_pairs.py`. The
results will be stored in `./data`.

<a name="model"></a>

## Released Datasets & Models

### SFT Datasets & SFT Models

Our [SFT dataset](https://huggingface.co/datasets/THUDM/LongReward-10k) contains 10k long-context QA instances, whose
context lengths range from 8k to 64k tokens. The QA pairs are generated
by [GLM-4-0520](https://bigmodel.cn/dev/api/normal-model/glm-4), following the self-instruct method
in [LongAlign](https://github.com/THUDM/LongAlign).
Using this dataset, we supervised fine-tune two
models: [LongReward-glm4-9b-SFT](https://huggingface.co/NeoZ123/LongReward-glm4-9b-SFT)
and [LongReward-llama3.1-8b-SFT](https://huggingface.co/NeoZ123/LongReward-llama3.1-8b-SFT), which are based
on [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b)
and [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), respectively, and support up to 64k
context.

### Preference Datasets & DPO Models

We utilize LongReward and prompts in the SFT dataset to construct
the [preference datasets](https://huggingface.co/datasets/THUDM/LongReward-10k) for each SFT model, and train their DPO
version: [LongReward-glm4-9b-DPO](https://huggingface.co/THUDM/LongReward-glm4-9b-DPO)
and [LongReward-llama3.1-8b-DPO](https://huggingface.co/THUDM/LongReward-llama3.1-8b-DPO). More Details can be found in
our paper.

### All Available Datasets and Models

Here is the full list of datasets and models we released:

| Name                                | Download Path                                                                                                                                                                                                                               |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LongReward-10k (SFT & DPO Datasets) | [🤗 HuggingFace](https://huggingface.co/datasets/THUDM/LongReward-10k)                                                                                                                                                                      |
| LongReward-glm4-9b-SFT              | [🤗 HuggingFace](https://huggingface.co/NeoZ123/LongReward-glm4-9b-SFT)                                                                                                                                                                     |
| LongReward-glm4-9b-DPO              | [🤗 HuggingFace](https://huggingface.co/THUDM/LongReward-glm4-9b-DPO), [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/LongReward-glm4-9b-DPO), [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/LongReward-glm4-9b-dpo)             |
| LongReward-llama3.1-8b-SFT          | [🤗 HuggingFace](https://huggingface.co/NeoZ123/LongReward-llama3.1-8b-SFT)                                                                                                                                                                 |
| LongReward-llama3.1-8b-DPO          | [🤗 HuggingFace](https://huggingface.co/THUDM/LongReward-llama3.1-8b-DPO), [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/LongReward-llama3.1-8b-dpo), [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/LongReward-llama3.1-8b-dpo) |

## QuickStart

Try our model with following step:

1. install requirements

```shell
pip install -r requirement.txt
```

2. run with model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'THUDM/LongReward-glm4-9b-DPO'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

message = [
    {
        "role": "user",
        "content": "W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938). \n\n What was Robert Geddes' profession?"
    }
]

inputs = tokenizer.apply_chat_template(
    message,
    return_tensors='pt',
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

input_len = inputs['input_ids'].shape[1]
generate_kwargs = {
    "input_ids": inputs['input_ids'],
    "attention_mask": inputs['attention_mask'],
    "max_new_tokens": 128,
    "do_sample": False,
}
out = model.generate(**generate_kwargs)
print(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
```

## Evaluation

We provide our evaluation code for [LongBench](https://github.com/THUDM/LongBench)
and [LongBench-Chat](https://github.com/THUDM/LongAlign) under `evaluation/`. Details can be found in
`evaluation/README.md` and `evaluation/LongBench_Chat/README.md`. Remember to configure your OpenAI API key in
`utils/llm_api.py` since we adopt GPT-4o as the judge.

To reproduce our results on other benchmarks, we refer to the code in [FastChat](https://github.com/lm-sys/FastChat)
and [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) for evaluating on MT-Bench and AlpacaEval2.

Here are our evaluation results:

![eval](https://github.com/user-attachments/assets/c8fc4503-42a1-4081-95b7-7d560f2ec366)


## Citation

If you find our work helpful, please consider citing the following paper:

```
@article{zhang2024longreward,
  title = {LongReward: Improving Long-context Large Language Models
with AI Feedback} 
  author={Jiajie Zhang and Zhongni Hou and Xin Lv and Shulin Cao and Zhenyu Hou and Yilin Niu and Lei Hou and Yuxiao Dong and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv:2410.21252},
  year={2024}
}
```

## License

+ The use of GLM-4 model weights must follow
  the [Model License](https://huggingface.co/THUDM/glm-4-9b/blob/main/LICENSE).

+ The code in this open source repository follows the [Apache 2.0](LICENSE) license.

Please strictly follow the open source license.

