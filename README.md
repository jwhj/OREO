# OREO: Offline REasoning Optimization

Source code for [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)

Model: [Policy](https://huggingface.co/jwhj/Qwen2.5-Math-1.5B-OREO) | [Value](https://huggingface.co/jwhj/Qwen2.5-Math-1.5B-OREO-Value)

<img src="./OREO.png" alt="Image description" width="50%" />


# Installation

This repo is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and the installation follows a similar process. We recommend using Docker to setup the environment.

First build Docker image
```bash
cd dockerfile
docker build -t [IMAGE_NAME] .
```

Start a docker container
```bash
docker run -itd --ipc host --gpus all [IMAGE_NAME] bash
```

Attach to the container
```bash
docker exec -it [CONTAINER_ID] /bin/bash
```

Install the current repo
```bash
cd [PATH_TO_THIS_REPO]
pip install -e .
```

As the data collection process involves randomness, we will publish the training data used in our experiments in the near future.

# Reproduction
## Training
You may need to change the following command line options in the following scripts:
- `--train_file` specifies the path of training data in OREO experiments.
- `--dataset` specifies the path of training data in SFT experiments.
- `--save_path` specifies the path to save the model.
- `--pretrain` specifies the path to load the pretrained model. In OREO experiments, this should be the path to the SFT model.

### Math Reasoning

Supervised fine-tuning
```bash
cd example/scripts
bash train_oreo_sft.sh
```

OREO training
```bash
cd example/scripts
bash train_oreo.sh
```

To train the `DeepSeekMath-7B-Instruct` model,
```bash
cd example/scripts
bash train_oreo_deepseek-math.sh
```
Note that `DeepSeekMath-7B-Instruct` is already supervise fine-tuned, so we don't have an SFT phase here.

### ALFWorld

Supervised fine-tuning
```bash
cd example/scripts
bash train_oreo_alfworld_sft.sh
```

OREO training
```bash
cd example/scripts
bash train_oreo_alfworld.sh
```

## Evaluation
### Math Reasoning

Make sure you have `antlr4-python3-runtime==4.11.0` installed.

For Qwen-based models
```bash
cd example/scripts
python ../scratch/run_qwen.py --model [PATH_TO_YOUR_MODEL] --save [SAVE_GENERATED_RESULTS_JSONL]
```

For DeepSeekMath-based models
```bash
cd example/scripts
python ../scratch/run_qwen.py --model [PATH_TO_YOUR_MODEL] --no_bos --save [SAVE_GENERATED_RESULTS_JSONL]
```
Note the `--no_bos` option here.

Here is a script that uses the OREO model to solve a specific math problem:
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "jwhj/Qwen2.5-Math-1.5B-OREO"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model_path)
params = SamplingParams(temperature=0, max_tokens=2048)

message = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {
        "role": "user",
        "content": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    },
]
prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

result = llm.generate(prompt, params)
print(result[0].outputs[0].text)
```
The output should be something like the following:
```
First find the total number of eggs Janet has each day: $16$ eggs/day
Then subtract the number of eggs she eats for breakfast: $16-3=13$ eggs/day
Then subtract the number of eggs she bakes for her friends: $13-4=9$ eggs/day
Then multiply the number of eggs she sells by the price per egg to find her daily earnings: $9\cdot2=\boxed{18}$ dollars/day
```

### ALFWorld

This part requires [ALFWorld](https://github.com/alfworld/alfworld) to be installed.

First start an vllm server
```bash
python -m vllm.entrypoints.openai.api_server --model [PATH_TO_YOUR_MODEL]
```

Then run evaluation with
```bash
cd example/scripts
python ../scratch/run_alfworld_async.py --model [PATH_TO_YOUR_MODEL] --save_dir [SAVE_GENERATED_TRAJS]
```
You can use `--split eval_in_distribution` for seen environments.

## Reference
```bibtex
@article{wang2024offline,
  title={Offline Reinforcement Learning for LLM Multi-Step Reasoning},
  author={Wang, Huaijie and Hao, Shibo and Dong, Hanze and Zhang, Shenao and Bao, Yilin and Yang, Ziran and Wu, Yi},
  journal={arXiv preprint arXiv:2412.16145},
  year={2024}
}
```
