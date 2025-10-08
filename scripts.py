from copy import deepcopy

from vllm import LLM, SamplingParams
from typing import Callable, List
import json
import os
from math_verify import parse

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_transformers(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str], dict[str, float]], 
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truth_list: List[str],
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    # output: prompt, prompt_token_ids, outputs[0].text, outputs[0].token_ids
    for output, ground_truth in list(zip(outputs, ground_truth_list)):
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, ground_truth)
        result = {
            "prompt": output.prompt,
            "answer": ground_truth,
            "generation": generated_text,
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
            "reward": reward["reward"],
        }
        results.append(result)

    # 序列化，存储
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f)


def load_data(prompt_path, data_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        r1_zero_prompt = f.read()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                question = example["question"]
                answer = example["answer"]
                prompt = r1_zero_prompt.format(question=question)

                yield prompt, answer


def math_baseline():
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    data_path = "data/gsm8k/test.jsonl"

    # sample
    # 此处测试使用全部数据
    prompts = []
    ground_truth_list = []
    for prompt, answer in load_data(prompt_path, data_path):
        prompts.append(prompt)
        ground_truth_list.append(parse(answer)[1])

    # generate
    llm = LLM(
        model="/home/junyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        gpu_memory_utilization=0.6,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 模型输出的停止标记为 "</answer>"
        include_stop_str_in_output=True,  # 模型输出中包含停止标记
    )

    evaluate_transformers(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truth_list=ground_truth_list,
    )


    # 测试读取
    with open("results/evaluation.json", "r", encoding="utf-8") as f:
        results = json.load(f)
        type1 = 0
        type2 = 0
        type3 = 0
        total = 0
        type2_list = []
        type3_list = []
        for result in results:
            total += 1
            if result["format_reward"]==1 and result["answer_reward"]==1:
                type1 += 1
            elif result["format_reward"]==1 and result["answer_reward"]==0:
                type2 += 1
                if len(type2_list) < 10:
                    type2_list.append({"Answer": result["answer"], "Generation": result["generation"]})
            else:
                type3 += 1
                if len(type3_list) < 10:
                    type3_list.append({"Answer": result["answer"], "Generation": result["generation"]})

    print(f"total: {total}")
    print(f"type1: {type1}")
    print(f"type2: {type2}")
    print(f"type3: {type3}")

    print(f"\nExamples of type2:")
    for i, example in enumerate(type2_list):
        print(f"Example {i}: {example}")
    print(f"\nExamples of type3:")
    for i, example in enumerate(type3_list):
        print(f"Example {i}: {example}")

    return results






#%% SFT 训练
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from math import ceil
from torch.optim import AdamW

from tests.adapters import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def format_data(prompt_path, data_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        r1_zero_prompt = f.read()

    save_path = data_path.replace("train.jsonl", "sft.jsonl")

    with open(save_path, "w", encoding="utf-8") as save_file:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    question = example["question"]
                    answer = example["answer"]

                    prompt = r1_zero_prompt.format(question=question)
                    response = answer.replace("\n#### ", "</think><answer>")
                    response = response + "</answer>"

                    json.dump({"prompt": prompt, "response": response}, save_file)
                    save_file.write("\n")


def load_format_data(data_path, batch_size):
    prompt_list = []
    response_list = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                prompt = example["prompt"]
                response = example["response"]

                prompt_list.append(prompt)
                response_list.append(response)
                if len(prompt_list) == batch_size:
                    yield prompt_list, response_list
                    prompt_list = []
                    response_list = []

    if prompt_list:
        yield prompt_list, response_list


def sft_experiment(batch_size, gradient_accumulation_steps, normalize_constant, epochs, lr, milestone):
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    train_path = "data/gsm8k/train.jsonl"
    test_path = "data/gsm8k/test.jsonl"

    # 1. 格式化训练数据
    save_path = train_path.replace("train.jsonl", "sft.jsonl")
    if not os.path.isfile(save_path):
        format_data(prompt_path, train_path)  # 7473 条训练数据

    num_batch = ceil(7473 / batch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "/home/junyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,  # 使用半精度，减少显存消耗
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    optimizer = AdamW(model.parameters(), lr=lr)

    llm = LLM(
        model="/home/junyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        gpu_memory_utilization=0.6,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 模型输出的停止标记为 "</answer>"
        include_stop_str_in_output=True,  # 模型输出中包含停止标记
    )

    print(f"Using device: {device}")


    for epoch in range(epochs):
        pbar = tqdm(
                load_format_data(save_path, batch_size),
                total=num_batch,
                unit="batch",
                desc=f"SFT Training Epoch {epoch}",
        )
        optimizer.zero_grad()

        torch.cuda.empty_cache()  # 清理缓存

        for i, (prompt_list, response_list) in enumerate(pbar):
            # 2. 加载训练数据，进行拼接
            tokens = tokenize_prompt_and_output(prompt_list, response_list, tokenizer)

            input_ids = tokens["input_ids"].to(device)
            labels = tokens["labels"].to(device)
            response_mask = tokens["response_mask"].to(device)

            # 3. 获取对数概率
            result = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )

            # 4. 计算损失
            loss, log = sft_microbatch_train_step(
                policy_log_probs=result["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=normalize_constant,
            )

            pbar.set_postfix(loss=loss.item()*gradient_accumulation_steps, log=log)

            # 5. 参数更新
            if (i+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 6. 评估
        if epoch % milestone == 0:
            prompts = []
            ground_truth_list = []
            for prompt, answer in load_data(prompt_path, test_path):
                prompts.append(prompt)
                ground_truth_list.append(parse(answer)[1])

            log_generations(
                vllm_model=llm,
                tokenizer=tokenizer,
                reward_fn=r1_zero_reward_fn,
                prompts=prompts,
                eval_sampling_params=sampling_params,
                ground_truth_list=ground_truth_list,
            )

    # 7. 保存
    model.save_pretrained(save_directory="results/model")
    tokenizer.save_pretrained(save_directory="results/tokenizer")


#%% GRPO 训练脚本（结合 GPT 修改）
from tests.adapters import grpo_microbatch_train_step, compute_group_normalized_rewards


import os
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

def grpo_train_loop(
        n_grpo_steps=200,
        learning_rate=1e-5,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        advantage_eps=1e-6,
        rollout_batch_size=256,
        group_size=8,
        sampling_temperature=1.0,
        sampling_min_tokens=4,
        sampling_max_tokens=1024,
        epochs_per_rollout_batch=3,
        train_batch_size=256,
        gradient_accumulation_steps=128,
        gpu_memory_utilization=0.85,
        loss_type="grpo_clip",
        use_std_normalization=True,
        milestone=5,  # 每隔 milestone 步更新一次 vLLM
):
    # -----------------------------
    # 1. Initialization
    # -----------------------------
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    train_path = "data/gsm8k/train.jsonl"
    test_path = "data/gsm8k/test.jsonl"

    save_path = train_path.replace("train.jsonl", "grpo.jsonl")
    if not os.path.isfile(save_path):
        format_data(prompt_path, train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "/home/junyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )

    # 初始化 vLLM 旧策略
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    sampler = load_format_data(save_path, rollout_batch_size)
    old_model = deepcopy(model).eval()

    # -----------------------------
    # 2. Training Loop
    # -----------------------------
    for step in range(n_grpo_steps):
        # --- 采样批次 ---
        try:
            prompt_list, response_list = next(sampler)
        except StopIteration:
            sampler = load_format_data(save_path, rollout_batch_size)
            prompt_list, response_list = next(sampler)

        # --- 每步同步 transformer 的旧策略 ---
        old_model = deepcopy(model).eval()

        # --- 每隔 milestone 步更新一次 vLLM ---
        if step % milestone == 0:
            print(f"[Step {step}] Updating vLLM old model ...")
            llm = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)
            # 为了节省显存，不使用 deepcopy；重新加载 checkpoint 即可
            # 如果希望更精确，也可以保存 transformer 的权重然后 reload

        # --- 构造 group 扩展 ---
        prompt_list_ = []
        response_list_ = []
        for prompt, response in zip(prompt_list, response_list):
            prompt_list_.extend([prompt] * group_size)
            response_list_.extend([response] * group_size)

        # --- rollout ---
        outputs = llm.generate(prompt_list_, sampling_params)
        output_strs = [output.outputs[0].text for output in outputs]

        # --- compute rewards & advantages ---
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=output_strs,
            repeated_ground_truths=response_list_,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # --- compute log probs ---
        tokens = tokenize_prompt_and_output(prompt_list_, output_strs, tokenizer)
        input_ids = tokens["input_ids"].to(device)
        labels = tokens["labels"].to(device)
        response_mask = tokens["response_mask"].to(device)

        with torch.no_grad():
            old_log_probs = get_response_log_probs(
                model=old_model,  # ✅ 用 deepcopy 同步的 transformer
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False
            )["log_probs"]

        # --- policy update ---
        model.train()
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

            policy_log_probs = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False
            )["log_probs"]

            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type=loss_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=0.2,
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- 定期评估 ---
        if step % milestone == 0:
            prompts, ground_truth_list = [], []
            for prompt, answer in load_data(prompt_path, test_path):
                prompts.append(prompt)
                ground_truth_list.append(parse(answer)[1])

            log_generations(
                vllm_model=llm,
                tokenizer=tokenizer,
                reward_fn=r1_zero_reward_fn,
                prompts=prompts,
                eval_sampling_params=sampling_params,
                ground_truth_list=ground_truth_list,
            )

    # -----------------------------
    # 3. Save Model
    # -----------------------------
    model.save_pretrained("results/model")
    tokenizer.save_pretrained("results/tokenizer")




# if __name__ == "__main__":
    # sft_experiment(
    #     batch_size=1,
    #     gradient_accumulation_steps=1,
    #     normalize_constant=1.0,
    #     epochs=1,
    #     lr=0.01,
    #     milestone=10,
    # )