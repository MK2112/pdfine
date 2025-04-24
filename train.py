import re
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from pdfine.extractor import raw_extract_pages
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead
from transformers import T5TokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast

# This thing is still a major TODO. We're getting there.

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('PDFine.train')

def compute_reward(world_models, world_tokenizers, text: str, structure_weight=1.0, halluc_weight=1.0, structure_targets=None) -> float:
    """
    Compute reward as negative cross-entropy loss under an ensemble of world models (adversarial critics).
    Add structure-aware and anti-hallucination bonuses/penalties.
    """
    rewards = []
    for wm, wt in zip(world_models, world_tokenizers):
        enc = wt(text, return_tensors='pt', truncation=True)
        inputs = {k: v for k, v in enc.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = wm(**inputs, labels=inputs['input_ids'])
        rewards.append(-outputs.loss.item())
    reward = sum(rewards) / len(rewards)
    # Structure-aware bonus: reward Markdown structure, penalize missing/extra sections
    if structure_targets is not None:
        struct_score = structure_alignment_score(text, structure_targets)
        reward += structure_weight * struct_score
    # Anti-hallucination: penalize hallucinated headings/sections
    halluc_score = -hallucination_score(text)
    reward += halluc_weight * halluc_score
    return reward

def structure_alignment_score(text: str, targets: list) -> float:
    """
    Reward if Markdown headings/sections in text match expected targets (e.g. ['## Page', '### Section']).
    Penalize missing/extra sections.
    """
    found = re.findall(r'^(#+ .+)', text, re.MULTILINE)
    found_set = set(f.split(' ', 1)[-1].strip().lower() for f in found)
    target_set = set(t.lower() for t in targets)
    correct = len(found_set & target_set)
    missing = len(target_set - found_set)
    extra = len(found_set - target_set)
    return (correct - missing - extra) / max(1, len(target_set))

def hallucination_score(text: str) -> float:
    """
    Penalize hallucinated Markdown structure, e.g. repeated headings, non-existent sections, or generic hallucinations.
    """
    lines = text.splitlines()
    heading_counts = {}
    halluc = 0
    for line in lines:
        if re.match(r'^#+ ', line):
            heading = line.strip().lower()
            heading_counts[heading] = heading_counts.get(heading, 0) + 1
            if heading_counts[heading] > 2:
                halluc += 1  # repeated heading
            if 'lorem' in heading or 'foo' in heading:
                halluc += 1  # generic hallucination
    return halluc / max(1, len(lines))

def cross_consistency(candidates):
    if len(candidates) < 2:
        return 1.0
    scores = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            a, b = candidates[i], candidates[j]
            ratio = SequenceMatcher(None, a, b).ratio()
            scores.append(ratio)
    return sum(scores) / len(scores) if scores else 1.0



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_dataset(args.dataset_name, split=args.split)
    seq_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.seq_model).to(device)
    seq_tokenizer = T5TokenizerFast.from_pretrained(args.seq_model)
    world_model_names = [args.world_model] + (args.adv_world_models or [])
    world_models = [GPT2LMHeadModel.from_pretrained(name).to(device) for name in world_model_names]
    world_tokenizers = [GPT2TokenizerFast.from_pretrained(name) for name in world_model_names]
    ppo_config = PPOConfig(
        model_name=args.seq_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs
    )
    ppo_trainer = PPOTrainer(ppo_config, model=seq_model, tokenizer=seq_tokenizer)
    best_reward = float('-inf')
    pdf_difficulty = {}
    # Meta-learning: adaptive reward weights
    structure_weight = args.structure_weight
    halluc_weight = args.halluc_weight
    structure_targets = args.structure_targets
    for epoch in range(1, args.epochs + 1):
        queries, responses, rewards = [], [], []
        reward_log = []
        pdfs = list(ds)
        if args.curriculum:
            pdfs.sort(key=lambda ex: -pdf_difficulty.get(ex.get('file') or ex.get('path'), 0))
        for example in tqdm(pdfs, desc=f'Epoch {epoch}', unit='pdf'):
            try:
                pdf_path = example.get('file') or example.get('path')
                pages = raw_extract_pages(pdf_path)
                if not pages:
                    continue
                text = '\n'.join(p['text'] for p in pages)
                input_ids = seq_tokenizer(text, return_tensors='pt', truncation=True, max_length=args.max_input).input_ids.to(device)
                n_candidates = args.selfplay or 1
                candidates, candidate_rewards = [], []
                for _ in range(n_candidates):
                    response_ids = seq_model.generate(input_ids, max_length=args.max_output)
                    response = seq_tokenizer.decode(response_ids[0], skip_special_tokens=True)
                    reward = compute_reward(
                        world_models, world_tokenizers, response,
                        structure_weight=structure_weight,
                        halluc_weight=halluc_weight,
                        structure_targets=structure_targets
                    )
                    candidates.append(response)
                    candidate_rewards.append(reward)
                consistency = cross_consistency(candidates)
                best_idx = max(range(len(candidates)), key=lambda i: candidate_rewards[i] + args.consistency_weight * consistency)
                best_response = candidates[best_idx]
                final_reward = candidate_rewards[best_idx] + args.consistency_weight * consistency
                queries.append(text)
                responses.append(best_response)
                rewards.append(final_reward)
                reward_log.append(final_reward)
                pdf_difficulty[pdf_path] = abs(consistency - 1.0) + (max(candidate_rewards) - min(candidate_rewards))
                if len(queries) >= args.batch_size:
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                    rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
                    ppo_trainer.step(queries, responses, rewards_norm.tolist())
                    queries, responses, rewards = [], [], []
            except Exception as e:
                logger.warning(f"Failed PDF: {example.get('file') or example.get('path')}, error: {e}")
        if queries:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            rewards_norm = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            ppo_trainer.step(queries, responses, rewards_norm.tolist())
        avg_reward = sum(reward_log) / max(1, len(reward_log))
        logger.info(f"Epoch {epoch} complete. Avg reward: {avg_reward:.4f}")
        # Meta-learning: adapt weights if reward plateaus or variance is high
        if args.meta_learning:
            std = np.std(reward_log)
            if std > 1.0:
                halluc_weight = min(halluc_weight + 0.1, 2.0)
                logger.info(f"Meta-learn: increased halluc_weight to {halluc_weight}")
            if avg_reward < 0.5:
                structure_weight = min(structure_weight + 0.1, 2.0)
                logger.info(f"Meta-learn: increased structure_weight to {structure_weight}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            seq_model.save_pretrained(f"{args.output_dir}/best")
            seq_tokenizer.save_pretrained(f"{args.output_dir}/best")
            logger.info(f"New best model saved: avg reward {avg_reward:.4f}")
    seq_model.save_pretrained(args.output_dir)
    seq_tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train PDF-to-Markdown RL model')
    parser.add_argument('--dataset-name', default='hf-internal-testing/pdf-sample', help='Huggingface dataset with PDF files')
    parser.add_argument('--split', default='train', help='Dataset split')
    parser.add_argument('--seq-model', default='t5-small', help='Seq2seq model for Markdown generation')
    parser.add_argument('--world-model', default='gpt2', help='World model (GPT-2) for rewards')
    parser.add_argument('--adv-world-models', nargs='*', default=[], help='Additional world models for adversarial ensemble')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-input', type=int, default=1024, help='Max input tokens')
    parser.add_argument('--max-output', type=int, default=512, help='Max output tokens')
    parser.add_argument('--output-dir', default='trained_model', help='Directory to save trained model')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for PPO sampling (2 is efficient for 3060 GPUs)')
    parser.add_argument('--ppo-epochs', type=int, default=2, help='PPO epochs per batch (2 is efficient for 3060 GPUs)')
    parser.add_argument('--selfplay', type=int, default=2, help='Number of self-play rollouts per PDF')
    parser.add_argument('--consistency-weight', type=float, default=1.0, help='Weight for cross-consistency in reward shaping')
    parser.add_argument('--structure-weight', type=float, default=1.0, help='Weight for structure-aware reward')
    parser.add_argument('--halluc-weight', type=float, default=1.0, help='Weight for anti-hallucination penalty')
    parser.add_argument('--structure-targets', nargs='*', default=['Page'], help='Expected Markdown section headings for structure reward')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning by resampling hard PDFs')
    parser.add_argument('--meta-learning', action='store_true', help='Enable meta-learning to adapt reward weights')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
