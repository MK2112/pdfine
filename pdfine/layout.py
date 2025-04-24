import torch
from torch import nn
from .utils import logger
from typing import List, Dict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

"""
ML-based layout interpretation refinement using RL and world model
"""

# Load world model (a pretrained GPT-2) for reward estimation
def load_world_model(model_name: str = "gpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

class PolicyNet(nn.Module):
    """Policy network to propose segment boundaries."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # boundary probability

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.rnn(x)
        logits = self.fc(out).squeeze(-1)
        return torch.sigmoid(logits)  # probabilities per token

# Caching for inference
_world_model = None
_tokenizer = None
_policy = None

def init_training(world_model_name="gpt2", lr=1e-4):
    global _world_model, _tokenizer, _policy, _optimizer
    if _world_model is None:
        _world_model, _tokenizer = load_world_model(world_model_name)
    vocab_size = _tokenizer.vocab_size
    _policy = PolicyNet(vocab_size)
    _optimizer = torch.optim.Adam(_policy.parameters(), lr=lr)

# Training loop: self-adversarial RL without labeled data
def train_agent(pdf_texts: List[str], epochs: int = 5, batch_size: int = 1):
    """
    Train policy network to place segment boundaries maximizing world-model reward.
    pdf_texts: list of full page texts
    """
    init_training()
    for epoch in range(epochs):
        total_reward = 0
        for text in pdf_texts:
            # Tokenize
            enc = _tokenizer(text, return_tensors='pt', truncation=True)
            input_ids = enc.input_ids
            # Get segmentation probabilities
            probs = _policy(input_ids)
            # Sample boundaries (Bernoulli)
            m = torch.distributions.Bernoulli(probs)
            sample = m.sample()
            logp = m.log_prob(sample).sum()
            # Create segments by splitting at boundary tokens
            segments = []
            seg = []
            for tok_id, b in zip(input_ids[0].tolist(), sample[0].tolist()):
                seg.append(tok_id)
                if b > 0.5:
                    segments.append(seg)
                    seg = []
            if seg:
                segments.append(seg)
            # Compute world-model log-likelihood reward
            reward = 0
            for seg_ids in segments:
                with torch.no_grad():
                    inputs = {'input_ids': torch.tensor([seg_ids])}
                    outputs = _world_model(**inputs, labels=torch.tensor([seg_ids]))
                    reward += -outputs.loss.item()  # negative loss as reward
            total_reward += reward
            # Policy gradient: maximize reward
            loss = -logp * reward
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Avg Reward: {total_reward/len(pdf_texts):.4f}")

# Refinement API for inference
def refine_pages(pdf_path: str, raw_pages: List[Dict]) -> List[Dict]:
    """
    Use trained policy to segment raw_pages into text_blocks. If no policy, return raw_pages as-is.
    Always returns [{'page_num', 'text_blocks'}].
    """
    global _policy, _tokenizer
    if _policy is None or _tokenizer is None:
        logger.info("No trained policy loaded; returning raw pages.")
        # Convert to expected format if missing text_blocks
        result = []
        for page in raw_pages:
            blocks = [page['text']] if 'text_blocks' not in page else page['text_blocks']
            result.append({'page_num': page['page_num'], 'text_blocks': blocks})
        return result
    refined = []
    for page in raw_pages:
        text = page['text']
        enc = _tokenizer(text, return_tensors='pt', truncation=True)
        probs = _policy(enc.input_ids)
        lines = []
        seg = []
        for tok_id, p in zip(enc.input_ids[0], probs[0]):
            seg.append(tok_id.item())
            if p.item() > 0.5:
                lines.append(_tokenizer.decode(seg))
                seg = []
        if seg:
            lines.append(_tokenizer.decode(seg))
        refined.append({'page_num': page['page_num'], 'text_blocks': lines})
    return refined
