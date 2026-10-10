#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definitions for the category-conditioned multi-task ACSA architecture,
split out of train_mtl_acsa_v2.py so the training script only has to import
the pieces it drives (data loading, loss, optimizer, CLI), not define them.

Architecture
------------
Review -> shared PLM encoder (default: PhoBERT-v2)
       -> category-query cross attention
       -> shared category-specific representation z_c
            |- ACD adapter/head: present vs absent
            |- Sentiment adapter/head: positive / neutral / negative
            |    with soft ACD -> sentiment gating
            `- Joint ACSA adapter/head: NONE / POS / NEU / NEG

arch-v2 (opt-in via entity_attribute_heads=True): 2 auxiliary sentence-level
heads predicting entity/attribute presence (derived from existing category
labels, ENTITY#ATTRIBUTE format), gating the ACD head's input the same way
ACD gates the sentiment head.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class TaskAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(F.gelu(self.down(x))))
        return self.norm(x + self.dropout(delta))


def add_vocab_with_mean_init(encoder: nn.Module, tokenizer, new_tokens: Sequence[str]) -> int:
    """
    Extend `tokenizer`'s vocab with `new_tokens` and resize `encoder`'s input
    embeddings to match, initializing each new token's embedding as the mean
    of the embeddings of the subword pieces it used to tokenize into (instead
    of the framework's default random init) -- a much better warm start given
    only a few thousand fine-tuning examples per domain. Mutates `tokenizer`
    in place, so every other place already holding a reference to it (e.g.
    the DataLoader collate functions) sees the extended vocab automatically.
    Returns the number of tokens actually added (0 if all were already in
    the vocab).
    """
    embedding_matrix = encoder.get_input_embeddings().weight
    piece_means = {}
    for tok_str in new_tokens:
        piece_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if not piece_ids:
            continue
        piece_means[tok_str] = embedding_matrix[piece_ids].mean(dim=0).detach().clone()

    num_added = tokenizer.add_tokens(list(new_tokens))
    if num_added == 0:
        return 0
    encoder.resize_token_embeddings(len(tokenizer))

    new_embedding_matrix = encoder.get_input_embeddings().weight
    with torch.no_grad():
        for tok_str, mean_vec in piece_means.items():
            new_id = tokenizer.convert_tokens_to_ids(tok_str)
            if new_id is not None and new_id != tokenizer.unk_token_id:
                new_embedding_matrix[new_id] = mean_vec
    return num_added


class CategoryConditionedMTL(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        categories: Sequence[str],
        category_texts: Sequence[str],
        num_attention_heads: int = 8,
        adapter_dim: int = 192,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        extra_vocab: Optional[Sequence[str]] = None,
        entity_attribute_heads: bool = False,
        learned_fusion: bool = False,
        fusion_gate: bool = False,
        gate_mode: str = "soft",
        category_query: str = "text",
        conditioning: str = "xattn",
        heads: str = "three",
    ):
        super().__init__()
        self.categories = list(categories)
        self.entity_attribute_heads = entity_attribute_heads
        self.learned_fusion = learned_fusion
        self.fusion_gate = fusion_gate and learned_fusion
        if gate_mode not in ("soft", "hard", "none"):
            raise ValueError(f"gate_mode must be soft|hard|none, got {gate_mode!r}")
        if category_query not in ("text", "id"):
            raise ValueError(f"category_query must be text|id, got {category_query!r}")
        if conditioning not in ("xattn", "attn_only", "concat", "none"):
            raise ValueError(f"conditioning must be xattn|attn_only|concat|none, got {conditioning!r}")
        if heads not in ("three", "joint_only"):
            raise ValueError(f"heads must be three|joint_only, got {heads!r}")
        if heads == "joint_only" and (learned_fusion or entity_attribute_heads):
            raise ValueError("heads='joint_only' cannot be combined with learned_fusion or entity_attribute_heads")
        # Ablation switches (defaults reproduce the main CAGE model):
        #   gate_mode: ACD->sentiment interaction, soft (paper), hard 1[p>=0.5], or none (sentiment reads z)
        #   category_query: text (description/name encoded by the shared PLM) or id (learned free embedding per category)
        #   conditioning: how the category enters z_c --
        #     xattn      cross-attention + residual/LayerNorm + FFN/residual/LayerNorm (paper)
        #     attn_only  z_c = MultiHead(q_c, H, H): the same attention without the residual/FFN block
        #     concat     z_c = MLP([h_<s>; q_c]): category-conditioned, but no attention over the review tokens
        #     none       z_c = h_<s> for every c, no category conditioning; per-category output layers replace the
        #                shared heads (otherwise every category would get the same prediction)
        #   heads: three (ACD + sentiment + joint, paper) or joint_only (one 4-way head; no ACD/sentiment heads, no gate,
        #     no GradNorm). joint_only still returns acd/sent logits, derived from the joint distribution, so the fixed
        #     fusion and metrics code run unchanged: presence = 1 - p(NONE), thresholded at the dev-tuned tau.
        self.gate_mode = gate_mode
        self.category_query = category_query
        self.conditioning = conditioning
        self.heads = heads
        self.encoder = AutoModel.from_pretrained(model_name)
        if extra_vocab:
            num_added = add_vocab_with_mean_init(self.encoder, tokenizer, extra_vocab)
            print(f"Vocab extension: added {num_added}/{len(extra_vocab)} new tokens "
                  f"(rest already in vocab), embeddings init'd from mean of original subword pieces")
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        if hidden % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden} must be divisible by num_attention_heads={num_attention_heads}"
            )

        # Category descriptions are tokenized once; they are re-encoded by the shared
        # PLM on each forward pass so gradients can update the semantic category queries.
        cat_enc = tokenizer(
            list(category_texts),
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )
        self.register_buffer("cat_input_ids", cat_enc["input_ids"], persistent=False)
        self.register_buffer("cat_attention_mask", cat_enc["attention_mask"], persistent=False)

        if category_query == "id":
            self.cat_id_embedding = nn.Embedding(len(self.categories), hidden)
        self.cat_query_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.cross_ffn_norm = nn.LayerNorm(hidden)
        if conditioning == "concat":
            self.concat_mlp = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden, hidden),
                nn.LayerNorm(hidden),
            )

        if heads == "three":
            self.acd_adapter = TaskAdapter(hidden, adapter_dim, dropout)
            self.sent_adapter = TaskAdapter(hidden, adapter_dim, dropout)
        self.joint_adapter = TaskAdapter(hidden, adapter_dim, dropout)

        num_categories = len(self.categories)
        if conditioning == "none":
            # One output layer per category (the review representation is the same for every category).
            bound = hidden ** -0.5
            def per_category(*shape):
                return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))
            if heads == "three":
                self.acd_head_w, self.acd_head_b = per_category(num_categories, hidden), per_category(num_categories)
                self.sent_head_w, self.sent_head_b = per_category(num_categories, 3, hidden), per_category(num_categories, 3)
            self.joint_head_w, self.joint_head_b = per_category(num_categories, 4, hidden), per_category(num_categories, 4)
        else:
            if heads == "three":
                self.acd_head = nn.Linear(hidden, 1)
                self.sent_head = nn.Linear(hidden, 3)
            self.joint_head = nn.Linear(hidden, 4)

        # Learned strength for the soft ACD -> sentiment interaction.
        # sigmoid(0)=0.5 initially.
        self.raw_gate_alpha = nn.Parameter(torch.tensor(0.0))

        # Learned fusion: replaces the fixed 0.5/0.5-weighted fuse_predictions()
        # formula. Diagnostic across every experiment run this project (any
        # technique, any domain) found ACD micro-F1 ~97-98% and sentiment-
        # oracle micro-F1 ~85-91% -- both near ceiling -- yet final ACSA
        # micro-F1 only ~73-76%, a consistent 10-17pt gap regardless of which
        # head-training technique was applied. That gap sits entirely in how
        # the 3 heads' outputs get combined into one answer, not in the heads
        # themselves -- so a small MLP that LEARNS the combination (trained
        # jointly with everything else) is targeted at the actual bottleneck,
        # unlike every other technique tried so far.
        if learned_fusion:
            self.fusion_head = nn.Sequential(
                nn.Linear(1 + 3 + 4, 32), nn.GELU(),
                nn.Linear(32, 32), nn.GELU(),
                nn.Linear(32, 4),
            )
            if self.fusion_gate:
                # Restaurant/Hotel results showed learned_fusion is a real win
                # on dense-data Restaurant but flat-to-slightly-worse on
                # sparse-category Hotel -- a full replacement of the fixed
                # formula trusts fusion_head's own (data-hungry) judgment
                # unconditionally, which sparse categories can't back up. This
                # scalar instead LEARNS how much to trust fusion_head's
                # distribution vs. the data-free fixed 0.5/0.5 formula
                # (reconstructed below as a proper distribution, not just a
                # thresholded scalar), so dense domains can lean on the
                # learned combination while sparse ones fall back toward the
                # formula automatically via gradient descent, no manual
                # per-domain tuning needed. sigmoid(0)=0.5 initially, same
                # pattern as raw_gate_alpha above.
                self.raw_fusion_gate = nn.Parameter(torch.tensor(0.0))

        # arch-v2: entity/attribute auxiliary heads. Categories are
        # "ENTITY#ATTRIBUTE" (e.g. ROOMS#CLEANLINESS) -- these two coarse,
        # sentence-level multi-label heads predict entity presence (ROOMS,
        # FACILITIES, ...) and attribute presence (CLEANLINESS, COMFORT, ...)
        # independent of each other, each pooling supervision signal across
        # every fine category that shares that entity/attribute (far more
        # support per class than any single rare category has alone). Their
        # per-category-gathered probabilities then gate the ACD head's input,
        # the same soft-gate pattern already used for ACD -> sentiment above,
        # applied one level earlier to help disambiguate categories whose
        # text embeddings alone are easy to confuse (e.g. ROOMS vs
        # ROOM_AMENITIES) -- observed as a real, repeated confusion for both
        # this model and an independent LLM extractor, not PhoBERT-specific.
        if entity_attribute_heads:
            entity_to_idx: Dict[str, int] = {}
            attribute_to_idx: Dict[str, int] = {}
            cat_entity_idx_list, cat_attribute_idx_list = [], []
            for cat in self.categories:
                entity, _, attribute = cat.partition("#")
                if entity not in entity_to_idx:
                    entity_to_idx[entity] = len(entity_to_idx)
                if attribute not in attribute_to_idx:
                    attribute_to_idx[attribute] = len(attribute_to_idx)
                cat_entity_idx_list.append(entity_to_idx[entity])
                cat_attribute_idx_list.append(attribute_to_idx[attribute])
            self.entities = list(entity_to_idx.keys())
            self.attributes = list(attribute_to_idx.keys())
            num_entities, num_attributes = len(self.entities), len(self.attributes)

            cat_entity_idx = torch.tensor(cat_entity_idx_list, dtype=torch.long)
            cat_attribute_idx = torch.tensor(cat_attribute_idx_list, dtype=torch.long)
            self.register_buffer("cat_entity_idx", cat_entity_idx, persistent=False)
            self.register_buffer("cat_attribute_idx", cat_attribute_idx, persistent=False)
            # [K, E] / [K, A] one-hot membership -- lets the training loop turn
            # batch["acd_labels"] [B, K] into entity/attribute gold labels via
            # a single matmul + threshold (OR-aggregation across a category's
            # sibling categories), no per-batch Python loop needed.
            self.register_buffer("entity_membership", F.one_hot(cat_entity_idx, num_entities).float(), persistent=False)
            self.register_buffer("attribute_membership", F.one_hot(cat_attribute_idx, num_attributes).float(), persistent=False)

            self.entity_head = nn.Linear(hidden, num_entities)
            self.attribute_head = nn.Linear(hidden, num_attributes)
            self.raw_ea_gate_alpha = nn.Parameter(torch.tensor(0.0))

    def _encode_category_queries(self) -> torch.Tensor:
        if self.category_query == "id":
            return self.cat_query_proj(self.cat_id_embedding.weight)  # [K, D], no category text is used
        cat_out = self.encoder(
            input_ids=self.cat_input_ids,
            attention_mask=self.cat_attention_mask,
            return_dict=True,
        ).last_hidden_state
        # <s>/CLS-style first token representation for each semantic description.
        return self.cat_query_proj(cat_out[:, 0, :])  # [K, D]

    def _head(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """Task output layer: shared across categories, or one per category when conditioning='none'."""
        if self.conditioning != "none":
            out = getattr(self, f"{task}_head")(x)
            return out.squeeze(-1) if task == "acd" else out
        w, b = getattr(self, f"{task}_head_w"), getattr(self, f"{task}_head_b")
        if task == "acd":
            return torch.einsum("bkd,kd->bk", x, w) + b  # [B, K]
        return torch.einsum("bkd,kcd->bkc", x, w) + b  # [B, K, C]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        sent_h = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state  # [B, L, D]

        batch_size = sent_h.size(0)
        num_categories = len(self.categories)
        if self.conditioning == "none":
            # No category conditioning: the pooled review state stands in for every category.
            z = sent_h[:, 0, :].unsqueeze(1).expand(-1, num_categories, -1).contiguous()  # [B, K, D]
        else:
            cat_q = self._encode_category_queries()  # [K, D]
            q = cat_q.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]
            if self.conditioning == "concat":
                pooled = sent_h[:, 0, :].unsqueeze(1).expand(-1, num_categories, -1)  # [B, K, D]
                z = self.concat_mlp(torch.cat([pooled, q], dim=-1))
            else:
                attn_out, _ = self.cross_attention(
                    query=q,
                    key=sent_h,
                    value=sent_h,
                    key_padding_mask=~attention_mask.bool(),
                    need_weights=False,
                )
                if self.conditioning == "attn_only":
                    z = attn_out
                else:
                    z = self.cross_norm(q + attn_out)
                    z = self.cross_ffn_norm(z + self.cross_ffn(z))  # shared branching representation

        if self.heads == "joint_only":
            joint_logits = self._head(self.joint_adapter(z), "joint")  # [B, K, 4]
            # Derived, not trained: log-odds of presence and the polarity logits of the joint head, so the fixed
            # fusion (presence 0.5/0.5 mix, sentiment 0.5/0.5 mix) reduces exactly to the joint head alone.
            acd_logits = torch.logsumexp(joint_logits[..., 1:], dim=-1) - joint_logits[..., 0]
            return {
                "acd_logits": acd_logits,
                "sent_logits": joint_logits[..., 1:],
                "joint_logits": joint_logits,
                "shared_z": z,
                "gate_alpha": torch.sigmoid(self.raw_gate_alpha).detach(),
            }

        entity_logits = attribute_logits = None
        z_for_acd = z
        if self.entity_attribute_heads:
            sent_pooled = sent_h[:, 0, :]  # [B, D] -- <s>/CLS-style pooled sentence repr
            entity_logits = self.entity_head(sent_pooled)  # [B, E]
            attribute_logits = self.attribute_head(sent_pooled)  # [B, A]
            entity_prob = torch.sigmoid(entity_logits)[:, self.cat_entity_idx]  # [B, K]
            attribute_prob = torch.sigmoid(attribute_logits)[:, self.cat_attribute_idx]  # [B, K]
            ea_gate = torch.sigmoid(self.raw_ea_gate_alpha)
            z_for_acd = z * (1.0 + ea_gate * 0.5 * (entity_prob + attribute_prob).unsqueeze(-1))

        acd_z = self.acd_adapter(z_for_acd)
        acd_logits = self._head(acd_z, "acd")  # [B, K]

        # Soft gate: sentiment remains trainable even when ACD is uncertain/wrong.
        acd_prob = torch.sigmoid(acd_logits)
        alpha = torch.sigmoid(self.raw_gate_alpha)
        if self.gate_mode == "soft":
            sent_input = z * (1.0 + alpha * acd_prob.unsqueeze(-1))
        elif self.gate_mode == "hard":
            sent_input = z * (acd_prob >= 0.5).to(z.dtype).unsqueeze(-1)
        else:  # none
            sent_input = z
        sent_z = self.sent_adapter(sent_input)
        sent_logits = self._head(sent_z, "sent")  # [B, K, 3]

        joint_z = self.joint_adapter(z)
        joint_logits = self._head(joint_z, "joint")  # [B, K, 4]

        outputs = {
            "acd_logits": acd_logits,
            "sent_logits": sent_logits,
            "joint_logits": joint_logits,
            "shared_z": z,
            "gate_alpha": alpha,
        }
        if entity_logits is not None:
            outputs["entity_logits"] = entity_logits
            outputs["attribute_logits"] = attribute_logits
        if self.learned_fusion:
            fusion_input = torch.cat([acd_logits.unsqueeze(-1), sent_logits, joint_logits], dim=-1)  # [B, K, 8]
            fused_logits = self.fusion_head(fusion_input)  # [B, K, 4]
            outputs["fused_logits"] = fused_logits
            if self.fusion_gate:
                # Differentiable reconstruction of fuse_predictions()'s fixed
                # 0.5/0.5 formula as a proper 4-class (NONE/POS/NEU/NEG)
                # distribution -- the original returns a thresholded scalar
                # decision, not a distribution, so it can't be blended
                # directly; this is the same math (presence_score,
                # sentiment_score) re-expressed so gradients can flow.
                joint_prob = F.softmax(joint_logits, dim=-1)
                joint_presence = 1.0 - joint_prob[..., 0]
                sent_prob = F.softmax(sent_logits, dim=-1)
                joint_sent = joint_prob[..., 1:]
                joint_sent = joint_sent / joint_sent.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                presence_score = 0.5 * acd_prob + 0.5 * joint_presence
                sentiment_score = 0.5 * sent_prob + 0.5 * joint_sent
                fixed_prob = torch.cat([
                    (1.0 - presence_score).unsqueeze(-1),
                    presence_score.unsqueeze(-1) * sentiment_score,
                ], dim=-1)  # [B, K, 4], sums to 1 along last dim

                fused_prob = F.softmax(fused_logits, dim=-1)
                gate = torch.sigmoid(self.raw_fusion_gate)
                outputs["blended_probs"] = gate * fused_prob + (1.0 - gate) * fixed_prob
                outputs["fusion_gate_value"] = gate.detach()
        return outputs


# -----------------------------------------------------------------------------
# GradNorm
# -----------------------------------------------------------------------------
class GradNormBalancer(nn.Module):
    """
    Dynamic task weighting based on GradNorm.

    Gradient norms are measured at `shared_z`, the representation immediately
    before the three task-specific adapters. This makes the method efficient and
    directly measures conflict/imbalance at the task branching point.
    """

    def __init__(self, num_tasks: int = 3, alpha: float = 1.5):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(num_tasks))
        self.alpha = alpha
        self.register_buffer("initial_losses", torch.zeros(num_tasks))
        self.initialized = False

    def normalized_weights(self) -> torch.Tensor:
        w = F.softplus(self.raw_weights) + 1e-6
        return len(w) * w / w.sum()

    def compute_weight_gradient(
        self,
        losses: Sequence[torch.Tensor],
        shared_representation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_vec = torch.stack(list(losses))
        if not self.initialized:
            self.initial_losses.copy_(loss_vec.detach().clamp_min(1e-8))
            self.initialized = True

        w = self.normalized_weights()
        grad_norms = []
        for i, loss in enumerate(losses):
            grad = torch.autograd.grad(
                w[i] * loss,
                shared_representation,
                retain_graph=True,
                create_graph=True,
            )[0]
            grad_norms.append(torch.norm(grad, p=2))
        grad_norms = torch.stack(grad_norms)

        with torch.no_grad():
            loss_ratio = loss_vec.detach() / self.initial_losses.clamp_min(1e-8)
            inverse_train_rate = loss_ratio / loss_ratio.mean()
            target = grad_norms.detach().mean() * (inverse_train_rate ** self.alpha)

        gradnorm_objective = torch.abs(grad_norms - target).sum()
        weight_grad = torch.autograd.grad(
            gradnorm_objective,
            self.raw_weights,
            retain_graph=True,
            create_graph=False,
        )[0]
        return w, weight_grad, gradnorm_objective.detach()
