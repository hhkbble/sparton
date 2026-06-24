import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from model import SpladeModel

@dataclass
class LSRModelArguments:
    model_name_or_path: str = field(
        default="FacebookAI/xlm-roberta-base",
        metadata={"help": "HuggingFace model identifier or local path"},
    )
    head: str = field(
        default="torch",
        metadata={"help": "'torch', 'compiled', or 'sparton' for SpartonHead kernel"},
    )
    sparton_backend: str = field(
        default=None,
        metadata={
            "help": "Sparton backend for head='sparton' ('hybrid', 'naive', or "
            "'optimized'); default keeps Sparton's own resolution"
        },
    )


@dataclass
class LSRDataArguments:
    dataset_name: str = field(
        default="nthakur/swim-ir-cross-lingual",
        metadata={"help": "HuggingFace dataset identifier"},
    )
    languages: str = field(
        default="de,es,fr",
        metadata={"help": "Comma-separated language codes to load from the dataset"},
    )
    query_max_length: int = field(default=64)
    document_max_length: int = field(default=256)
    query_column: str = field(default="query")
    document_column: str = field(default="text")


@dataclass
class LSRTrainingArguments(TrainingArguments):
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for InfoNCE loss"},
    )
    lambda_l1: float = field(
        default=1e-4,
        metadata={"help": "Target L1 regularization weight"},
    )
    lambda_flops: float = field(
        default=1e-4,
        metadata={"help": "Target FLOPs regularization weight"},
    )
    reg_warmup_steps: int = field(
        default=10000,
        metadata={"help": "Linear warmup steps for regularization weights"},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep all dataset columns so the collator can access query/text"},
    )


def info_nce_loss(query_reps, doc_reps, temperature=1.0):
    """In-batch negatives contrastive loss."""
    scores = torch.matmul(query_reps, doc_reps.t()) / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)


def l1_regularization(reps):
    """L1 norm on sparse representations — encourages sparsity."""
    return torch.mean(torch.sum(torch.abs(reps), dim=-1))


def flops_regularization(reps):
    """FLOPs reg: penalizes frequent term activation across the batch."""
    return torch.mean(torch.sum(reps, dim=0) ** 2)


def get_reg_weight(current_step, warmup_steps, target_lambda):
    """Linear warmup from 0 to target_lambda."""
    if warmup_steps <= 0:
        return target_lambda
    return target_lambda * min(1.0, current_step / warmup_steps)


class ContrastiveCollator:
    """Tokenizes query/document pairs on-the-fly with dynamic padding."""

    def __init__(
        self,
        tokenizer,
        query_max_length=64,
        document_max_length=256,
        query_column="query",
        document_column="text",
    ):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.query_column = query_column
        self.document_column = document_column

    @staticmethod
    def _pad_to_multiple(batch, multiple=256):
        """Pad input_ids and attention_mask to a multiple of `multiple`."""
        seq_len = batch["input_ids"].size(1)
        remainder = seq_len % multiple
        if remainder == 0:
            return batch
        pad_len = multiple - remainder
        batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, pad_len), value=0)
        batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, pad_len), value=0)
        return batch

    def __call__(self, features):
        queries = [f[self.query_column] for f in features]
        documents = [f["title"] + " " + f[self.document_column] for f in features]

        query_batch = self.tokenizer(
            queries,
            max_length=self.query_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        doc_batch = self.tokenizer(
            documents,
            max_length=self.document_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        query_batch = self._pad_to_multiple(query_batch, multiple=8)
        doc_batch = self._pad_to_multiple(doc_batch, multiple=64)

        return {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "doc_input_ids": doc_batch["input_ids"],
            "doc_attention_mask": doc_batch["attention_mask"],
        }


class LSRTrainer(Trainer):
    """HuggingFace Trainer with InfoNCE + L1/FLOPs regularization."""

    def __init__(self, model_args, data_args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.custom_logs = defaultdict(float)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query_output = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        doc_output = model(
            input_ids=inputs["doc_input_ids"],
            attention_mask=inputs["doc_attention_mask"],
        )

        query_reps = query_output["reps"]
        doc_reps = doc_output["reps"]

        # InfoNCE
        contrastive_loss = info_nce_loss(
            query_reps, doc_reps, self.args.temperature
        )

        # Regularization with warmup
        current_step = self.state.global_step
        reg_w_l1 = get_reg_weight(
            current_step, self.args.reg_warmup_steps, self.args.lambda_l1
        )
        reg_w_flops = get_reg_weight(
            current_step, self.args.reg_warmup_steps, self.args.lambda_flops
        )

        l1_q = l1_regularization(query_reps)
        l1_d = l1_regularization(doc_reps)
        flops_q = flops_regularization(query_reps)
        flops_d = flops_regularization(doc_reps)

        reg_loss = reg_w_l1 * (l1_q + l1_d) + reg_w_flops * (flops_q + flops_d)
        total_loss = contrastive_loss + reg_loss

        # Logging
        self.custom_logs["contrastive_loss"] += contrastive_loss.detach().item()
        self.custom_logs["l1_reg"] += (l1_q + l1_d).detach().item()
        self.custom_logs["flops_reg"] += (flops_q + flops_d).detach().item()
        self.custom_logs["reg_weight_l1"] += reg_w_l1
        self.custom_logs["reg_weight_flops"] += reg_w_flops
        self.custom_logs["query_nonzero_ratio"] += (
            (query_reps > 0).float().mean().item()
        )
        self.custom_logs["doc_nonzero_ratio"] += (
            (doc_reps > 0).float().mean().item()
        )

        if return_outputs:
            return total_loss, {"loss": total_loss}
        return total_loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.control.should_log and self.state.global_step > 0:
            steps_since_last = max(
                1,
                self.state.global_step - self._globalstep_last_logged,
            ) * self.args.gradient_accumulation_steps
            log = {}
            for metric, value in self.custom_logs.items():
                log[metric] = round(value / steps_since_last, 6)
            self.log(log)
            for metric in self.custom_logs:
                self.custom_logs[metric] = 0.0
            self.control.should_log = True
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        # SpladeModel ties the projection head weight to the backbone word
        # embeddings; safetensors refuses shared tensors and SpladeModel is a
        # plain nn.Module, so Trainer._save cannot serialize it. torch.save
        # handles shared storage natively.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, "pytorch_model.bin"),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def load_swim_ir_dataset(dataset_name, languages, **kwargs):
    """Load and concatenate swim-ir subsets for the given languages."""
    lang_list = [lang.strip() for lang in languages.split(",")]
    datasets = []
    for lang in lang_list:
        print(f"Loading {dataset_name} [{lang}]...")
        ds = load_dataset(dataset_name, lang, split="train", **kwargs)
        datasets.append(ds)
    combined = concatenate_datasets(datasets)
    print(f"Total training samples: {len(combined)}")
    return combined

def main():
    parser = HfArgumentParser(
        (LSRModelArguments, LSRDataArguments, LSRTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Model
    model = SpladeModel(
        model_name_or_path=model_args.model_name_or_path,
        head=model_args.head,
        sparton_backend=model_args.sparton_backend,
    )

    # Dataset
    dataset = load_swim_ir_dataset(
        data_args.dataset_name,
        data_args.languages,
    )

    # Collator
    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        query_max_length=data_args.query_max_length,
        document_max_length=data_args.document_max_length,
        query_column=data_args.query_column,
        document_column=data_args.document_column,
    )

    # Trainer
    trainer = LSRTrainer(
        model_args=model_args,
        data_args=data_args,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Train
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
