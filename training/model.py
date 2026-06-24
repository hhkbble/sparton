import torch
from torch import nn
from transformers import AutoModelForMaskedLM
from transformers.activations import gelu


def _get_backbone_and_head(llm):
    """
    Extract backbone, transform, and decoder from an AutoModelForMaskedLM.
    Supports bert and roberta/xlm-roberta architectures. Can be easily extended.
    """
    model_type = llm.config.model_type

    if model_type in ("bert", "distilbert"):
        backbone = llm.bert
        predictions = llm.cls.predictions
        transform = predictions.transform
        decoder = predictions.decoder
        return backbone, transform, decoder

    elif model_type in ("roberta", "xlm-roberta"):
        backbone = llm.roberta
        lm_head = llm.lm_head

        class RobertaTransform(nn.Module):
            def __init__(self, lm_head):
                super().__init__()
                self.dense = lm_head.dense
                self.layer_norm = lm_head.layer_norm

            def forward(self, x):
                x = self.dense(x)
                x = gelu(x)
                x = self.layer_norm(x)
                return x

        transform = RobertaTransform(lm_head)
        decoder = lm_head.decoder
        return backbone, transform, decoder

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: bert, distilbert, roberta, xlm-roberta."
        )


class ProjectionPyTorch(nn.Module):
    """Pure PyTorch head: transform → decoder → max-pool → relu → log1p."""

    def __init__(self, transform, decoder):
        super().__init__()
        self.transform = transform
        self.decoder = decoder

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        reps = torch.log1p(
            torch.relu(logits * attention_mask.unsqueeze(-1))
        ).max(dim=1).values
        return reps


class ProjectionSparton(nn.Module):
    """Wraps the MLM transform + SpartonHead (Triton kernel) with weight tying."""

    def __init__(self, transform, decoder, sparton_kernel=None):
        super().__init__()
        self.transform = transform

        from sparton import SpartonHead

        vocab_size = decoder.out_features
        hidden_dim = decoder.in_features
        device = decoder.weight.device
        self.sparton_head = SpartonHead(
            vocab_size,
            hidden_dim,
            use_bias=True,
            kernel=sparton_kernel,
        ).to(device=device)
        self.sparton_head.tie_weights(decoder)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.transform(hidden_states)
        reps = self.sparton_head(hidden_states, attention_mask)
        return reps


class SpladeModel(nn.Module):
    """
    Flexible Splade model with switchable PyTorch / Triton head.

    Args:
        model_name_or_path: HuggingFace model identifier or local path
        head: "torch", "sparton", or "compiled" (torch.compile'd PyTorch head)
        model_kwargs: optional dict passed to from_pretrained
        sparton_kernel: optional Sparton kernel name for head="sparton"
            ("hybrid", "naive", or "optimized"); None keeps the Sparton
            default resolution (SPARTON_KERNEL env var or "hybrid")
    """

    def __init__(
        self,
        model_name_or_path,
        head="torch",
        model_kwargs=None,
        sparton_kernel=None,
    ):
        super().__init__()
        self.head = head
        self.model_name_or_path = model_name_or_path

        if model_kwargs is None:
            model_kwargs = {}

        model_kwargs.setdefault("attn_implementation", "sdpa")
        llm = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **model_kwargs)

        if head == "torch":
            self.llm = llm
        elif head == "compiled":
            backbone, transform, decoder = _get_backbone_and_head(llm)
            self.backbone = backbone
            self.projection = ProjectionPyTorch(transform, decoder)
        elif head == "sparton":
            backbone, transform, decoder = _get_backbone_and_head(llm)
            self.backbone = backbone
            self.projection = ProjectionSparton(
                transform,
                decoder,
                sparton_kernel=sparton_kernel,
            )
        else:
            raise ValueError(
                f"head must be 'torch', 'compiled', or 'sparton', got '{head}'"
            )

    def forward(self, input_ids, attention_mask, **kwargs):
        if self.head == "torch":
            output = self.llm(input_ids, attention_mask)
            logits = output.logits
            reps = torch.log1p(
                torch.relu(
                    logits * attention_mask.unsqueeze(-1)
                )
            ).max(dim=1).values
        else:  # compiled or triton
            last_hidden_state = self.backbone(
                input_ids, attention_mask
            ).last_hidden_state
            reps = self.projection(last_hidden_state, attention_mask)

        return {"reps": reps}

    def encode(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask, **kwargs)
        return result["reps"]
