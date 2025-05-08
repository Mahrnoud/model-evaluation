import torch
import logging
import time
from transformers import AutoTokenizer
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# Import the Transformer model architecture classes
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class LayerScale(torch.nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        self._update_cos_sin_cache(max_position_embeddings)

    def _update_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            self._update_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embeddings(q, k, cos, sin):
    q_seq_len = q.shape[2]
    k_seq_len = k.shape[2]

    q_cos = cos[:, :, :q_seq_len, :]
    q_sin = sin[:, :, :q_seq_len, :]
    k_cos = cos[:, :, :k_seq_len, :]
    k_sin = sin[:, :, :k_seq_len, :]

    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)

    return q_embed, k_embed


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=2048, use_flash_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_flash_attn = use_flash_attn and False  # Set to False as we're not importing flash_attn

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.attention_dropout = torch.nn.Dropout(dropout)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, init_values=1e-5)
        self.rotary_emb = RotaryEmbedding(self.depth, max_position_embeddings=max_len)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size = query.shape[0]

        if key is None:
            key = query
        if value is None:
            value = query

        q = self.split_heads(self.q_proj(query), batch_size)
        k = self.split_heads(self.k_proj(key), batch_size)
        v = self.split_heads(self.v_proj(value), batch_size)

        max_seq_len = max(q.shape[2], k.shape[2])
        cos, sin = self.rotary_emb(q, seq_len=max_seq_len)
        q, k = apply_rotary_embeddings(q, k, cos, sin)

        # Standard attention (no flash attention)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_weights = self.attention_dropout(torch.nn.functional.softmax(attn_logits, dim=-1))
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)
        # Apply layer scaling and dropout
        output = self.layer_scale(output)
        return self.output_dropout(output)


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(FeedForward, self).__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.w1 = torch.nn.Linear(d_model, d_ff)
        self.w2 = torch.nn.Linear(d_model, d_ff)
        self.w3 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, init_values=1e-5)

    def forward(self, x):
        # SwiGLU-like activation
        gated_output = self.w1(x) * torch.sigmoid(self.w2(x) * 1.0)
        output = self.w3(self.dropout(gated_output))
        # Apply layer scaling
        return self.layer_scale(output)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_input = self.norm1(x)
        attn_output = self.self_attention(attn_input, mask=mask)
        x = x + self.dropout1(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout2(ff_output)

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x) * self.embed_scale
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Pre-norm architecture
        attn_input = self.norm1(x)
        self_attn_output = self.self_attention(attn_input, mask=tgt_mask)
        x = x + self.dropout1(self_attn_output)

        cross_attn_input = self.norm2(x)
        cross_attn_output = self.cross_attention(
            query=cross_attn_input,
            key=enc_output,
            value=enc_output,
            mask=src_mask
        )
        x = x + self.dropout2(cross_attn_output)

        ff_input = self.norm3(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout3(ff_output)

        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = torch.nn.Dropout(dropout)

        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_projection = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * self.embed_scale
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        x = self.norm(x)
        return self.output_projection(x)


class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_encoder_layers, num_decoder_layers,
                 vocab_size, max_len, pad_idx, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_encoder_layers,
                                          vocab_size, max_len, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_decoder_layers,
                                          vocab_size, max_len, dropout)
        self.pad_idx = pad_idx

    def create_masks(self, src, tgt):
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.tensor(tgt)

        src_pad_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0)

        return src_pad_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output


def load_model_and_tokenizer(model_path, tokenizer_path, config=None):
    """
    Load the model and tokenizer from specified paths.

    Args:
        model_path: Path to the saved model checkpoint
        tokenizer_path: Path to the tokenizer
        config: Optional configuration dictionary for model parameters

    Returns:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        device: Device the model is loaded on
    """
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading tokenizer from {tokenizer_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the model from saved checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extract config and state dict
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            # Full model with config
            model_config = checkpoint["config"]
            model_state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Checkpoint with separate state dict
            model_state_dict = checkpoint["model_state_dict"]
            model_config = checkpoint.get("config", config or {
                "d_model": 768,  # Default value, replace with your config
                "num_heads": 12,
                "d_ff": 3072,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "vocab_size": len(tokenizer),
                "max_seq_length": 512,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            })
        else:
            # Just a state dict
            model_state_dict = checkpoint
            model_config = config or {
                "d_model": 768,  # Default value, replace with your config
                "num_heads": 12,
                "d_ff": 3072,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "vocab_size": len(tokenizer),
                "max_seq_length": 512,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            }

        # Create the model
        model = Transformer(
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
            vocab_size=model_config["vocab_size"],
            max_len=model_config.get("max_seq_length", 512),
            pad_idx=model_config.get("pad_token_id", 0),
            dropout=0.0  # Always use 0 dropout for inference
        )

        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully and set to {device} evaluation mode")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    return model, tokenizer, device


def greedy_decode(
        model,
        tokenizer,
        prompt,
        max_length=128,
        device="cpu"
):
    """
    Decode using greedy search (argmax) - completely deterministic.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_length: Maximum length of generated text
        device: Device to run inference on

    Returns:
        output_text: The generated text
    """
    logger.info(f"Performing greedy decoding for prompt: '{prompt}'")

    # Ensure model is in eval mode
    model.eval()

    # Tokenize input
    tokenizer_output = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    input_ids = tokenizer_output["input_ids"].to(device)

    # Log input shape for debugging
    logger.info(f"Input shape: {input_ids.shape}")

    # Get special token IDs
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 3

    # Start with BOS token
    decoder_input = torch.tensor([[bos_token_id]], device=device)
    generated = [bos_token_id]

    # Track generation time
    start_time = time.time()

    with torch.no_grad():
        # Create source mask once
        src_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Generate encoder outputs once for efficiency
        try:
            enc_output = model.encoder(input_ids, src_mask)
        except Exception as e:
            logger.error(f"Error in encoder forward pass: {e}")
            raise

        # Generate tokens one by one
        for i in range(max_length):
            try:
                # Create target mask
                _, tgt_mask = model.create_masks(input_ids, decoder_input)

                # Get decoder output
                dec_output = model.decoder(decoder_input, enc_output, src_mask, tgt_mask)

                # Get next token using argmax (greedy)
                next_token = torch.argmax(dec_output[:, -1, :], dim=-1, keepdim=True)

                # Add the token to the output
                token_id = next_token.item()
                generated.append(token_id)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Stop if we generate EOS token
                if token_id == eos_token_id:
                    logger.info(f"Generated EOS token at position {i + 1}")
                    break

                # Stop on repetition (simple check)
                if i >= 3 and len(set(generated[-4:])) == 1:
                    logger.info(f"Stopping due to repetition at position {i + 1}")
                    break

            except Exception as e:
                logger.error(f"Error at generation step {i}: {e}")
                break

    # Decode the generated sequence
    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f} seconds")

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    logger.info(f"Generated text: '{output_text}'")

    return output_text


def generate_response(prompt, model_path, tokenizer_path, config=None, max_length=128):
    """
    Wrapper function to load model and generate text

    Args:
        prompt: Input text prompt
        model_path: Path to the model checkpoint
        tokenizer_path: Path to the tokenizer
        config: Optional model configuration
        max_length: Maximum length of generated text

    Returns:
        output_text: The generated text
    """
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path, config)

    # Generate text using greedy decoding
    output = greedy_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        device=device
    )

    return output


if __name__ == "__main__":
    # Example usage
    prompt = "What is the capital of France?"
    model_path = "models/latest2/model_full.pth"
    tokenizer_path = "miscovery/tokenizer"

    # Or load model and use greedy_decode directly:
    model, tokenizer, device = load_model_and_tokenizer(model_path, tokenizer_path)
    response = greedy_decode(model, tokenizer, prompt, max_length=512, device=device)
    print(f"Generated response: {response}")
