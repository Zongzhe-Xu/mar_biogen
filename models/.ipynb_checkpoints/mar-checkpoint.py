from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = nn.SiLU()  # Fallback to regular SiLU
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TextEmbedder(nn.Module):
    """
    Embeds text into vector representations using CLIP ViT-L/14 and OpenCLIP ViT-bigG/14 (like SD3).
    Compatible with LabelEmbedder interface for classifier-free guidance.
    """
    def __init__(self, hidden_size, dropout_prob, 
                 sd3_model_path="/projects/besp/BiosignalGen_zitao/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"): # SD3 model path
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        # Import CLIP components
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError:
            raise ImportError("transformers library is required for TextEmbedder")
        
        # Load text encoders from SD3 model subfolders
        # Since you're loading locally, all processes can load simultaneously
        try:
            self.clip_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder", local_files_only=True
            )  # First CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP encoder from SD3 model '{sd3_model_path}/text_encoder': {str(e)}")
            
        try:
            self.clip_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder_2", local_files_only=True
            )  # Second CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP encoder from SD3 model '{sd3_model_path}/text_encoder_2': {str(e)}")
            
        # Load two separate tokenizers (like SD3)
        try:
            self.clip_tokenizer_1 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer", local_files_only=True
            )  # First tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer': {str(e)}")
            
        try:
            self.clip_tokenizer_2 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer_2", local_files_only=True
            )  # Second tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer_2': {str(e)}")
        
        # Register CLIP encoders as submodules so they get moved to device automatically
        # self.register_module('clip_encoder_1', self.clip_encoder_1)
        # self.register_module('clip_encoder_2', self.clip_encoder_2)
        
        # Freeze the CLIP encoders
        for param in self.clip_encoder_1.parameters():
            param.requires_grad = False
        for param in self.clip_encoder_2.parameters():
            param.requires_grad = False
        
        # Get CLIP embedding dimensions
        clip_hidden_size_1 = self.clip_encoder_1.config.hidden_size
        clip_hidden_size_2 = self.clip_encoder_2.config.hidden_size
        
        # Concatenate pooled embeddings first, then use single projection (like SD3)
        combined_input_size = clip_hidden_size_1 + clip_hidden_size_2
        self.projection = PixArtAlphaTextProjection(
            in_features=combined_input_size,  # pooled_projection_dim
            hidden_size=hidden_size,          # embedding_dim
            out_features=hidden_size,         # embedding_dim
            act_fn="silu"
        )
        
        # For classifier-free guidance: empty text embedding
        # self.register_buffer("empty_embedding", torch.randn(1, hidden_size) * 0.02)
        self.empty_embedding = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        
    def tokenize_text(self, text):
        """
        Tokenize text using both CLIP tokenizers (like SD3).
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize with first tokenizer
        text_inputs_1 = self.clip_tokenizer_1(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize with second tokenizer
        text_inputs_2 = self.clip_tokenizer_2(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        return text_inputs_1.input_ids, text_inputs_2.input_ids
    
    def encode_text(self, text_input_ids_1, text_input_ids_2):
        """
        Encode tokenized text using two CLIP encoders and get pooled embeddings.
        """
        with torch.no_grad():
            # Move to same device as encoders
            device = next(self.clip_encoder_1.parameters()).device
            text_input_ids_1 = text_input_ids_1.to(device)
            text_input_ids_2 = text_input_ids_2.to(device)
            
            # Encode with first CLIP encoder using first tokenizer
            outputs_1 = self.clip_encoder_1(text_input_ids_1, output_hidden_states=True)
            pooled_embeds_1 = outputs_1[0]  # First element of the tuple
            
            # Encode with second CLIP encoder using second tokenizer
            outputs_2 = self.clip_encoder_2(text_input_ids_2, output_hidden_states=True)
            pooled_embeds_2 = outputs_2[0]  # First element of the tuple
            
            # Concatenate pooled embeddings along the last dimension
            combined_pooled_embeds = torch.cat([pooled_embeds_1, pooled_embeds_2], dim=-1)
            
            # Project using single PixArtAlphaTextProjection (like SD3)
        text_embeddings = self.projection(combined_pooled_embeds)
        
        return text_embeddings
    
    def token_drop(self, text_input_ids_1, text_input_ids_2, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(text_input_ids_1.shape[0], device=text_input_ids_1.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        return drop_ids
        
    def forward(self, text, train, force_drop_ids=None):
        import torch.distributed as dist
        use_dropout = self.dropout_prob > 0

        # Tokenize (CPU is fine here)
        text_input_ids_1, text_input_ids_2 = self.tokenize_text(text)

        # Build per-sample mask on the same device as the token IDs (likely CPU)
        if (train and use_dropout) or (force_drop_ids is not None):
            drop_ids = self.token_drop(text_input_ids_1, text_input_ids_2, force_drop_ids)
        else:
            drop_ids = torch.zeros(text_input_ids_1.shape[0], dtype=torch.bool, device=text_input_ids_1.device)

        # Encode text -> embeddings are on the model/device (CUDA for DDP+NCCL)
        text_embeddings = self.encode_text(text_input_ids_1, text_input_ids_2)

        # ***** MOVE/BROADCAST ON CUDA (NCCL requires CUDA tensors) *****
        # 1) move mask to embeddings' device, 2) cast to int for NCCL, 3) broadcast
        if dist.is_available() and dist.is_initialized():
            sync = drop_ids.to(device=text_embeddings.device, dtype=torch.int32).contiguous()
            dist.broadcast(sync, src=0)               # NCCL broadcast on CUDA tensor
            drop_ids = sync.to(dtype=torch.bool)      # back to bool for masking
        else:
            drop_ids = drop_ids.to(text_embeddings.device)

        # Replace dropped embeddings with learnable empty embedding
        if drop_ids.any():
            empty_emb = self.empty_embedding.to(text_embeddings.device)
            text_embeddings = torch.where(
                drop_ids.unsqueeze(1).expand_as(text_embeddings),
                empty_emb.expand(text_embeddings.shape[0], -1),
                text_embeddings
            )

        # Always-touch trick so empty_embedding gets a grad slot every step
        text_embeddings = text_embeddings + 0.0 * self.empty_embedding.sum()

        return text_embeddings


# def mask_by_order(mask_len, order, bsz, seq_len):
#     device = order.device  # Use the same device as the input tensor
#     masking = torch.zeros(bsz, seq_len, device=device)
#     masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=device)).bool()
#     return masking

def mask_by_order(mask_len, order, bsz, seq_len):
    device = order.device
    mlen = int(mask_len.item() if torch.is_tensor(mask_len) else mask_len)
    idx = order[:, :mlen].long()                # Long indices are required
    masking = torch.zeros(bsz, seq_len, device=device)
    masking.scatter_(dim=-1, index=idx, value=1.0)
    return masking.bool()


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 use_text_conditioning=False,
                 sd3_model_path=None,
                 time_series_length=None,
                 timeseries_channels=None,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Time series setup (commented out image-specific parameters)
        # self.vae_embed_dim = vae_embed_dim
        # self.img_size = img_size
        # self.vae_stride = vae_stride
        # self.patch_size = patch_size
        # self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        
        # Time series parameters
        if time_series_length is None:
            raise ValueError("time_series_length must be provided")
        if timeseries_channels is None:
            raise ValueError("timeseries_channels must be provided")
            
        self.time_series_length = time_series_length
        self.timeseries_channels = timeseries_channels
        self.patch_size = patch_size  # Use the patch_size parameter
        
        # Calculate effective sequence length after patching
        if time_series_length % patch_size != 0:
            raise ValueError(f"time_series_length ({time_series_length}) must be divisible by patch_size ({patch_size})")
        self.seq_len = time_series_length // patch_size
        
        # Use timeseries_channels for token_embed_dim to match diffusion loss expectations
        self.vae_embed_dim = vae_embed_dim  # Keep for internal compatibility
        self.token_embed_dim = vae_embed_dim * patch_size  # Each patch contains timeseries_channels * patch_size dimensions
        
        # Convnet for converting timeseries_channels to vae_embed_dim and patching
        # kernel_size=stride=patch_size achieves patching effect in one operation
        # This is more efficient than separate channel conversion + reshape operations
        self.patch_conv = nn.Conv1d(
            in_channels=timeseries_channels,
            out_channels=vae_embed_dim * patch_size,  # Output channels = vae_embed_dim * patch_size
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
            
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Conditioning setup
        self.use_text_conditioning = use_text_conditioning
        self.label_drop_prob = label_drop_prob
        
        if use_text_conditioning:
            # Text conditioning
            if sd3_model_path is None:
                raise ValueError("sd3_model_path must be provided when use_text_conditioning=True")
            self.text_embedder = TextEmbedder(
                hidden_size=encoder_embed_dim, 
                dropout_prob=0.0,
                sd3_model_path=sd3_model_path
            )
            # Fake embedding for CFG's unconditional generation
            self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        else:
            # Label conditioning
            self.num_classes = class_num
            self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
            # Fake class embedding for CFG's unconditional generation
            self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=timeseries_channels * patch_size,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        if self.use_text_conditioning:
            # Initialize text conditioning components
            torch.nn.init.normal_(self.fake_latent, std=.02)
        else:
            # Initialize label conditioning components
            torch.nn.init.normal_(self.class_emb.weight, std=.02)
            torch.nn.init.normal_(self.fake_latent, std=.02)
            
        # Initialize convnet layers for channel conversion and patching
        torch.nn.init.xavier_uniform_(self.patch_conv.weight)
        if self.patch_conv.bias is not None:
            nn.init.constant_(self.patch_conv.bias, 0)
            

            

            
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # Time series mode: x shape is [batch_size, timeseries_channels, seq_len]
        # Apply patching using 1D convolution with kernel_size=stride=patch_size
        bsz, c, seq_len = x.shape
        if seq_len % self.patch_size != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by patch_size ({self.patch_size})")
        
        # Apply 1D convolution for patching
        # Input: [batch_size, timeseries_channels, seq_len]
        # Output: [batch_size, vae_embed_dim * patch_size, num_patches]
        x = self.patch_conv(x)
        
        # Transpose to get [batch_size, num_patches, vae_embed_dim * patch_size]
        x = x.transpose(1, 2)
        
        return x
        
        # Image mode (commented out)
        # bsz, c, h, w = x.shape
        # p = self.patch_size
        # h_, w_ = h // p, w // p
        # x = x.reshape(bsz, c, h_, p, w_, p)
        # x = torch.einsum('nchpwq->nhwcpq', x)
        # x = x.reshape(bsz, h_ * w_, c * p ** 2)
        # return x  # [n, l, d]

    def unpatchify(self, x):
        # Time series mode: x shape is [batch_size, num_patches, timeseries_channels * patch_size]
        # Use reshape operations for unpatchify (reverse of patchify)
        bsz = x.shape[0]
        p = self.patch_size
        c = self.timeseries_channels
        seq_len = self.time_series_length

        assert x.shape[1]*p == seq_len, f"seq_len {seq_len} must be divisible by patch_size {p}"
        
        # Reshape back to [batch_size, timeseries_channels, seq_len]
        num_patches = seq_len // p
        x = x.reshape(bsz, num_patches, c, p)
        x = x.permute(0, 2, 1, 3)  # [batch_size, timeseries_channels, num_patches, patch_size]
        x = x.contiguous()  # Make tensor contiguous after permute
        x = x.reshape(bsz, c, seq_len)  # [batch_size, timeseries_channels, seq_len]
        
        return x
        
        # Image mode (commented out)
        # bsz = x.shape[0]
        # p = self.patch_size
        # c = self.vae_embed_dim
        # h_, w_ = self.seq_h, self.seq_w
        # x = x.reshape(bsz, h_, w_, c, p, p)
        # x = torch.einsum('nhwcpq->nchpwq', x)
        # x = x.reshape(bsz, c, h_ * p, w_ * p)
        # return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long().to(next(self.parameters()).device)
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        import torch.distributed as dist
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device, dtype=mask.dtype), mask], dim=1)

        # --- Classifier-free drop of class embedding (DDP-safe) ---
        if self.training:
            # per-rank draw, then broadcast so ALL ranks use THE SAME mask
            drop_latent_mask = (torch.rand(bsz, device=x.device) < self.label_drop_prob)

            if dist.is_available() and dist.is_initialized():
                t = drop_latent_mask.to(torch.int32).contiguous()
                dist.broadcast(t, src=0)              # every rank now has identical mask
                drop_latent_mask = t.bool()

            # shape + dtype alignment
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(dtype=x.dtype)

            # select unconditional vs conditional class embedding
            #   self.fake_latent: [1, D]  -> expand to [B, D]
            class_embedding = torch.where(
                drop_latent_mask.bool(),
                self.fake_latent.expand(bsz, -1).to(dtype=class_embedding.dtype, device=class_embedding.device),
                class_embedding
            )

            # Always-touch trick: ensure fake_latent participates in the graph every step
            # (adds zero, but creates a grad path so DDP hooks fire even if mask is all False)
            class_embedding = class_embedding + 0.0 * self.fake_latent.sum()

        # write class tokens into the prefix buffer
        x[:, :self.buffer_size] = class_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(~mask_with_buffer.bool()).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(~mask_with_buffer.bool()).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):
        """
        Forward pass for time series generation.
        
        Args:
            imgs: Time series data of shape [batch_size, timeseries_channels, seq_len]
            labels: Either class labels (integers) or text prompts (strings) depending on use_text_conditioning
            
        Returns:
            loss: Training loss
        """

        # Get conditioning embeddings
        if self.use_text_conditioning:
            # Text conditioning
            class_embedding = self.text_embedder(labels, train=self.training)
        else:
            # Label conditioning
            class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        # Time series mode: input is [batch_size, timeseries_channels, seq_len]
        # Apply patching using 1D convolution (combines channel conversion and patching)
        x = self.patchify(imgs)  # [batch_size, num_patches, vae_embed_dim * patch_size]
        # Note: x is used for the MAR encoder/decoder, but the diffusion loss targets the original time series format
        
        # Create ground truth latents in the original time series format
        # This is the target for the diffusion loss - the model learns to generate this format directly
        # Reshape from [batch_size, timeseries_channels, seq_len] to [batch_size, num_patches, timeseries_channels * patch_size]
        gt_latents = imgs.reshape(imgs.size(0), imgs.size(1), imgs.size(2)//self.patch_size, self.patch_size).clone()
        gt_latents = gt_latents.permute(0, 2, 1, 3)  # [batch_size, num_patches, timeseries_channels, patch_size]
        gt_latents = gt_latents.reshape(imgs.size(0), imgs.size(2)//self.patch_size, imgs.size(1) * self.patch_size)
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

#     def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

#         # init and sample generation orders
#         device = next(self.parameters()).device
#         mask = torch.ones(bsz, self.seq_len, device=device)
#         # tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)
#         tokens = torch.zeros(bsz, self.seq_len, self.timeseries_channels*self.patch_size, device=device)
#         orders = self.sample_orders(bsz)

#         indices = list(range(num_iter))
#         if progress:
#             indices = tqdm(indices)
#         # generate latents
#         for step in indices:
#             cur_tokens = tokens.clone()

#             patched_tokens = self.patchify(self.unpatchify(cur_tokens))

#             # class embedding and CFG
#             if labels is not None:
#                 if self.use_text_conditioning:
#                     # Text conditioning
#                     class_embedding = self.text_embedder(labels, train=False)
#                 else:
#                     # Label conditioning
#                     class_embedding = self.class_emb(labels)
#             else:
#                 class_embedding = self.fake_latent.repeat(bsz, 1)
#             if not cfg == 1.0:
#                 patched_tokens = torch.cat([patched_tokens, patched_tokens], dim=0)
#                 class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
#                 mask = torch.cat([mask, mask], dim=0)

#             # mae encoder
#             x = self.forward_mae_encoder(patched_tokens, mask, class_embedding)

#             # mae decoder
#             z = self.forward_mae_decoder(x, mask)

#             # mask ratio for the next round, following MaskGIT and MAGE.
#             mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
#             mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device=device)

#             # masks out at least one for the next iteration
#             mask_len = torch.maximum(torch.tensor([1], device=device),
#                                      torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

#             # get masking for next iteration and locations to be predicted in this iteration
#             mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
#             if step >= num_iter - 1:
#                 mask_to_pred = mask[:bsz].bool()
#             else:
#                 mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
#             mask = mask_next
#             if not cfg == 1.0:
#                 mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

#             # sample token latents for this step
#             z = z[mask_to_pred.nonzero(as_tuple=True)].to(torch.float32)
#             # cfg schedule follow Muse
#             if cfg_schedule == "linear":
#                 cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
#             elif cfg_schedule == "constant":
#                 cfg_iter = cfg
#             else:
#                 raise NotImplementedError
#             # Make sure CFG is a Python float (not a 0-D tensor)
#             cfg_iter = float(cfg_iter.item()) if torch.is_tensor(cfg_iter) else float(cfg_iter)
#             sampled_token_latent = self.diffloss.sample(z, float(temperature), cfg_iter)
#             if not cfg == 1.0:
#                 sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
#                 mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

#             cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
#             tokens = cur_tokens.clone()

#         # The diffusion model generates [batch_size, num_patches, timeseries_channels * patch_size]
#         # Use unpatchify to convert back to [batch_size, timeseries_channels, seq_len]
#         tokens = self.unpatchify(tokens)
        
#         return tokens

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        # init and sample generation orders
        device = next(self.parameters()).device
        mask = torch.ones(bsz, self.seq_len, device=device)
        # tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)
        tokens = torch.zeros(bsz, self.seq_len, self.timeseries_channels*self.patch_size, device=device)
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            patched_tokens = self.patchify(self.unpatchify(cur_tokens))
            # ---- STEP 4 sentinel ----
            assert torch.isfinite(patched_tokens).all(), "patched_tokens has non-finite values"

            # class embedding and CFG
            if labels is not None:
                if self.use_text_conditioning:
                    # Text conditioning
                    class_embedding = self.text_embedder(labels, train=False)
                else:
                    # Label conditioning
                    class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                patched_tokens = torch.cat([patched_tokens, patched_tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(patched_tokens, mask, class_embedding)
            # ---- STEP 4 sentinel ----
            assert torch.isfinite(x).all(), "encoder output x has non-finite values"

            # mae decoder
            z = self.forward_mae_decoder(x, mask)
            # ---- STEP 4 sentinel ----
            assert torch.isfinite(z).all(), "decoder output z has non-finite values"

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device=device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.tensor([1], device=device),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)].to(torch.float32)
            # ---- STEP 4 sentinel ----
            assert torch.isfinite(z).all(), "z (to sample) has non-finite values"

            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            # Make sure CFG is a Python float (not a 0-D tensor)
            cfg_iter = float(cfg_iter.item()) if torch.is_tensor(cfg_iter) else float(cfg_iter)
            sampled_token_latent = self.diffloss.sample(z, float(temperature), cfg_iter)
            # ---- STEP 4 sentinel ----
            assert torch.isfinite(sampled_token_latent).all(), "DiffLoss.sample produced non-finite values"

            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # The diffusion model generates [batch_size, num_patches, timeseries_channels * patch_size]
        # Use unpatchify to convert back to [batch_size, timeseries_channels, seq_len]
        tokens = self.unpatchify(tokens)
        # ---- STEP 4 sentinel ----
        assert torch.isfinite(tokens).all(), "final tokens (after unpatchify) have non-finite values"

        return tokens


def mar_tiny(**kwargs):
    model = MAR(
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
