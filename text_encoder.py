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
            self.act_1 = FP32SiLU()
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
        
        # Load text encoders from SD3 model subfolders (like the training script)
        try:
            self.clip_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder"
            )  # First CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP encoder from SD3 model '{sd3_model_path}/text_encoder': {str(e)}")
            
        try:
            self.clip_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder_2"
            )  # Second CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP encoder from SD3 model '{sd3_model_path}/text_encoder_2': {str(e)}")
            
        # Load two separate tokenizers (like SD3)
        try:
            self.clip_tokenizer_1 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer"
            )  # First tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer': {str(e)}")
            
        try:
            self.clip_tokenizer_2 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer_2"
            )  # Second tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer_2': {str(e)}")
        
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
        """
        Forward pass of TextEmbedder.
        Args:
            text: string or list of strings
            train: whether in training mode
            force_drop_ids: optional tensor to force certain items to be dropped
        Returns:
            embeddings: (batch_size, hidden_size)
        """
        use_dropout = self.dropout_prob > 0
        
        # Tokenize text with both tokenizers
        text_input_ids_1, text_input_ids_2 = self.tokenize_text(text)
        
        # Apply dropout for classifier-free guidance
        if (train and use_dropout) or (force_drop_ids is not None):
            drop_ids = self.token_drop(text_input_ids_1, text_input_ids_2, force_drop_ids)
        else:
            drop_ids = torch.zeros(text_input_ids_1.shape[0], dtype=torch.bool, device=text_input_ids_1.device)
        
        # Encode text
        text_embeddings = self.encode_text(text_input_ids_1, text_input_ids_2)
        drop_ids = drop_ids.to(text_embeddings.device)
        # Replace dropped embeddings with empty embedding
        if drop_ids.any():
            device = text_embeddings.device
            empty_emb = self.empty_embedding.to(device)
            text_embeddings = torch.where(
                drop_ids.unsqueeze(1).expand_as(text_embeddings),
                empty_emb.expand(text_embeddings.shape[0], -1),
                text_embeddings
            )
        # print(f"shape of text_embeddings is {text_embeddings.shape}")
        return text_embeddings