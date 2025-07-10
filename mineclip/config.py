def get_config(variant: str = "attn", ckpt_path: str = None):
    assert variant in ["attn", "avg"], "variant must be 'attn' or 'avg'"

    return {
        # model architecture
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn" if variant == "attn" else "avg",
        "resolution": [160, 256],

        # CLIP/MineCLIP-specific settings
        "clip_model": "ViT-B/32",
        "mc_version": "1.18",
        "use_attn_pooling": variant == "attn",
        "dropout": 0.0,
        "video_length": 60,

        # Device
        "device": "cuda",  # can override later

        # Checkpoint
        "ckpt_path": ckpt_path,
        "ckpt_checksum": None,
    }
