def get_config(variant="attn", ckpt_path=None):
    return {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn" if variant == "attn" else "avg",
        "resolution": [160, 256],
        "video_length": 60,
        "device": "cuda",  # will be overridden
        "ckpt_path": ckpt_path
    }
