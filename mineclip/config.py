# def get_config(variant="attn", ckpt_path=None):
#     return {
#         "arch": "vit_base_p16_fz.v2.t2",
#         "hidden_dim": 512,
#         "image_feature_dim": 512,
#         "mlp_adapter_spec": "v0-2.t0",
#         "pool_type": "attn" if variant == "attn" else "avg",
#         "resolution": [160, 256],
#         "ckpt_path": ckpt_path
#     }

def get_config(variant="attn", ckpt_path=None, checksum=None):
    return {
        "arch": "vit_base_p16_fz.v2.t2",
        "hidden_dim": 512,
        "image_feature_dim": 512,
        "mlp_adapter_spec": "v0-2.t0",
        "pool_type": "attn.mean" if variant == "attn" else "avg",
        "resolution": [160, 256],

        # ✅ Add this block to support model.load_ckpt()
        "ckpt": {
            "path": ckpt_path,
            "checksum": checksum,
        }
    }
