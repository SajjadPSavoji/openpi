# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block (dims inferred)."""
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self,
                 visual:    jnp.ndarray,  # [batch, seq_len, vision_dim]
                 lang_mean: jnp.ndarray   # [batch, llm_dim]
                 ) -> jnp.ndarray:
        # infer dims from runtime shapes
        vision_dim = visual.shape[-1]

        # project language → γ, β (features=vision_dim)
        gamma = nn.Dense(vision_dim,
                         dtype=self.dtype_mm,
                         name="film_scale")(lang_mean)
        beta  = nn.Dense(vision_dim,
                         dtype=self.dtype_mm,
                         name="film_shift")(lang_mean)

        # broadcast over seq dimension
        gamma = gamma[:, None, :]
        beta  = beta[:, None, :]

        # keep your FSDP sharding constraints
        visual = sharding.activation_sharding_constraint(visual)
        out    = (1.0 + gamma) * visual + beta
        return sharding.activation_sharding_constraint(out)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + FiLM + MLP)."""

    mlp_dim:   int | None = None
    num_heads: int        = 12
    dropout:   float      = 0.0
    dtype_mm:  str        = "float32"

    @nn.compact
    def __call__(self,
                 x:          jnp.ndarray,  # [batch, seq_len, vision_dim]
                 lang_mean:  jnp.ndarray,  # [batch, llm_dim]
                 deterministic: bool = True):
        out = {}

        # Self-attention as before
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        # Pre-MLP LayerNorm
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)

        # FiLM modulation (dims inferred inside FiLMBlock)
        y = out["film"] = FiLMBlock(
            dtype_mm=self.dtype_mm,
            name="FiLM",
        )(y, lang_mean)

        # MLP + residual
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y

        x = sharding.activation_sharding_constraint(x)
        return x, out

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence-to-sequence translation with FiLM support."""

    depth:        int
    mlp_dim:      int | None = None  # Defaults to 4× input dim
    num_heads:    int        = 12
    dropout:      float      = 0.0
    scan:         bool       = False
    remat_policy: str        = "nothing_saveable"
    dtype_mm:     str        = "float32"

    @nn.compact
    def __call__(
        self,
        x:          jnp.ndarray,  # [batch, seq_len, vision_dim]
        lang_mean:  jnp.ndarray,  # [batch, llm_dim]
        deterministic: bool = True
    ):
        out: dict[str, Any] = {}

        if self.scan:
            # Wrap Encoder1DBlock in remat/ checkpointing
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # treat `deterministic` as static
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )

            # Scan over `depth` steps, broadcasting x & lang_mean & deterministic each time,
            # but giving each layer its own params via variable_axes={"params": 0}.
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, lang_mean, deterministic)

            # Unpack per-layer outputs
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)

        else:
            # Simple Python loop over depth
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, lang_mean, deterministic)

            out["pre_ln"] = x  # alias for last block output

        # Final layer norm
        x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)
        return x, out



class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model with language-conditioned FiLM in the encoder."""

    num_classes: int | None = None
    patch_size:  Sequence[int] = (16, 16)
    width:       int         = 768
    depth:       int         = 12
    mlp_dim:     int | None  = None  # Defaults to 4× input dim
    num_heads:   int         = 12
    posemb:      str         = "learn"  # Can also be "sincos2d"
    rep_size:    int | bool  = False
    dropout:     float       = 0.0
    pool_type:   str         = "gap"    # Can also be "map", "0", "tok", or "none"
    head_zeroinit: bool      = True
    scan:        bool        = False
    remat_policy:str         = "nothing_saveable"
    dtype_mm:    str         = "float32"

    @nn.compact
    def __call__(
        self,
        image:     jnp.ndarray,  # [batch, H, W, 3]
        lang_mean: jnp.ndarray,  # [batch, llm_dim]
        *,
        train:    bool = False
    ):
        out: dict[str, Any] = {}

        # 1) Patch‐embed + pos-emb
        image = jnp.asarray(image, jnp.float32)
        x     = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        x = out["with_posemb"] = x + get_posemb(
            self, self.posemb, (h, w), c, "pos_embedding", jnp.float32
        )

        # 2) Optional CLS token
        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x   = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        # 3) Dropout + cast
        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        # 4) → our modified Encoder that takes lang_mean
        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, lang_mean, not train)
        encoded = out["encoded"] = x

        # 5) Pooling head (map, gap, etc.)
        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type in ("0", "tok"):
            x = out["head_input"] = x[:, 0]
            if self.pool_type == "tok":
                encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            x = encoded
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        # 6) Reshape for 2D head, rep_size, logits
        x_2d = jnp.reshape(encoded, [n, h, w, -1])
        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x    = nn.tanh(hid(x))
        out["pre_logits_2d"] = x_2d
        out["pre_logits"]    = x

        if self.num_classes:
            kw   = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x    = out["logits"]    = head(x)

        return x, out



def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }