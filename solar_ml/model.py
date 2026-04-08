"""U-Net model builder using tf.keras application backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class BackboneSpec:
    builder: Callable[..., tf.keras.Model]
    skip_names: tuple[str, ...]


BACKBONES: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(
        builder=tf.keras.applications.ResNet50,
        skip_names=("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"),
    ),
    "resnet101": BackboneSpec(
        builder=tf.keras.applications.ResNet101,
        skip_names=("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out"),
    ),
    "mobilenetv2": BackboneSpec(
        builder=tf.keras.applications.MobileNetV2,
        skip_names=("block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"),
    ),
    "efficientnetb0": BackboneSpec(
        builder=tf.keras.applications.EfficientNetB0,
        skip_names=("block1a_project_bn", "block2b_add", "block3b_add", "block5c_add"),
    ),
    "resnet152": BackboneSpec(
        builder=tf.keras.applications.ResNet152,
        skip_names=("conv1_relu", "conv2_block3_out", "conv3_block8_out", "conv4_block36_out"),
    ),
}


class RGBProjectionInitializer(tf.keras.initializers.Initializer):
    """Initialize 13-band input adapter to approximate RGB from B4/B3/B2."""

    def __call__(self, shape, dtype=None):
        kernel = np.zeros(shape, dtype=np.float32)
        if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] >= 4 and shape[3] >= 3:
            kernel[0, 0, 3, 0] = 1.0
            kernel[0, 0, 2, 1] = 1.0
            kernel[0, 0, 1, 2] = 1.0
        return tf.convert_to_tensor(kernel, dtype=dtype or tf.float32)


def _attention_gate(
    skip: tf.Tensor,
    gating: tf.Tensor,
    inter_filters: int,
    name: str,
) -> tf.Tensor:
    """Additive attention gate (Oktay et al. 2018).

    skip    — encoder feature map at this resolution  (H, W, C_skip)
    gating  — decoder signal from the layer below     (H/2, W/2, C_gate)
              upsampled to match skip spatial dims inside this gate.
    Returns a spatially-weighted version of skip.
    """
    # Project both tensors to the same inter_filters space
    theta = tf.keras.layers.Conv2D(
        inter_filters, 1, padding="same", use_bias=True, name=f"{name}_theta"
    )(skip)
    phi = tf.keras.layers.Conv2D(
        inter_filters, 1, padding="same", use_bias=True, name=f"{name}_phi"
    )(gating)
    # Upsample gating signal to match skip resolution
    phi = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_phi_up")(phi)
    # Additive combination → ReLU → 1×1 conv → sigmoid attention map
    add = tf.keras.layers.Add(name=f"{name}_add")([theta, phi])
    act = tf.keras.layers.Activation("relu", name=f"{name}_relu")(add)
    psi = tf.keras.layers.Conv2D(1, 1, padding="same", use_bias=True, name=f"{name}_psi")(act)
    alpha = tf.keras.layers.Activation("sigmoid", name=f"{name}_sigmoid")(psi)
    # Broadcast attention map over all channels of skip
    return tf.keras.layers.Multiply(name=f"{name}_out")([skip, alpha])


def _se_block(x: tf.Tensor, ratio: int = 16, name: str = "se") -> tf.Tensor:
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    Globally pools each channel, learns a channel importance vector, and
    re-scales the feature map channel-wise.  Adds ~2% parameters.
    """
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = tf.keras.layers.Dense(max(filters // ratio, 1), activation="relu", use_bias=False, name=f"{name}_fc1")(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid", use_bias=False, name=f"{name}_fc2")(se)
    se = tf.keras.layers.Reshape((1, 1, filters), name=f"{name}_reshape")(se)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, se])


def _conv_block(
    x: tf.Tensor,
    filters: int,
    dropout: float = 0.0,
    name: str = "block",
    se: bool = False,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu2")(x)
    if se:
        x = _se_block(x, name=f"{name}_se")
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return x


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor | None,
    filters: int,
    dropout: float,
    name: str,
    attention: bool = False,
    se: bool = False,
) -> tf.Tensor:
    if skip is not None and attention:
        inter = max(filters // 2, 1)
        skip = _attention_gate(skip, gating=x, inter_filters=inter, name=f"{name}_attn")
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name=f"{name}_upsample")(x)
    if skip is not None:
        x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, skip])
    return _conv_block(x, filters=filters, dropout=dropout, name=name, se=se)


def build_unet_model(
    input_shape: tuple[int, int, int],
    backbone_name: str,
    pretrained: bool = True,
    decoder_filters: tuple[int, int, int, int, int] = (256, 128, 96, 64, 32),
    decoder_dropout: float = 0.1,
    attention: bool = False,
    se: bool = False,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone_key = backbone_name.lower()
    if backbone_key not in BACKBONES:
        supported = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported: {supported}")

    spec = BACKBONES[backbone_key]
    weights = "imagenet" if pretrained else None

    inputs = tf.keras.Input(shape=input_shape, name="s2_input")
    x = tf.keras.layers.Conv2D(
        3,
        kernel_size=1,
        padding="same",
        kernel_initializer=RGBProjectionInitializer(),
        bias_initializer="zeros",
        name="rgb_adapter",
    )(inputs)

    backbone = spec.builder(
        include_top=False,
        weights=weights,
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    encoder = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(name).output for name in spec.skip_names] + [backbone.output],
        name=f"{backbone_key}_encoder",
    )

    encoder_outputs = encoder(x)
    skips = encoder_outputs[:-1]
    x = encoder_outputs[-1]

    for idx, (filters, skip) in enumerate(zip(decoder_filters[:-1], reversed(skips))):
        x = _decoder_block(
            x,
            skip=skip,
            filters=filters,
            dropout=decoder_dropout,
            name=f"decoder_{idx + 1}",
            attention=attention,
            se=se,
        )

    x = _decoder_block(
        x,
        skip=None,
        filters=decoder_filters[-1],
        dropout=decoder_dropout,
        name="decoder_final",
        se=se,
    )
    outputs = tf.keras.layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="mask")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"s2_unet_{backbone_key}")
    return model, encoder
