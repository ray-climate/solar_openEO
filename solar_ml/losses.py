"""Losses and metrics for solar segmentation."""

from __future__ import annotations

from functools import partial

import tensorflow as tf


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denominator = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(score)


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


def binary_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
    return tf.reduce_mean(loss)


def binary_crossentropy_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def tversky_index(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> tf.Tensor:
    """Tversky index: generalises Dice. alpha=FP weight, beta=FN weight.
    alpha=beta=0.5 → Dice. alpha<beta → recall-focused."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    tp = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f, axis=1)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f), axis=1)
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return tf.reduce_mean(score)


def _tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    alpha: tf.Tensor,
    beta: tf.Tensor,
) -> tf.Tensor:
    return 1.0 - tversky_index(y_true, y_pred, alpha=float(alpha), beta=float(beta))


def _focal_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    """Focal Tversky loss (Abraham & Khan 2019). gamma in [0.5, 1.0] typical."""
    ti = tversky_index(y_true, y_pred, alpha=float(alpha), beta=float(beta))
    return tf.pow(1.0 - ti, float(gamma))


def _bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, *, bce_weight: tf.Tensor, dice_weight: tf.Tensor) -> tf.Tensor:
    bce_term = binary_crossentropy_loss(y_true, y_pred)
    dice_term = tf.cast(dice_loss(y_true, y_pred), tf.float32)
    return bce_weight * bce_term + dice_weight * dice_term


def _focal_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    focal_weight: tf.Tensor,
    dice_weight: tf.Tensor,
) -> tf.Tensor:
    focal_term = tf.cast(binary_focal_loss(y_true, y_pred), tf.float32)
    dice_term = tf.cast(dice_loss(y_true, y_pred), tf.float32)
    return focal_weight * focal_term + dice_weight * dice_term


def get_loss(
    loss_name: str,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_tversky_gamma: float = 0.75,
):
    loss_name = loss_name.lower()
    bce_weight_t = tf.constant(bce_weight, dtype=tf.float32)
    dice_weight_t = tf.constant(dice_weight, dtype=tf.float32)
    focal_weight_t = tf.constant(focal_weight, dtype=tf.float32)
    alpha_t = tf.constant(tversky_alpha, dtype=tf.float32)
    beta_t = tf.constant(tversky_beta, dtype=tf.float32)
    gamma_t = tf.constant(focal_tversky_gamma, dtype=tf.float32)

    if loss_name == "bce_dice":
        loss_fn = partial(_bce_dice_loss, bce_weight=bce_weight_t, dice_weight=dice_weight_t)
        loss_fn.__name__ = "bce_dice"
        return loss_fn
    if loss_name == "focal_dice":
        loss_fn = partial(_focal_dice_loss, focal_weight=focal_weight_t, dice_weight=dice_weight_t)
        loss_fn.__name__ = "focal_dice"
        return loss_fn
    if loss_name == "dice":
        return dice_loss
    if loss_name == "tversky":
        loss_fn = partial(_tversky_loss, alpha=alpha_t, beta=beta_t)
        loss_fn.__name__ = "tversky"
        return loss_fn
    if loss_name == "focal_tversky":
        loss_fn = partial(_focal_tversky_loss, alpha=alpha_t, beta=beta_t, gamma=gamma_t)
        loss_fn.__name__ = "focal_tversky"
        return loss_fn
    raise ValueError(f"Unsupported loss: {loss_name}")
