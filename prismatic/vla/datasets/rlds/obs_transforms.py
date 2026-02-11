"""
obs_transforms.py

Contains observation-level transforms used in the orca data pipeline.

These transforms operate on the "observation" dictionary, and are applied at a per-frame level.
"""

from typing import Dict, Tuple, Union

import dlimp as dl
import tensorflow as tf
from absl import logging


# ruff: noqa: B023
def augment(obs: Dict, seed: tf.Tensor, augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images.
    Works whether pad masks are scalar (single frame) or vector (batched/time).
    """
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    # Make seed a scalar int32 base; dlimp typically expects stateless seeds shaped [2]
    seed = tf.cast(seed, tf.int32)
    if tf.rank(seed) != 0:
        # If someone passes [2] or [B], collapse deterministically to a scalar
        seed = tf.reduce_sum(seed)

    for i, name in enumerate(sorted(image_names)):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        img_key = f"image_{name}"

        img = obs[img_key]
        pm = obs["pad_mask_dict"][img_key]  # "True" means real (non-padding), per your original logic
        pm = tf.cast(pm, tf.bool)

        # Create a stateless seed of shape [2] for this image stream
        seed2 = tf.stack([seed, tf.cast(i, tf.int32)], axis=0)

        aug_img = dl.transforms.augment_image(img, **kwargs, seed=seed2)

        # If pm is scalar, behave like the old tf.cond
        def _scalar_case():
            return tf.cond(pm, lambda: aug_img, lambda: img)

        # If pm is vector (e.g., [T] or [T,1]), apply elementwise selection
        def _vector_case():
            # Squeeze [T,1] -> [T]
            if tf.rank(pm) == 2 and tf.shape(pm)[-1] == 1:
                pm2 = tf.squeeze(pm, axis=-1)
            else:
                pm2 = pm

            # Broadcast mask to image rank: [T] -> [T,1,1,1] (or whatever image rank is)
            broadcast_shape = tf.concat([tf.shape(pm2), tf.ones([tf.rank(img) - tf.rank(pm2)], tf.int32)], axis=0)
            pm_b = tf.reshape(pm2, broadcast_shape)

            return tf.where(pm_b, aug_img, img)

        obs[img_key] = tf.cond(tf.equal(tf.rank(pm), 0), _scalar_case, _vector_case)

    return obs
