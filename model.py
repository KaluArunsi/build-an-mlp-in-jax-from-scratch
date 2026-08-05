"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    subkeys = jax.random.split(key, num=num)
    return subkeys

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    normal_sample = jax.random.normal(key, shape=shape)
    return normal_sample

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    shape = (batch_size, num_features)
    sample_matrix = jax.random.normal(key, shape)
    return sample_matrix

# Step 5 - assign_class_labels
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    # TODO: return an int32 label per row using the first num_classes feature columns.
    selection = inputs[:, :num_classes]
    assigned_labels = jnp.argmax(selection, axis=1).astype(jnp.int32)
    return assigned_labels

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # TODO: Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    labels = labels[:, None]
    class_index = jnp.arange(num_classes)
    one_hot = (labels == class_index).astype(jnp.float32)
    return one_hot

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    W = sample_normal_matrix(key, (in_dim, out_dim))
    W = W * scale
    b = jnp.zeros(out_dim)

    return {
        'W': W,
        'b': b
    }

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    subkeys = split_prng_key(key, num = len(layer_sizes) - 1)
    params = []

    for i, k in enumerate(subkeys):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i+1]

        params.append(init_linear_layer(k, in_dim, out_dim, scale=scale))
    
    return params

# Step 9 - linear_forward
import jax
import jax.numpy as jnp

def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    xw = x @ layer_params['W'] 
    y = xw + layer_params['b']

    return y

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    activation = jnp.maximum(x, 0)
    return activation

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

