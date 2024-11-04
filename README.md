# zephyr

![Work in Progress](https://img.shields.io/badge/work%20in%20progress-blue)
![Version 0.0.4](https://img.shields.io/badge/version-0.0.4-green)
![Early Stage](https://img.shields.io/badge/stage-early-yellow)

Zephyr is a new FP-oriented neural network library/framework on top of JAX that helps
you write short, simple, declarative neural networks quickly and easily with minimal
learning curve.

**For those coming from other frameworks**: The main difference is that this oriented towards writing in an FP-style.
No initialization of models is needed because models or nets are just regular functions - no need for separate init/build/construct
and a call/forward; just one call that includes everything.

**For new to deep learning**: As seen in many textbooks or materials, a neural network $M$ is a function that takes in parameters $\theta$, data or input $X$
and hyper-parameters $\alpha$ and produces an output $y$. Symbolically, it is $y = M(\theta, X, \alpha)$.

## Overview

This overview tries to cover everything briefly one example. Before that, a few things.

- `params`: The parameters of a model is contained in a dictionary PyTree with Array leaves patterned after Haiku
- `_`: is an alias for a placeholder*holder which is a zephyr construct to make partializing a function easier. A `*` hole simply leaves that parameter empty and is passed later. (Same as python partial in functools)

When writing neural nets with zephyr, you should think declaratively - you just assume that you already have the `params` built-in and you can access with any key you like (e.g. `params["some_name"]`.

Here is a random model with comments to show the shapes of $x$ as it passes throught the layers (it is ommited when obvious).

```python
from zephyr import nets
from zephyr.functools.partial import placeholder_hole as _
from jax import numpy as jnp

def model(params, x):
    x = nets.mlp(params["mlp"][0], x, [256, 256]) # b 256
    x nn.relu(x) # mlp does not activate the last layer by default
    x = nets.mlp(params["mlp"][1], x, [256, 256])
    x = nets.relu(x)
    x = nets.layer_norm(params["ln"][0], x, -1)
    x = nets.mlp(params["mlp"][2], x, [256, 256]) # b 256
    x = nets.branch_linear(["br"][0], x, 64) # b 64 256
    x = nets.conv_1d(params["conv1d"][0], x, 64, 5) # b 64 64
    x = nn.relu(x)
    x = nets.layer_norm(params["ln"][1], x, -1)
    x = nets.conv_1d(params["conv1d"][1], x, 64, 5) # b 64 64
    x = nn.relu(x)
    x = nets.max_pool(x, (3,3), 2) # b 32 32
    x = jnp.reshape(x, [x.shape[0], -1]) # b 1024
    x = nets.layer_norm(params["ln"][2], x, -1)
    x = nets.branch_linear(params["br"][1], x, 64) # b 64 1024
    for i in range (3):
        x = nets.multi_head_self_attention(params["mha"][i], x, 8) # b 64 1024
        x = nets.mlp(params["attn_mlp"][i], x, [64, x.shape[-1]]) # b 64 1024
        x = nets.layer_norm(params["attn_ln"][i], x, -1)
    x = jnp.reshape(x, [x.shape[0], -1]) # b (64 * 1024) = b 65536
    x = nets.linear(params["mlp_final"], x, 8) # b 8
    x = nn.sigmoid(x) # b 8
    return x
```

Of course, this model isn't something you would use, but it shows most of zephyr. Like, with other frameworks, a simple step-by-step process like this can be gathered
with a `chain` with zephyr as follows. Notice the use of `_`: placeholder_hole which stands in for `x` until it is passed. `chain` is used like this `chain(sequence_of_layers)(x)`. Below is how we would re-write the previous model.

```python
from zephyr.functools.partial import placeholder_hole as _
from zephyr.nets import chain
from jax import numpy as jnp

def model(params, x):
    return chain([
        nets.mlp(params["mlp"][0], _, [256,256]), nn.relu,
        nets.mlp(params["mlp"][1], _, [256,256]), nn.relu,
        nets.layer_norm(params["ln"][0], _, -1),
        nets.mlp(params["mlp"][2], _, [256,256]), nn.relu,
        nets.branch_linear(params["br"][0], _, 64),
        nets.conv_1d(params["conv1d"][0], _, 64, 5), nn.relu,
        nets.layer_norm(params["ln"][1], _, -1),
        nets.conv_1d(params["conv1d"][1], _, 64, 5), nn.relu,
        nets.max_pool(_, (3,3), 2),
        lambda x: jnp.reshape(x, [x.shape[0], -1]),
        nets.layer_norm(params["ln"][2], _, -1),
        nets.branch_linear(params["br"][1], _, 64),
        *[
            chain([nets.multi_head_self_attention(params["mha"][i], _, 8),
            lambda x: nets.mlp(params["attn_mlp"][i], x, [64, x.shape[-1]]), nn.relu,
            nets.layer_norm(params["attn_ln"][i], _, -1)])
            for i in range(3)

        ],
        lambda x: jnp.reshape(x, [x.shape[0], -1]),
        nets.linear(params["mlp_final"], _, 8),
        nn.sigmoid,
    ])(x)
```

To use the model, is simple, we construct the `params` which is easy! We use the `trace` from zephyr. We use the model just by a normal
function call `model(params, x)`.

```python
from zephyr import trace
from jax import random, jit

key = random.PRNGKey(0) # needed to randomly initialize weights
x = jnp.ones([64, 8]) # sample input batch:w


params = trace(model, key)

fast_model = jit(model) # tracing of `trace` cannot trace a jit-ed function, please use the non-jit-ed version when tracing
sample_outputs = fast_model(params, x) # b 8
```

Ignore below, readme under construction

## Current Gotchas

1. Documentation Strings are sparse
2. JIT Gotchas
3. Bugs

> New: **Common networks and layers** such as Linear, MLP, Convolution, Attention, etc. are here.

NOTE: Work in progress; Feature requests are very welcome;

Currently working on: core networks and layers

- [Summary](#summary) | [Core Principle](#core)
- Examples: [Autoencoder](#autoencoder) | [Chaining](#thread) | [Linear](#linear)
- [Motivation and Inspiration](#motivation) | [Installation](#installation)

## Summary<a id="summary"></a>

The [JAX](https://github.com/jax-ml/jax) library offers most things that you need for making neural networks, but most frameworks are not suitable for FP-oriented coding (meaning: these frameworks will use FP for jax behind the scenes, but you'll be writing writing classes)

zephyr focuses on 2 things:

- **Parameter Creation**. The number one pain point for using jax-numpy for neural networks is the difficulty of the laborious and tedious process of creating the parameters
- **Simplicity**. Neural networks are pure functions, but none of the frameworks present neural network as such pure functions. They always treat a neural network as something extra which is why you would need some special methods or transforms or re-duplicated jax methods.

## Core Principle<a id="core"></a>

A neural network $f$ is simply mathematical function of data $X$, parameters $\theta$, and hyper-parameters $\alpha$. We place $\theta$ as the first parameter of $f$ because `jax.grad` creates the gradient of $f$ wrt to the first parameter by default.

$$ f(\theta, X, \alpha) $$

## Examples

Here are two examples to demonstrate and highlight what zephyr empowers: simplicity, and control.

### Making an autoencoder<a id="autoencoder"></a>

Let's make a simple autoencoder. The encoder will use 2 mlp's in succession and the decoder will use just 1.

```python
from zephyr.nets import mlp
def encoder(params, x, embed_dim, latent_dim):
    x = mlp(params["mlp_1"], x, [embed_dim, embed_dim])
    x = mlp(params["mlp_2"], x, [embed_dim, latent_dim])
    return x

def decoder(params, x, embed_dim, original_dim):
    x = mlp(params, x, [embed_dim, embed_dim, original_dim])
    return x

def autoencoder(params, x, embed_dim, latent_dim):
    encoding = encoder(params["encoder"], x, embed_dim, latent_dim)
    reconstruction = decoder(params["decoder"], x, embed_dim, x.shape[-1])

    return reconstruction
```

Notice that we named `params` whenever it was passed to the encoder mlp: `params["mlp_1"]` and `params["mlp_2"]`.
These names are essential and is part of zephyr's design to allow maximum control over all parameters.

Notice that an `mlp` is not some object, not some function passed to a transform, not a dataclass PyTree object, it is simply
a function `mlp(params, x, num_out_per_layer)`. There is no need to instatiate a model or a neural network. It's just a function!
(Later we will show more reasons why this would be advantageous)

We have an autoencoder, now how do we instatiate the model? As said before, no instatiation needed. What we do need is a an initial
`params`. This is easy with the `trace` function.

```python
from zephyr import trace
from jax import random

batch_size = 8
initial_dim = 64
latent_dim = 256
embed_dim = 512

key = random.PRNGKey(0)
x = jnp.ones([batch_size, initial_dim]) # this is a sample input

params = trace(autoencoder, key, x, embed_dim, latent_dim)

"""
params = {
    encoder: {
        mlp_1: {weights: ..., biases: ...},
        mlp_2: {weight: ..., biases: ...}
    },
    decoder: {
        weights: ...,
        biases: ...
    }
}
"""
```

Notice how each of the entries in `params` were appropriately named. This would be automatic in some frameworks, but having explicit names
allows us to take apart the model with ease as we will see below.

```python
# assume you are done training and params contained trained weights (use another library like optax for this)

# what if you want to use just the encoder?
encodings = encoder(params["encoder"], x, embed_dim, latent_dim)

# what you want to use just the decoder?
some_reconstructions = decoder(params["decoder"], encodings, embed_dim, x.shape[-1])

# what if you want to just use the mlp_2 in encoder?
mlp(params["encoder"]["mlp_2"], some_input, [embed_dim, latent_dim])
```

As you can see, by being on the jax-level all the time, you are free to do whatever you want. Coding becomes short and to the point.

### Threading and Chaining <a id="thread"></a>

Threading does not refer to the multi-threading of parallelization, but a metaphor for passing an argument through several function but on each function, the argument is split into 2 - one is passed to the current function and the other one goes through.

Threading is particularly useful if you have several functions `f_1, f_2, ..., f_n` with the same first argument like `params` or a `key` but each pass should be different. For example, a key should be split before each pass to a function. This is usually tedious, and after a while can be boring. Threading the key through the functions gives you functions without the argument (partially applied) and with the split already done.

Zephyr offers the generic `thread` function, but also offers the specific `thread_params`, `thread_key`, and `thread_identity` to thread an argument through a sequence of functions. To show this will compare two examples, one normal, and another with threading.

Additionally, the `chain` function is very useful for functions of the form `x_like = f(x)` similar to Sequential models. The chain function takes a sequence of functions as input and pairs nicely with threading.

Let's implement a model with blocks of mlp with dropout normally.

```python
from zephyr.nets import dropout
from functools import partial

def mlp_with_dropout(params, key, x, out_dims, drop_prob):
    return dropout(key, mlp(params, x, out_dims), drop_prob)

def model(params, key, x, out_dims, dp, num_blocks):
    validate(params, expression=lambda params: len(params) == num_blocks)

    for i in range(num_blocks):
        key, subkey = random.split(key)
        x = mlp_with_dropout(params[i], subkey, x, out_dims, dp)
    return x

```

With threading and chaining:

```python
def model(params, key, x, out_dims, dp, num_blocks):
    validate(params, expression=lambda params: len(params) == num_blocks)
    blocks = [ partial( mlp_with_dropout, out_dims=out_dims, dp=dp ) for i range(num_blocks) ]
    return chain(thread_key(thread_params(blocks, params), key) (x)
```

### Building Layers From Scratch<a id="linear"></a>

Usually it is rare that one would need to instantiate their own trainable weights (specifying the shape and initializer) since Linear / MLP layers usually suffice for that. Frameworks usually differ in how to handle parameter building and it is part of what makes the core
experience in these frameworks. This part is also where clever things in each framework is hidden. For zephyr, it wanted to keep
functions pure, but parameter building is hard, so that's what zephyr makes it easy.

Let's implement the linear layer from scratch. A linear layer would need `weights` and `biases`. We assume that we already have formed `params` and we just have to
validate to ensure that 1) it exists and 2) it is of the right shape (also an initializer can be supplied so that the tracer takes note if you use the tracer/trace).
If you try to handcraft your own params, instead of using the `trace` function, this validate will tell you if there is a mismatch with what you created and what it expected.

```python
from zephyr.building.initializers import initializer_base, Initializer
from zephyr.building.template import validate

def linear(
    params: PyTree,
    x: Array,
    target_out: int,
    with_bias: bool = True,
    initializer: Initializer=initializer_base,
) -> Array:
    validate(params["weights"], shape=(target_out, x.shape[-1]), initializer=initializer)
    z = jnp.expand_dims(x, axis=-1)
    z =  z @ params["weights"]

    if with_bias:
        validate(params["bias"], shape=(target_out,), initializer=initializer)
        z = params["bias"] + z

    return z
```

As a rule of thumb, if you're going to manipulate a params or do arithmetic with it (eg. `jnp.transpose(params)` or `params + 2`), then validate it before those operations (you only need to validate it once).

And as seen, earlier, to use this, just use the `trace` function.

```python
from jax import numpy as jnp, random

key = random.PRNGKey(0)
dummy_inputs = jnp.ones([64, 8])
params = trace(linear, key, dummy_inputs, 128)

sample_outputs = linear(params, dummy_inputs, 128) # shape: [64, 128]
```

## Motivation and Inspiration<a id="motivation"></a>

This library is heavily inspired by [Haiku](https://github.com/google-deepmind/dm-haiku)'s `transform` function which eventually
converts impure functions/class-method-calls into a pure function paired with an initilized `params` PyTree. This is my favorite
approach so far because it is closest to pure functional programming. Zephyr tries to push this to the simplest and make neural networks
simply just a function.

This library is also inspired by other frameworks I have tried in the past: Tensorflow and PyTorch. Tensorflow allows for shape
inference to happen after the first pass of inputs, PyTorch (before the Lazy Modules) need the input shapes at layer creation. Zephyr
wants to be as easy as possible and will strive to always use at-inference-time shape-inference and use relative axis positions whenever possible.

## Installation<a id="installation"></a>

```bash
pip install z-zephyr --upgrade
```

This version offers (cummulative, no particular order)
Major Features:

- parameter tracing and initialization with `zephyr.building.tracing.trace` (well developed, core feature)
- common layers and nets in `zephyr.nets` (unfinished)
- common initializers (kaiming and glorot)
- utilities for FP in python, jax and zephyr (useful, but not needed)
  - placeholders instead of partial application for more readability

### Final Note

If you encounter bugs, please submit them to Issues and I'll try to fix them as soon as possible.
