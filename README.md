# zephyr

![Work in Progress](https://img.shields.io/badge/work%20in%20progress-blue)
![Version 0.0.4](https://img.shields.io/badge/version-0.0.4-green)
![Early Stage](https://img.shields.io/badge/stage-early-yellow)

Zephyr is a new FP-oriented neural network library/framework on top of JAX that helps
you write short, simple, declarative neural networks quickly and easily with minimal
learning curve.

**For those coming from other frameworks**: The main difference is that zephyr is oriented towards writing in an FP-style.
No initialization of models is needed because models or nets are just regular functions - no need for separate init/build/construct
and a call/forward; just one call that includes everything.

**For new to deep learning**: As seen in many textbooks or materials, a neural network $M$ is a function that takes in parameters $\theta$, data or input $X$
and hyper-parameters $\alpha$ and produces an output $y$. Symbolically, it is $y = M(\theta, X, \alpha)$.

## Overview

The main mindset in writing zephyr is to think in FP and declarative-manner. Think of composable transformations instead of methods - transformations of both
data or arrays AND functions. The examples below, will progressively re-write procedural/imperative-oriented code to the use of function transformations.
This puts the focus on what the transformation will be, rather than what the arrays become after each step.

Before we start. A neural network is just function, usually of `params`, `x`, and hyper-parameters. `f(params, x, **hyperparameters)`. If we wanted to get a function
without the hyperparameters, since those never change, we can use python's `partial` and rewrite as `new_f = partial(f, **hyperparameters)` and use `new_f(params, x)`. However, using `partial` could get tedious as it doesn't give you signature hints in your editor. Instead, you can use the more readable, zephyr's `_` notation which is an alias for `placeholder_hole` which zephyr nets accept and auto-partializes the function. So we could write `new_f = f(_,_, **hyperparameters)` where `_` stands in for values we pas in later. To make your own function accept `_` holes, you can use the `flexible` decorator.

One more thing, this library was heavily inspired by Haiku, and so `params` is a dictionary whose leaves are Arrays. Zephyr, uses the same convention.

Look at the Following Examples

0. Imports: Common Gateway for Imports
1. Encoder and Decoder: This example will show you some of the layers in `zephyr.nets`. We use zephyr's `chain` function to chain functions(neural networks) together.
2. Parameter Creation: This example will show you how to use custom parameters in your functions/nets.
3. Dealing with random keys: This example will show you that keys are just Arrays and part of your input. Nevertheless, there are some zephyr utilities you could use
   to transform functions in ways that are useful for dealing with keys.

### Imports

```python
from jax import numpy as jnp, random, jit, nn
from zephyr import nets, trace
from zephyr.nets import chain
from zephyr.functools.partial import placeholder_hole as _, flexible
```

### Example: Encoder and Decoder

Let's write a random encoder and decoder. Notice that we access `params` as if we already have a `params` made. Indeed, this declarative style is something you would have to get used to. Don't worry, zephyr handles making these parameters for you.

For each of the `encoder`, `decoder`, and `model` we offer 2 versions. One focusing on `x`, and the other building the transformation then applying it to `x`.

Encoder: Notice that there neural networks are used just like normal functions. Within each use, we can explicitly see everything, the params, the input/s, and the hyperparameters. This makes code short and concise.

```python

@flexible
def encoder(params, x):
    x = nets.mlp(params["mlp"], x, [256,256,256]) # b 256
    x = nets.layer_norm(params["ln"], x, -1)
    x = nets.branch_linear(params["br"], x, 64) # b 64 256

    for i in range(3):
        x = nets.conv_1d(params["conv"][i], x, 64, 5)
        x = nn.relu(x)
        x = nets.max_pool(x, (3,3), 2)

    x = jnp.reshape(x, [x.shape[0], -1]) # b 256
    x = nets.linear(params["linear"], x, 4) # b 4
    return x


@flexible
def encoder(params, x):
    return chain([
        nets.mlp(params["mlp"], _, [256, 256, 256]),
        nets.layer_norm(params["ln"], _, -1),
        nets.branch_linear(params["br"], _, 64),
        * [
            chain([
                nets.conv_1d(params["conv"][i], _, 64, 5),
                nn.relu,
                nets.max_pool(_, (3,3), 2),
            ]) for i in range(3)
        ],
        lambda x: jnp.reshape(x, [x.shape[0], -1]),
        nets.linear(params["linear"], _, 4)
    ])(x)

```

Decoder: Notice that skip connections can be wrapped within a `skip` function/network that automatically adds a skip connection as `skip(f)(x) = f(x) + x`.

```python
@flexible
def decoder(params, z):
    x = nets.mlp(params["mlp"], x, [256,256,256]) # b 256
    x = nets.branch_linear(params["br"], x, 64) # b 64 256

    for i in range(3):
        x = nets.multi_head_self_attention(params["mha"][i], x, 64, 5)
        x = x + nets.mlp(params["attn_mlp"][i], x, [256, 256])
        x = nets.layer_norm(params["attn_ln"][i], x, -1)

    x = jnp.reshape(x, [x.shape[0], -1]) # b (64 * 128) = b 16384
    x = nets.linear(params["linear"], x, 128) # b 128
    return x

@flexible
def decoder(params, z):
    return chain([
        nets.mlp(params["mlp"], _, [256, 256, 256]),
        nets.branch_linear(params["br"], _, 64),
        *[
            chain([
                nets.multi_head_self_attention(params["mha"][i], _, 64, 5),
                nets.skip(nets.mlp(params["attn_mlp"][i], _, [256,256])),
                nets.layer_norm(params["attn_ln"][i], _, -1),
            ]) for i in range(3)
        ],
        lambda x: jnp.reshape(x, [x.shape[0], -1]),
        nets.linear(params["linear"], _, 128) # b 128
    ])(x)
```

Model:

```python
def model(params, x):
    z = encoder(params["encoder"], x)
    reconstructed_x = decoder(params["decoder"], z)
    return reconstructed_x

def model(params, x):
    return chain([
        encoder(params["encoder"], _),
        decoder(params["decoder"], _),
    ])(x)
```

To get an initial `params`, we simply use the trace function as follows.

```python
key = random.PRNGKey(0) # needed to randomly initialize weights
x = jnp.ones([64, 8]) # sample input batch:w


params = trace(model, key)

fast_model = jit(model) # tracing of `trace` cannot trace a jit-ed function, please use the non-jit-ed version when tracing
sample_outputs = fast_model(params, x) # b 8
```

For model surgery or study: if you wanted to use just the enoder, then you can do `z = encoder(params["encoder"], x)`. You can do the same with any function/layer.

### Examples: Making your own parameters

To illustrate this, we will make our own `linear` layer using zephyr. In line with the declarative thinking, we specify what the shape of the paramters would look like -
Ideally, we can put this in the type annotation, but that's ignored by Python, so we instead use zephyr's `valudate` as an alternative. One main use of `validate` is
to specify parameter shape, initializer, and other relationships it might have with hyperparameters.

```python
@flexible
def linear(params, x, out_target):
    validate(params["weights"], (x.shape[-1], out_target))
    validate(params["bias"], (out_target,))
    x = x @ params["weights"] + params["bias"]
    return x
```

As said, earlier we wil show rewrites which is up to you. This is just to show what is possible. There is a way to write this in way that resembles the pattern of
other FP languages where they assume some variables exist and give it to you with a `where` keyword, similar to math statements.

```python
@flexible
def linear(params, x, out_target):
    return (lambda w, b: x @ w + b)(
        validate(params["weights"], (x.shape[-1], out_target)),
        validate(params["bias"], (out_target,)),

    )
```

Notice the use of `validate` here. `validate` is actually just a way to enfore "type annotations" (albeit dependent types because we're really specifying shapes)
because they have to be specified somewhere for zephyr to trace it. Nevertheless, `validate` acts like the identity function and returns its first parameter unchanged.

To use it, we simply use the `trace` function and use normally as follows.

```python
model = linear(_,_, 256)
params = trace(model, key, x)
model(params, x) # use it like this

# or jit it
fast_model = jit(model)
fast_model(params, x)
```

### Dealing with random keys

Random keys or RNGs are somewhat an unfamiliar concept usually, since in FP you have to be explicit with these. So when you try to get rid of it using OO then
it tends to stick out like a sore thumb at the end. In zephyr, we embrace this and treat key as you would anything - it is just input to data.

Here a simple model using dropout.

```python
def model(params, x, key):
    for i in range(3):
        x = nets.mlp(params["mlp"][i], x, [256, 256])
        key, subkey = random.split(key)
        x = nets.dropout(subkey, x, 0.2)
    x = nn.sigmoid(x)
    return x
```

As with previous examples, we offer rewrites of this, none of which are "more elegent". Choose the one that best suits you.

Zephyr has a `thread` function with specific variants such as `thread_key`, `thread_params`, and `thread_identity` which should be enough for most cases.

Here is the same model but using the `thread_key` function to "thread" the `key` to multiple `dropouts` (this could be any function with a key as a first parameter).

```python
def model(params, x, key):
    dropouts = thread_key([nets.dropout(_, _, 0.2) for i in range(3)], key) # each dropout is now dropout(x), the 1st hole is filled by the threaded key
    for i in range(3):
        x = nets.mlp(params["mlp"][i], x, [256, 256])
        x = dropouts[i](x)
    x = nn.sigmoid(x)
    return x
```

Another rewrite would factor out the repeating block into its own function as follows.

```python
def block(params, key, x):
    return chain([
        nets.mlp(params["mlp"], _, [256,256]),
        nets.dropout(key, _, 0.2)
    ])(x)

def model(params, x, key):
    blocks = thread_params([block for i in range(3)], params) # each block is block(key,x)
    blocks = thread_key(blocks, key) # each block is block(x)

    return chain(blocks + [nn.relu])(x)

```

To use it, we simply use the `trace` function and use normally as follows.

```python
trace_key, apply_key_1, apply_key_2, key = random.split(key, 4) # split the keys ;p

params = trace(model, trace_key, x, apply_key_1)
model(params, x, apply_key_2) # use it like this

# or jit it
fast_model = jit(model)
fast_model(params, x, apply_key_2)
```

# Ignore below, readme under construction

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
