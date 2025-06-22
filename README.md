# zephyr

![Version 0.0.13](https://img.shields.io/badge/version-0.0.13-green)

Zephyr makes coding your machine learning ideas, short, fast, and to the point. Do a lot more while still writing less and still being more readable.

- fast: it is built on JAX
- easy: declarative syntax makes coding a lot shorter. If you know math, python, (jax) numpy, then you can write zephyr
- short: no boiler plate, focus on computations, not on initializing modules.
- precise: tags makes it possible to target groups of weights for nuanced update rule
- generic but still easier to write: use whatever you want, even (fixed size) recursions are welcome like f(params, x, i) = f(params, x-1, i+1)

## Overview

Zephyr, at the core, are just the `trace` and `validate` functions with extra utilities. `trace` gives you parameters. `validate` checks expressions related to parameters.

The main mindset in writing zephyr is to think in FP and declarative-manner. Think of composable transformations instead of methods - transformations of both
data or arrays AND functions. The examples below, will progressively re-write procedural/imperative-oriented code to the use of function transformations.
This puts the focus on what the transformation will be, rather than what the arrays become after each step.

Before we start. A neural network is just function, usually of `params`, `x`, and hyper-parameters. `f(params, x, **hyperparameters)`. If we wanted to get a function
without the hyperparameters, since those never change, we can use python's `partial` and rewrite as `new_f = partial(f, **hyperparameters)` and use `new_f(params, x)`. However, using `partial` could get tedious as it doesn't give you signature hints in your editor. Instead, you can use the more readable, zephyr's `_` notation which is an alias for `placeholder_hole` which zephyr nets accept and auto-partializes the function. So we could write `new_f = f(_,_, **hyperparameters)` where `_` stands in for values we pas in later. To make your own function accept `_` holes, you can use the `flexible` decorator.

One more thing, this library was heavily inspired by Haiku, and so `params` is a dictionary whose leaves are Arrays. Zephyr, uses the same convention.

## New Features not on the README yet

- tags : A way to update weights in a more precise manner. Example: update weights differently depending how deep they are in a layer; update different subnetworks differently; so on. (it's rare to this so I don't have an example here, but it is possible in zephyr)

## Installation

```bash
pip install z-zephyr --upgrade
```

## Contents

[Examples](#examples) | [Sharp Bits](#gotchas) | [Direction](#direction) | [Motivation](#motivation)

## Examples<a id="examples"></a>

Look at the Following Examples

0. [Imports](#imports): Common Gateway for Imports
1. [Encoder and Decoder](#ende): This example will show you some of the layers in `zephyr.nets`. We use zephyr's `chain` function to chain functions(neural networks) together.
2. [Parameter Creation](#parameters): This example will show you how to use custom parameters in your functions/nets.
3. [Dealing with random keys](#thread): This example will show you that keys are just Arrays and part of your input. Nevertheless, there are some zephyr utilities you could use
   to transform functions in ways that are useful for dealing with keys.

### Imports <a id="imports"></a>

These are the imports for all the examples

```python
from zephyr.functools.composition import thread_key, thread_params
from jax import numpy as jnp, random, jit, nn
from zephyr import nets, trace
from zephyr.nets import chain
from zephyr.functools.partial import placeholder_hole as _, flexible
```

### Example: Encoder and Decoder<a id="ende"></a>

Let's write a random encoder and decoder. Notice that we access `params` as if we already have a `params` made. Indeed, this declarative style is something you would have to get used to. Don't worry, zephyr handles making these parameters for you.

For each of the `encoder`, `decoder`, and `model` we offer 2 versions. One focusing on `x`, and the other building the transformation then applying it to `x`. These 2 versions are on the extreme, with the first being several lines of code, and the second being a single line of code(broken up). The next examples will use other rewrites that are less extreme.

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
        x = nets.max_pool(params, x, (3,3), 2)

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
                nets.max_pool(params, _, (3,3), 2),
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

### Examples: Making your own parameters<a id="parameters"></a>

To illustrate this, we will make our own `linear` layer using zephyr. In line with the declarative thinking, we specify what the shape of the paramters would look like -
Ideally, we can put this in the type annotation, but that's ignored by Python, so we instead use zephyr's `validate` as an alternative. One main use of `validate` is
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
key = random.PRNGKey(0)
model = linear(_,_, 256)
params = trace(model, key, x)
model(params, x) # use it like this

# or jit it
fast_model = jit(model)
fast_model(params, x)
```

### Dealing with random keys <a id="thread"></a>

Random keys or RNGs are somewhat an unfamiliar concept usually, since in FP you have to be explicit with these. So when you try to get rid of it using OO then
it tends to stick out like a sore thumb at the end. In zephyr, we embrace this and treat key as you would anything - it is just input to data.

Here a simple model using dropout.

```python
def model(params, x, key):
    for i in range(3):
        x = nets.mlp(params["mlp"][i], x, [256, 256])
        key, subkey = random.split(key)
        x = nets.dropout(params, subkey, x, 0.2)
    x = nn.sigmoid(x)
    return x
```

As with previous examples, we offer rewrites of this, none of which are "more elegent". Choose the one that best suits you.

Zephyr has a `thread` function with specific variants such as `thread_key`, `thread_params`, and `thread_identity` which should be enough for most cases.

Another rewrite would factor out the repeating block into its own function as follows.

```python
def block(params, key, x):
    return chain([
        nets.mlp(params["mlp"], _, [256,256]),
        nets.dropout(params, key, _, 0.2)
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

## Sharp Bits <a id="gotchas"></a>

1. Documentation Strings are sparse: I'll add them soon :3.
2. JAX Sharp Bits: You'll be dealing with JAX sharp bits sometimes like "str and int can't be compared" which is a jax thing, since Zephyr is such a thin library on top of JAX (it isn't even a thin wrapper). Any trouble you might have, you can open an issue and i'll help.
3. Bugs: If you use it, there'll probably be bugs, if you report them, I'll work on them immediately.
4. Missing nets: like RNNs, I'll add them soon when I need them or requested.
5. Instability: Things are still changing a lot. I might implement other nets/layers in a different way or change names or move things.

## Direction <a id="direction"></a>

I would like to provide more FP tooling for python in zephyr and so I could write zephyr nets in more FP-style. Zephyr itself, it's core, is probably close
to stable: mainly `trace` and `validate`, anything else is just to make coding easier or shorter.

## Motivation and Inspiration<a id="motivation"></a>

This library is heavily inspired by [Haiku](https://github.com/google-deepmind/dm-haiku)'s `transform` function which eventually
converts impure functions/class-method-calls into a pure function paired with an initilized `params` PyTree. This is my favorite
approach so far because it is closest to pure functional programming. Zephyr tries to push this to the simplest and make neural networks
simply just a function.

This library is also inspired by other frameworks I have tried in the past: Tensorflow and PyTorch. Tensorflow allows for shape
inference to happen after the first pass of inputs, PyTorch (before the Lazy Modules) need the input shapes at layer creation. Zephyr
wants to be as easy as possible and will strive to always use at-inference-time shape-inference and use relative axis positions whenever possible.
