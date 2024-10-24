# zephyr

![Work in Progress](https://img.shields.io/badge/work%20in%20progress-purple)
![Version 0.0.0a2](https://img.shields.io/badge/version-0.0.0a4-yellow)
![Early Stage](https://img.shields.io/badge/stage-early-yellow)

NOTE: Work in progress; enough to demonstrate the core feature; very early stage

Feature requests are very welcome, ask them in the Github Issues.

Currently working on: experimental features to help in model surgery


- [Summary](#summary) | [Core Principle](#core) 
- Examples: [Autoencoder](#autoencoder) | [Holes and Model Surgery](#FPTools) | [Linear](#linear) 
- [Motivation and Inspiration](#motivation) | [Installation](#installation)

## Summary<a id="summary"></a>
The [JAX](https://github.com/jax-ml/jax) library offers most things that you need for making neural networks, but there is no 
shortage of frameworks/libraries that build on JAX to cater to neural net building. 


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

Notice that we named `params` whenever it was passed to the encoder mlp: `params["mlp_1]` and `params["mlp_2"]`. 
These names are essential and is part of zephyr's design to allow maximum control over all parameters. 

Notice that an `mlp` is not some object, not some function passed to a transform, not a dataclass PyTree object, it is simply
a function `mlp(params, x, num_out_per_layer)`. There is no need to instatiate a model or a neural network. It's just a function! 
(Later we will show more reasons why this would be advantageous)

We have an autoencoder, now how do we instatiate the model? As said before, no instatiation needed. What we do need is a an initial
`params`. This is easy with the `trace` function.
```python
from zephyr.building.tracing import trace
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

### Tracing: trace function OR Tracer object
Aside from using the trace function, another way you could have initialized `params` is using a `Tracer` object which would need a `key: KeyArray` (like all random things in JAX). 
```python
from zephyr.building.tracing import Tracer
tracer = Tracer(key)
_dummy_outputs = autoencoder(tracer, x, embed_dim, latent_dim)
params = tracer.materialize() # this returns a working initialized params: PyTree
```

While it is longer than `trace`, it might be more intuitive for some.


For completeness, here is a side by side of it. (`trace` is doing the same thing: it uses the `Tracer`)
```python
# tracer object
tracer = Tracer(key)
_dummy_outputs = autoencoder(tracer, x, embed_dim, latent_dim)
params = tracer.materialize() 

# trace function
params = trace(autoencoder, key, x, embed_dim, latent_dim)
```

### Experimental: Holes (Placeholder and Derivable) <a id="FPTools"></a>
Zephyr introduces the concept of a placeholder hole and a derivable hole. Each is an object of type PlaceholderHole and DerivableHole, respecetively. However, we do not need multiple instances of them and so they would be named `_` and `__`, respectively. 

Holes make FP easier and eager-execution (op-by-op) mode easier. We'll use the example up above to show how holes can be useful.

Two decorators that zephyr has are `@hole_aware` and `@deriving_holes` (should be applied in this order). These make functions respond to placeholder holes and derivable holes.

```python
from zephyr.nets import mlp
from zephyr.functools.partal import hole_aware, deriving_holes, placeholder_hole as _, derivable_hole as __

@hole_aware
@deriving_holes
def encoder(params, x, embed_dim, latent_dim):
    x = mlp(params["mlp_1"], x, [embed_dim, embed_dim])
    x = mlp(params["mlp_2"], x, [embed_dim, latent_dim])
    return x

@hole_aware
@deriving_holes
def decoder(params, x, embed_dim, original_dim):
    x = mlp(params, x, [embed_dim, embed_dim, original_dim])
    return x

@hole_aware
@deriving_holes
def autoencoder(params, x, embed_dim, latent_dim):
    encoding = encoder(params["encoder"], x, embed_dim, latent_dim)
    reconstruction = decoder(params["decoder"], x, embed_dim, x.shape[-1])

    return reconstruction
```

`jit`, by default, can only trace functions with Array arguments and specifyin arg_nums/arg_names may be inconvient. One way to get around this is using python's `functools.partial` function and partially-apply the non-Array arguments (usually hyperparameters). However, `partial` **does not provide function signature hints** and makes it hard. So instead of using partial, we can use placeholder holes and it auto-partializes the function. Let's partially-apply the hyperparameters to autoencoder.

```python
model = autoencoder(_, _, embed_dim, latent_dim) # model can be called as model(params, x) <- notice the holes and the order


params = trace(model, key, x) # no hyperparameters needed since it's already known by the model

```

After tracing we can `jit` the model as `fast_model = jit(model)` and use it as normal : `fast_model(params, x)`.

**Model Surgery**. This is where derivable holes come into play. Do note, that derivable holes are only useful for not jitted functions as deriving them is not a jit-friendly operation. A while ago, we had to supply hyperparameters to the model's inner layers, but now, we can use a derivable hole instead (if the hole cannot be derived, an IncompleteDerivationError will be raised). 

```python
# assume you are done training and params contained trained weights (use another library like optax for this)

# what if you want to use just the encoder?
encodings = encoder(params["encoder"], x, __, __)

# what you want to use just the decoder?
some_reconstructions = decoder(params["decoder"], encodings, __, __)

# what if you want to just use the mlp_2 in encoder? you can do either of the following lines
mlp(params["encoder"]["mlp_2"], some_input, [__, __]) # provide a template of the shape (this will check if the mlp had 2 layers, and return an error if not) 
mlp(params["encoder"]["mlp_2"], some_input, __) #  make the whole output_dims argument a derivable hole
mlp(params["encoder"]["mlp_2"], some_input, [__, initial_dim]) # you can even use holes to check!, it will raise an error if it sees inconsistencies 
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
    z = params["weights"] @ z
    z = jnp.squeeze(z, axis=-1)

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
Warning: still in the **alpha** version. If you encounter bugs, please submit them to Issues and I'll try to fix them as soon as possible.

```bash
pip install z-zephyr
```
This version offers (cummulative, no particular order)
Major Features:
- parameter tracing and initialization with `zephyr.building.tracing.trace` (well developed, core feature)
- common layers and nets in `zephyr.nets` (unfinished)
- common initializers (kaiming and glorot)
- utilities for FP in python, jax and zephyr (useful, but not needed)
    - placeholders instead of partial application for more readability



