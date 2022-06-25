from jax import grad
import jax.numpy as jnp

def fun(x1,x2,x3,x4):
    a = jnp.array([[x2, x1],[x3,x4]])
    a = jnp.linalg.inv(a)
    return a[0][0]

res = grad(fun)(1.0,2.0,3.0,4.0)

print(res)