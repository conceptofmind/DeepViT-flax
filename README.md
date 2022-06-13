## DeepViT-flax

This <a href="https://arxiv.org/abs/2103.11886">paper</a> notes that ViT struggles to attend at greater depths (past 12 layers), and suggests mixing the attention of each head post-softmax as a solution, dubbed Re-attention. The results line up with the <a href="https://github.com/lucidrains/x-transformers#talking-heads-attention">Talking Heads</a> paper from NLP.

## Install

```bash
$ pip install deepvit-flax
```

## Usage

```python
import flax
import numpy as np
from deepvit_flax import DeepViT

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 256, 256, 3))

v = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")
```

## Acknowledgements:
- [Dr. Phil Wang](https://github.com/lucidrains/)

## Todo

- [x] Build model
- [x] Implement pip installer
- [ ] Implement huggingface streaming dataloaders
- [ ] Implement training script
- [ ] Add config for training

## Author:
- Enrico Shippole

## Citations
```bibtex
@article{zhou2021deepvit,
  title={DeepViT: Towards Deeper Vision Transformer},
  author={Zhou, Daquan and Kang, Bingyi and Jin, Xiaojie and Yang, Linjie and Lian, Xiaochen and Hou, Qibin and Feng, Jiashi},
  journal={arXiv preprint arXiv:2103.11886},
  year={2021}
}
```

```bibtex
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
```
