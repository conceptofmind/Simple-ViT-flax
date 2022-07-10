# Simple-ViT-flax

"It is commonly accepted that the Vision Transformer model requires sophisticated regularization techniques to excel at ImageNet-1k scale data. Surprisingly, we find this is not the case and standard data augmentation is sufficient. This note presents a few minor modifications to the original Vision Transformer (ViT) vanilla training setting that dramatically improve the performance of plain ViT models. Notably, 90 epochs of training surpass 76% top-1 accuracy in under seven hours on a TPUv3-8, similar to the classic ResNet50 baseline, and 300 epochs of training reach 80% in less than one day." - Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov

An update from some of the same authors of the original paper proposes simplifications to ViT that allows it to train faster and better.

Among these simplifications include 2d sinusoidal positional embedding, global average pooling (no CLS token), no dropout, batch sizes of 1024 rather than 4096, and use of RandAugment and MixUp augmentations. They also show that a simple linear at the end is not significantly worse than the original MLP head.

## Acknowledgement:
I have been greatly inspired by the work of [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

## Usage
```python
import numpy as np

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (1, 3, 256, 256))

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
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

## Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

## Citation:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2205.01580,
  doi = {10.48550/ARXIV.2205.01580},
  
  url = {https://arxiv.org/abs/2205.01580},
  
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Better plain ViT baselines for ImageNet-1k},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```