# hwak_cuda
a hasegawa wakatani solver using cupy

## Install

### using git:
The default version can be installed in a directory `dest_dir`

```bash
cd dest_dir
git clone git@github.com:gurcani/hwak_cuda.git
cd hwak_cuda
git submodule init
git submodule update
```
note that there are other branches, like the `cpu` or `cpu_gpu`, which may not need submodules but may require compilation depending on the choice of solvers. It seems that a `vode` solver with the `cpu_gpu` branch is quite fast if you have a normal gpu, whereas if you have a performant gpu, the best choice is the default branch.
