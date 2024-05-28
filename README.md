# hwak_cuda
a hasegawa wakatani solver using cupy

![hwak](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3d5bjA5eXR2dTRoODZxN2FoajZucGVicGFrazl4bWd1bHl6Y3R5cCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/7iNNeQLJJ1PFCZNkm5/giphy-downsized-large.gif)

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

## Running

you can start by making a copy of the `run` directory:

```bash
cd hwak_cuda
cp -r run testrun
```

you can now edit `testrun/run.py` as you like. The hasegawa_wakatani class initializer accepts keyword parameters from `default_parameters`, `default_solver_parameters` and `default_controls` that you can find in the file `hwak_cuda.py`.

You can run it using:

```
cd testrun
pyton run.py
```
note that `run.py` assumes that it will be able to find hwak_cuda.py at its parent directory, you can change this accordingly if you want to move your run.py files elsewhere.

you can also find an `anim.py` in the `run` directory. This is given as an example of how to read the output data. It takes the file `out.h5` and generates `out.mp4`, and you can run it in parallel using something like:

```
mpirun -np 4 python anim.py
```
Note that if you change the output file name in `run.py`, you need to change it in `anim.py` as well.

A basic run with a 1024x1024 padded resolution up to t=500, should take 1-2 hours on a normal gpu. Though it depends on the parameters, with large `C` values not saturating at t=500, and small `C` values taking a bit more time to run as they reach steady state more rapidly.
