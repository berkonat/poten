# poten

POTEN is a package for developing interatomic potentials. The package uses Torch7 project and is based on Lua. As of today, POTEN only supports atom-based Neural Network potentials and pair interactions.

## INN Potential usage with 'poten' and LuaTorPy.

See example notebook at [examples/poten_examples.ipynb](https://github.com/berkonat/poten/tree/master/examples/poten_examples.ipynb) .

<p align="left">
  <img src="https://github.com/berkonat/poten/blob/master/luatorpy-INN.png?raw=true" width="500" title="INN-LuaTorPy"></p>
  <img src="https://github.com/berkonat/poten/blob/master/Bulk-Mod.png?raw=true" width="500" title="INN-LuaTorPy">
</p>

## License

Please read the [LICENCE](LICENSE) file.

## Reference publication

Berk Onat, Ekin D. Cubuk, Brad D. Malone, and Efthimios Kaxiras, "Implanted neural network potentials: Application to Li-Si alloys", [Phys. Rev. B 97, 094106 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.094106)

## Installation

TORCHINSTALL="path/to/torch/install/dir/" ./compile.sh
