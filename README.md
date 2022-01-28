# Installation

#### Install Firedrake
```
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
firedrake-install --doi 10.5281/zenodo.5762538
```

#### Activate Firedrake virtualenv
`source firedrake/bin/activate`

#### Install the Rapid Optimization Library
`pip install --no-cache-dir roltrilinos rol`

#### Install Fireshape
```
git clone git@github.com:fireshape/fireshape.git
cd fireshape
git checkout 2891d813f38ba69c6273ca312177deb5fdb0fe77
pip install -e .
```

# Run the code

Select one example of the paper in the folder FitzHugh_Nagumo, Ginzburg_Landau, Navier_Stokes, or Rayleigh_Benard and run
`python3 main.py`

# Reference
```
@article{boulle2022Optimal,
  title={Optimal control of Hopf bifurcations},
  author={Boull{\'e}, Nicolas and Farrell, Patrick E and Rognes, Marie E},
  journal={arXiv preprint arXiv:2201.11684},
  year={2022}
}
```
