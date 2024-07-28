#!/bin/bash
conda init

# Activate the "rl-portfolio-tool" environment
conda activate rl-portfolio-tool


# if numpy version is not 1.24.3, do not proceed, it is important that the numpy version matches the one used in the colab
if [ "$(python -c "import numpy; print(numpy.__version__)")" != "1.24.3" ]; then
    echo "Please install numpy version 1.24.3"
    exit 1
fi

# Run the create_universe.py script
python create_universe.py

# Run the macro_economic_factors.py script
python macro_economic_factors.py