# Einsum

Einsum is a C++ based Python library designed for efficient tensor contraction, employing the Einstein summation convention.

# Install

To install the Einsum library, navigate to the folder containing the `setup.py` file and run the following command:

```bash
python setup.py build_ext --inplace
```

This command compiles the source and generates the shared object file `einsum(version).so`, making it ready for use within your Python environment.

# Usage

To utilize the Einsum library in your projects, simply import it into your Python script as follows:

```python
import einsum
einsum.c_einsum('ijab,kjca->ikbc',A,B)
```

The library accepts numpy ndarray objects and a string that specifies the subscript for summation as input, mirroring the usage of NumPy's own `einsum` function. With this, you can perform tensor contractions in the same intuitive manner as NumPy's einsum, benefiting from the efficiency and speed of a C++ backend.

