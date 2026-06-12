# Fourier Continuation Gram Matrices

Matlab code to compute the Gram matrices required for Fourier Continuation with the FC-Gram algorithm. 

The matrices enable efficient extension of non-periodic functions to periodic ones for spectral differentiation.

## Example Usage

```matlab
FCGram_Matrices(4, 25, 12, 25, 20, 2, 256)
```

## Parameters

| Parameter | Type | Description | Typical Values |
|-----------|------|-------------|----------------|
| `d` | int | Number of matching points (degree of approximation) | 2-10 (faster convergence for higher d) |
| `C` | int | Number of continuation points (must be even) | ~25 |
| `Z` | int | Number of zero padding points for smooth extension | ~C/2 |
| `E` | int | Number of extra points for numerical stability | ~C |
| `n_over` | int | Oversampling factor for fine grid construction | >10 |
| `modes_to_reduce` | int | Number of modes to reduce in SVD truncation | 0-2 |
| `num_digits` | int | Number of digits for symbolic precision in computations | ≥128, recommended 256 |

### Parameter Details

- **`d`**: Typically varies between 2 and 10, with faster convergence for higher values
- **`C`**: Seemed to work well with values around 25
- **`Z`**: Works well when roughly equal to C/2
- **`E`**: Works well when roughly equal to C
- **`n_over`**: Should be greater than 10 for optimal results
- **`modes_to_reduce`**: Can be set to 0 (no impact) or 2 (removes smallest singular values for better approximation)
- **`num_digits`**: Should use at least 128, or 256 for higher precision

## Returns

- **`ArQr`**: Right boundary continuation matrix (C × d)
- **`AlQl`**: Left boundary continuation matrix (C × d)

## Algorithm

1. Construct monomial basis on coarse grid
2. Orthonormalize using QR decomposition with full re-orthogonalization
3. Evaluate basis on fine grid with oversampling
4. Build trigonometric basis for FC approximation
5. Compute SVD and solve for coefficients
6. Evaluate FC matrix at continuation points
7. Construct boundary continuation matrices with flip operations

## Notes

The function automatically saves the computed matrices to a `.mat` file in the format:
```
FCGram_data_d{d}_C{C}.mat
```

## References

[1] Amlani, F., & Bruno, O. P. (2016). An FC-based spectral solver for elastodynamic problems in general three-dimensional domains. *Journal of Computational Physics*, 307, 333-354.

## Authors

- **Daniel Leibovici** - [dleibovi@caltech.edu](mailto:dleibovi@caltech.edu)
- **Valentin Duruisseaux** - [vduruiss@caltech.edu](mailto:vduruiss@caltech.edu)
