import torch
import numpy as np
from petsc4py import PETSc


def convert_petsc_mat_to_torch_sparse_coo_tensor(A: PETSc.Mat, dtype=torch.float32, eps: float = 0.0) -> torch.Tensor:
    """
    Convert a PETSc.Mat to torch.sparse_coo_tensor.

    Parameters:
        A (PETSc.Mat): PETSc matrix.
        dtype (torch.dtype): Target PyTorch dtype.
        eps (float): Threshold to filter out small entries.

    Returns:
        torch.sparse_coo_tensor
    """
    # Ensure the matrix is assembled
    A.assemble()

    # Extract CSR representation
    ai, aj, av = A.getValuesCSR()
    
    # Get number of rows and columns
    num_rows, num_cols = A.getSize()

    # Convert CSR to COO format
    rows = np.repeat(np.arange(len(ai) - 1), np.diff(ai))
    cols = aj
    values = av

    # Filter out near-zero entries
    mask = np.abs(values) > eps
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]

    # Convert to PyTorch tensor
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(values, dtype=dtype)

    return torch.sparse_coo_tensor(indices, values, size=(num_rows, num_cols), dtype=dtype)

# def convert_petsc_matrix_to_torch_sparse_coo_tensor(
#     A: PETSc.Mat,
#     dtype=torch.float32,
#     eps: float = 0.0,
#     separate_real_imag: bool = False
# ):
#     """
#     Converts a PETSc matrix (including MatNest) to a PyTorch sparse COO tensor.
#     Supports optional real/imaginary part separation.

#     Returns:
#         - Single complex-valued sparse tensor, or
#         - Tuple (real_part, imag_part) if `separate_real_imag=True`
#     """
#     if not isinstance(A, PETSc.Mat):
#         raise TypeError("Input A must be of type petsc4py.PETSc.Mat")

#     def process_block(A_block, row_offset=0, col_offset=0):
#         A_block.assemble()
#         ai, aj, av = A_block.getValuesCSR()

#         rows = np.repeat(np.arange(len(ai) - 1), np.diff(ai)) + row_offset
#         cols = aj + col_offset
#         values = av

#         mask = np.abs(values) > eps
#         return rows[mask], cols[mask], values[mask]

#     def flatten_matnest(A_nest: PETSc.Mat):
#         row_iss, col_iss = A_nest.getNestISs()
#         nblocks_row = len(row_iss)
#         nblocks_col = len(col_iss)

#         # Compute row and column block sizes
#         row_sizes = [A_nest.getNestSubMatrix(i, 0).getSize()[0] for i in range(nblocks_row)]
#         col_sizes = [A_nest.getNestSubMatrix(0, j).getSize()[1] for j in range(nblocks_col)]

#         # Compute offsets
#         row_offsets = np.cumsum([0] + row_sizes[:-1])
#         col_offsets = np.cumsum([0] + col_sizes[:-1])

#         rows_all, cols_all, vals_all = [], [], []

#         for i in range(nblocks_row):
#             for j in range(nblocks_col):
#                 A_ij = A_nest.getNestSubMatrix(i, j)
#                 if A_ij is None:
#                     continue
#                 r_off = row_offsets[i]
#                 c_off = col_offsets[j]
#                 r, c, v = process_block(A_ij, r_off, c_off)
#                 rows_all.append(r)
#                 cols_all.append(c)
#                 vals_all.append(v)

#         shape = (sum(row_sizes), sum(col_sizes))
#         return (
#             np.concatenate(rows_all),
#             np.concatenate(cols_all),
#             np.concatenate(vals_all),
#             shape
#         )

#     if A.getType() == "nest":
#         rows, cols, values, shape = flatten_matnest(A)
#     else:
#         rows, cols, values = process_block(A)
#         shape = A.getSize()

#     indices = torch.tensor([rows, cols], dtype=torch.int64)

#     if separate_real_imag and dtype in (torch.complex64, torch.complex128):
#         real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
#         real_tensor = torch.sparse_coo_tensor(indices, torch.tensor(np.real(values), dtype=real_dtype), shape)
#         imag_tensor = torch.sparse_coo_tensor(indices, torch.tensor(np.imag(values), dtype=real_dtype), shape)
#         return real_tensor, imag_tensor
#     else:
#         return torch.sparse_coo_tensor(indices, torch.tensor(values, dtype=dtype), shape)


def convert_petsc_vector_to_torch_sparse_coo_tensor(
    x: PETSc.Vec,
    dtype=torch.float32,
    eps: float = 0.0,
    separate_real_imag: bool = False
):
    """
    Converts a PETSc vector (petsc4py.PETSc.Vec) to a PyTorch sparse COO tensor,
    with optional separation of real and imaginary parts.

    Args:
        x (PETSc.Vec): The PETSc vector.
        dtype (torch.dtype): Desired PyTorch tensor data type.
        eps (float): Threshold below which values are treated as zero.
        separate_real_imag (bool): If True, returns (real_tensor, imag_tensor);
                                       otherwise returns one complex-valued sparse tensor.

    Returns:
        torch.sparse_coo_tensor or Tuple[torch.sparse_coo_tensor, torch.sparse_coo_tensor]:
            A sparse tensor of shape (n, 1) or a tuple of two such tensors for real and imaginary parts.
    """
    if not isinstance(x, PETSc.Vec):
        raise TypeError("Input x must be of type petsc4py.PETSc.Vec")

    x.assemble()
    x_array = x.getArray()
    n = x_array.size

    # Mask for entries with absolute value above eps
    mask = np.abs(x_array) > eps
    nz_indices = np.nonzero(mask)[0]
    nz_values = x_array[nz_indices]

    indices = torch.tensor([nz_indices, np.zeros_like(nz_indices)], dtype=torch.int64)

    if separate_real_imag and dtype in (torch.complex64, torch.complex128):
        real_values = np.real(nz_values)
        imag_values = np.imag(nz_values)
        if dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise ValueError("dtype must be either torch.complex64 or torch.complex128 for separate_real_imag=True")

        real_tensor = torch.sparse_coo_tensor(indices, torch.tensor(real_values, dtype=real_dtype), (n,1))
        imag_tensor = torch.sparse_coo_tensor(indices, torch.tensor(imag_values, dtype=real_dtype), (n,1))
        return real_tensor, imag_tensor
    else:
        values = torch.tensor(nz_values, dtype=dtype)
        return torch.sparse_coo_tensor(indices, values, (n,1))


def convert_weight_to_tensor(weight, dtype):
    weight_tensor = {'A00': None, 'A01': None, 'A11': None}
    weight_tensor['A00'] = convert_petsc_mat_to_torch_sparse_coo_tensor(weight['A00'], dtype=dtype)
    weight_tensor['A01'] = torch.tensor(weight['A01'].getArray(),dtype=dtype)
    weight_tensor['A11'] = torch.tensor(weight['A11'], dtype=dtype)
    return weight_tensor