import torch
import numpy as np

if 1==2:
    data = [[1, 2],[3, 4]]
    x_data = torch.tensor(data)

    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

if 1==2:
    shape = (2,3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    tensor = torch.rand(3,4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

if 1==2:
    tensor = torch.rand(3,4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


# We move our tensor to the current accelerator if available
tensor = torch.ones(4, 4)

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

print(f"Device tensor is stored on: {tensor.device}")

print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]} \n")
tensor[:,1] = 0
print(tensor)
print()

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Tensor concatenation: {t1} \n")


# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
print(f"y1: {y1} \n")
y2 = tensor.matmul(tensor.T)
print(f"y2: {y2} \n")

y3 = torch.rand_like(y1)
print(f"y3 rand_like: {y3} \n")

torch.matmul(tensor, tensor.T, out=y3)
print(f"y3: {y3} \n")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(tensor)
print()

# sum
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# tensor to NumPy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n} \n")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")