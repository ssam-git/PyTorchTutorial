import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


digit_training_data = datasets.MNIST(    
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

digit_labels_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(digit_training_data), size=(1,)).item()
    # print(sample_idx)
    img, label = digit_training_data[sample_idx]
    print(digit_training_data[sample_idx])
    figure.add_subplot(rows, cols, i)
    plt.title(digit_labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze()) #, cmap="gray")
plt.show()