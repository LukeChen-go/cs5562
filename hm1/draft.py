import pandas as pd
import torch

import matplotlib.pyplot as plt

x, y = pd.read_pickle("x-y.pkl")


# y = [y_ for y_ in y]


plt.plot(x, y, marker='o')


plt.title(
    f"Model accuracy on adversarial images for various epsilon values\n Alpha = 2/255, Steps = {20}, Batch size = {100}, Batch num = {20}")

plt.xlabel("Epsilon values")
plt.ylabel("Attack accuracy (%)")


plt.savefig("result.png")
plt.show()