import numpy as np
import torch 
from typing import Tuple
#dataset: npt.NDArray, batch_size: int, context_length: int, device: str
class DataLoader:
    def load(self, dataset: np.array, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start_index = len(dataset) - context_length - 1
        input_index = np.random.randint(0, max_start_index + 1, size=batch_size)
        input_value = np.array([dataset[i : i + context_length] for i in input_index])
        output_value = np.array([dataset[i + 1 : i + context_length + 1] for i in input_index])

        inputs = torch.tensor(input_value, device=device,  dtype=torch.long)
        targets = torch.tensor(output_value, device=device,  dtype=torch.long)

        return inputs, targets



