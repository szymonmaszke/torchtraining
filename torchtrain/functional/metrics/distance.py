import torch


# assert 2 size
def cosine(output, target, epsilon: float = 1e-8):
    return 1 - (
        output
        @ target.T
        / torch.max(
            torch.dot(
                torch.nn.functional.norm(output, p=2, dim=0),
                torch.nn.functional.norm(target, p=2, dim=0),
            ),
            epsilon,
        )
    )
