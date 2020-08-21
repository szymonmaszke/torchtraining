import torch


def random_split(
    dataset: torch.utils.data.Dataset, *p: float, generator=torch.default_generator
):
    """Randomly split a dataset into non-overlapping new datasets of given proportions.

    Arguments
    ---------
    dataset: torch.utils.data.Dataset
        Dataset to be split
    *p: float
        Floating point values in the `[0, 1]`. All of them should sum to `1`
        (if not they will be normalized to `[0, 1]` range). Split dataset
        according to those proportions.
    generator: Generator
        Generator used for the random permutation.

    Returns
    -------
    Tuple[torch.utils.data.Dataset]
        Tuple containing splitted datasets.

    """
    if len(p) < 2:
        raise ValueError(
            "number of proportions should be at least 2, got {}".format(len(p))
        )
    # Normalize to [0, 1] range
    p = torch.tensor(p)
    p = p / torch.sum(p)
    lengths = [int(len(dataset) * split) for split in p[:-1]]
    return torch.utils.data.random_split(
        dataset, lengths + [len(dataset) - sum(lengths)], generator=generator
    )
