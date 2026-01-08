def infinite_loader(loader):
    """
    Get an infinite stream of batches from a data loader.

    Useful for iteration-based training where you want to iterate
    for a fixed number of iterations rather than epochs.

    Args:
        loader: PyTorch DataLoader

    Yields:
        Batches from the loader, infinitely cycling through epochs

    Example:
        >>> train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        >>> train_iter = iter(infinite_loader(train_loader))
        >>> for i in range(10000):
        >>>     batch = next(train_iter)
    """
    while True:
        yield from loader
