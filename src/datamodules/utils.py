def Iterator(dataloader):
    while True:
        for data in dataloader:
            yield data