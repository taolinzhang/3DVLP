import torch
class Prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data_dict = next(self.loader)
        except StopIteration:
            self.data_dict = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.data_dict:
                # data_dict[key] = data_dict[key].cuda()
                self.data_dict[key] = self.data_dict[key].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_dict = self.data_dict
        self.preload()
        return data_dict