class FreezeModule(nn.Module):
    def __init__(self, module):
        super(FreezeModule, self).__init__()
        self.module = module
        
    def forward(self, inp):
        results = model.track(source=inp, tracker="bytetrack.yaml")
        return x