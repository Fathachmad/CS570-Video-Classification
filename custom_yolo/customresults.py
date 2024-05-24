from ultralytics.engine.results import Results

class CustomResults(Results):
    def __init__(
        self, orig_img, path, names, boxes=None, tot_conf=None
    ) -> None:
        super().__init__(orig_img=orig_img, path=path, names=names, boxes=boxes)
        self.tot_conf = tot_conf
    
    # Custom indexing
    def indexing(self, idx):
        self.boxes.data = self.boxes.data[idx, :]
        self.tot_conf = self.tot_conf[idx, :]