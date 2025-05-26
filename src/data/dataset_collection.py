class DatasetCollection:
    """
    Dataset collection (train_f, val_f, test_f)
    """

    def __init__(self, **kwargs):
        self.seed = None

        self.processed_data_encoder = False
        self.processed_data_decoder = False
        self.processed_data_multi = False
        self.processed_data_msm = False

        self.train_f = None
        self.val_f = None
        self.test_f = None
        self.train_scaling_params = None
        self.prev = None
        self.projection_horizon = None

        self.autoregressive = None
        self.has_covariate = None

    def process_data_multi(self):
        self.train_f.process_data(self.train_scaling_params)
        if hasattr(self, 'val_f') and self.val_f is not None:
            self.val_f.process_data(self.train_scaling_params)
        self.test_f.process_data(self.train_scaling_params)

        self.processed_data_multi = True
