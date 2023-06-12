class Config:
    def __init__(self, args=None):
        if args != None:
            for name, value in vars(args).items():
                setattr(self, name, value)
        
        self.name = "Atel"
        self.labels = "./CheXpert-v1.0/test_project.csv"
        self.model_list = ["DenseNet201.tflite",
                           "densenet121_1005.tflite",
                           ]