class Config:
    def __init__(self, args=None):
        if args != None:
            for name, value in vars(args).items():
                setattr(self, name, value)
        
        self.name = "Atel"
        self.labels = "./CheXpert-v1.0/test_project.csv"
        self.model_list = ["atel_densenet121_1005_test_edgetpu.tflite",
                           "atel_densenet121_1203_test_edgetpu.tflite",
                           "atel_densenet121_317_test_edgetpu.tflite",
                           "atel_densenet121_7613_test_edgetpu.tflite",
                           ]
