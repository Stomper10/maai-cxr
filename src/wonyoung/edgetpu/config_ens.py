class Config:
    def __init__(self, args=None):
        if args != None:
            for name, value in vars(args).items():
                setattr(self, name, value)
        
        self.labels = "./CheXpert-v1.0/test_project.csv"
        self.model_list = [#"densenet121_1000_test_edgetpu.tflite",
                           #"densenet121_3000_test_edgetpu.tflite",
                           #"densenet121_4000_test_edgetpu.tflite",
                           "densenet121_5000_test_edgetpu.tflite",
                           #"densenet121_7000_test_edgetpu.tflite",
                           #"densenet121_8000_test_edgetpu.tflite",
                           #"densenet121_9000_test_edgetpu.tflite",
                           "densenet121_10000_test_edgetpu.tflite",
                           ]