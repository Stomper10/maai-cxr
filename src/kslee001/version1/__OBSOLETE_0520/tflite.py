import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import AUC
import h5py


# AUROC
class AUROC(AUC):
    def __init__(self, name='auroc', **kwargs):
        # super().__init__(name=name, curve='ROC', **kwargs)
        super().__init__(name=name, **kwargs)

# lr scheduler
class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, total_steps, warmup_steps=0):
        super(CosineDecayWithWarmup, self).__init__()
        assert warmup_steps < total_steps
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            learning_rate, total_steps - warmup_steps)

        
    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps:
            return (self.learning_rate / 
                    tf.cast(self.warmup_steps, tf.float32) * 
                    tf.cast((step + 1), tf.float32))
        return self.cosine_decay(step - self.warmup_steps)

    def get_config(self):
        return {"learning_rate": np.array(self.learning_rate),
                "total_steps": np.array(self.total_steps),
                "warmup_steps": np.array(self.warmup_steps)}

if __name__ == '__main__':

    with CustomObjectScope({
            'CosineDecayWithWarmup': CosineDecayWithWarmup,
            'AUROC': AUROC, 
        }):
        model = tf.keras.models.load_model('/home/n1/gyuseonglee/workspace/AmbientAI-2023/project/EfficientNetV2B0-last_model.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('converted_model.tflite', 'wb') as f:
        f.write(tflite_model)



