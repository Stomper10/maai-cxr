import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import wandb
from wandb.keras import WandbCallback


class CustomOneCycleSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, epochs, steps_per_epoch, start_lr=None, end_lr=None, warmup_fraction=None):
        super().__init__()

        if start_lr is None:
            start_lr = max_lr / 10

        if end_lr is None:
            end_lr = start_lr / 1000

        if warmup_fraction is None:
            warmup_fraction = 0.1  # default to warm up for 10% of total steps

        self.start_lr = start_lr
        self.max_lr = max_lr
        self.end_lr = end_lr
        self.total_steps = int(epochs * steps_per_epoch)
        self.warmup_steps = int(warmup_fraction * self.total_steps)


    def __call__(self, step):
        def warmup_fn():
            return self.start_lr + (self.max_lr - self.start_lr) * (step / self.warmup_steps)

        def decay_fn():
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.max_lr * 0.5 * (1.0 + tf.math.cos(np.pi * progress))

        step = tf.cast(step, tf.float32)

        lr = tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)
        return tf.where(step >= self.total_steps, self.end_lr, lr, name="learning_rate")


    def get_config(self):
        return {
            'start_lr': self.start_lr,
            'max_lr': self.max_lr,
            'end_lr': self.end_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps
        }


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # In this case, the learning rate is a function of the current step
            lr = lr(self.model.optimizer.iterations)
        else:
            # Otherwise, it's a constant
            lr = lr.numpy()

        wandb.log({'learning_rate': lr})

    def on_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        else:
            lr = lr.numpy()

        wandb.log({'learning_rate': lr})
