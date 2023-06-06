import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

# private module
from .densenet import DenseNet, Expert
IMAGE_CHANNEL = 1

class A2IModelBase(tf.keras.Model):
    def __init__(self, configs):
        super().__init__()
        # configuration
        self.configs = configs
        initializer = tf.keras.initializers.HeNormal(seed=self.configs.seed)
        regularizer = tf.keras.regularizers.l2(self.configs.regularization)

        # logger (tracker)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.atel_auc = tf.keras.metrics.AUC()
        self.card_auc = tf.keras.metrics.AUC()
        self.cons_auc = tf.keras.metrics.AUC()
        self.edem_auc = tf.keras.metrics.AUC()
        self.plef_auc = tf.keras.metrics.AUC()

        # loss functions
        self.criterion_atel = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.configs.label_smoothing)
        self.criterion_card = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.configs.label_smoothing)
        self.criterion_cons = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.configs.label_smoothing)
        self.criterion_edem = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.configs.label_smoothing)
        self.criterion_plef = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.configs.label_smoothing)

        # architecture 0 : image augmentation (training data only)
        self.augmentation = tf.keras.Sequential([
            # random translation
            tf.keras.layers.RandomTranslation(
                height_factor=self.configs.translation_height_factor,
                width_factor =self.configs.translation_width_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.seed,
                fill_value=0.0,
            ),
            # random rotation
            tf.keras.layers.RandomRotation(
                factor=self.configs.rotation_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.seed,
                fill_value=0.0,
            ),
            # random scaling -> scaling tensor values?? scaling tensor size ???
            tf.keras.layers.RandomZoom(
                height_factor=self.configs.zoom_height_factor,
                width_factor=self.configs.zoom_width_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.seed,
                fill_value=0.0,
            )
        ])

        # architecture 1 : feature extractor
        self.feature_extractor = DenseNet(
            blocks=configs.blocks,
            input_shape=configs.image_size,
            image_channels=configs.image_channels,
            seed=configs.seed,
            reg=configs.regularization,
        )

        # architecture 2 : expert (classifier)
        # atel : ones (label '-1' as '1')
        # [0,1] binary classification
        self.atel_classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

        # card : multi (label '-1' as '2') 
        # [0,1,2] multiclass classification 
        self.card_classifier = tf.keras.layers.Dense(
            3,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

        # cons : ignore (label '-1' as '0', but no loss included) 
        # [0,1] binary classification
        self.cons_classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

        # edem : ones (label '-1' as '1')
        # [0, 1] binary classification 
        self.edem_classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )

        # plef : multi (label '-1' as '2') 
        # [0,1,2] multiclass classification 
        self.plef_classifier = tf.keras.layers.Dense(
            3,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
        )
        
        if self.configs.use_aux_information:
            self.auxiliary_layer = None

    def call(self, inputs, training=False):
        x, x_aux = inputs
        """augmentation during training"""
        x = self.augmentation(img, training=training)

        feature = self.feature_extractor(x)
        
        atel_pred = self.atel_classifier(feature)
        card_pred = self.card_classifier(feature)
        cons_pred = self.cons_classifier(feature)
        edem_pred = self.edem_classifier(feature)
        plef_pred = self.plef_classifier(feature)

        # if self.use_aux_information:
        #     information = self.auxiliary_layer(x_aug)
        #     out = out + information
        return (atel_pred, card_pred, cons_pred, edem_pred, plef_pred)


    def train_step(self, data):
        (img, aux_info), y = data

        """label (Y) setting"""
        # atel : ones (replace -1 with 1)
        atel_gt = y[:, 0]
        atel_gt = self.replace_value_onehot(array=atel_gt, value=1.0, classes=2)

        # card : multi (replace -1 with 2)
        card_gt = y[:, 1]
        card_gt = self.replace_value_onehot(array=card_gt, value=2.0, classes=3)

        # cons : ignore
        cons_gt = y[:, 2]
        cons_indices = tf.where(cons_gt != tf.constant(-1.0, dtype=self.configs.tf_dtype)) # ignore -1 rows
        cons_indices = tf.reshape(cons_indices, [-1])
        cons_gt = tf.gather(cons_gt, cons_indices)

        # edem : ones (replace -1 with 1)
        edem_gt = y[:, 3]
        edem_gt = self.replace_value_onehot(array=edem_gt, value=1.0, classes=2)

        # plef : multi (replace -1 with 2)
        plef_gt = y[:, 4]
        plef_gt = self.replace_value_onehot(array=plef_gt, value=2.0, classes=3)


        with tf.GradientTape() as tape:
            """forward pass"""
            atel_pred, card_pred, cons_pred, edem_pred, plef_pred = self(inputs=(img, aux_info), training=True)

            """loss calculation (each label)"""
            # atel : ones (replace -1 with 1)
            atel_loss = self.criterion_atel(atel_gt, atel_pred)
            
            # card : multi
            card_loss = self.criterion_atel(card_gt, card_pred)
            
            # cons : ignore
            cons_pred = tf.gather(cons_pred, cons_indices) # ignore some predictions
            cons_loss = self.criterion_cons(cons_gt, cons_pred)
            
            # edem : ones
            edem_loss = self.criterion_edem(edem_gt, edem_pred)

            # plef : multi
            plef_loss = self.criterion_plef(plef_gt, plef_pred)

            # TODO: Decide whether to use weighted sum or simple averaging for combining losses
            total_loss = atel_loss + card_loss + cons_loss + edem_loss + plef_loss 
            # scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            
        # scaled_gradients = tape.gradient(total_loss, self.trainable_variables)
        # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # model weight update
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # logging
        self.loss_tracker.update_state(total_loss)
        self.atel_auc.update_state(atel_gt, atel_pred)
        self.card_auc.update_state(card_gt, card_pred)
        self.cons_auc.update_state(cons_gt, cons_pred)
        self.edem_auc.update_state(edem_gt, edem_pred)
        self.plef_auc.update_state(plef_gt, plef_pred)

        return {
            "loss":self.loss_tracker.result(),
            "atel_auc":self.atel_auc.result(),
            "card_auc":self.card_auc.result(),
            "cons_auc":self.cons_auc.result(),
            "edem_auc":self.edem_auc.result(),
            "plef_auc":self.plef_auc.result(),
        }


    def test_step(self, data):
        (img, aux_info), y = data

        """No augmentation during validation"""
        # img = self.augmentation(img)

        """label (Y) setting"""
        # atel : ones (replace -1 with 1)
        atel_gt = y[:, 0]
        atel_gt = self.replace_value_onehot(array=atel_gt, value=1.0, classes=2)

        # card : multi (replace -1 with 2)
        card_gt = y[:, 1]
        card_gt = self.replace_value_onehot(array=card_gt, value=2.0, classes=3)

        # cons : ignore
        cons_gt = y[:, 2]
        cons_indices = tf.where(cons_gt != tf.constant(-1.0, dtype=self.configs.tf_dtype)) # ignore -1 rows
        cons_indices = tf.reshape(cons_indices, [-1])
        cons_gt = tf.gather(cons_gt, cons_indices)

        # edem : ones (replace -1 with 1)
        edem_gt = y[:, 3]
        edem_gt = self.replace_value_onehot(array=edem_gt, value=1.0, classes=2)

        # plef : multi (replace -1 with 2)
        plef_gt = y[:, 4]
        plef_gt = self.replace_value_onehot(array=plef_gt, value=2.0, classes=3)


        # with tf.GradientTape() as tape:
        if True:
            """forward pass"""
            atel_pred, card_pred, cons_pred, edem_pred, plef_pred = self(inputs=(img, aux_info), training=True)

            """loss calculation (each label)"""
            # atel : ones (replace -1 with 1)
            atel_loss = self.criterion_atel(atel_gt, atel_pred)
            
            # card : multi
            card_loss = self.criterion_atel(card_gt, card_pred)
            
            # cons : ignore
            cons_pred = tf.gather(cons_pred, cons_indices) # ignore some predictions
            cons_loss = self.criterion_cons(cons_gt, cons_pred)
            
            # edem : ones
            edem_loss = self.criterion_edem(edem_gt, edem_pred)

            # plef : multi
            plef_loss = self.criterion_plef(plef_gt, plef_pred)

            # TODO: Decide whether to use weighted sum or simple averaging for combining losses
            total_loss = atel_loss + card_loss + cons_loss + edem_loss + plef_loss 


        # scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        # scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        # gradients = tape.gradient(total_loss, self.trainable_variables)

        # # model weight update
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # logging
        self.loss_tracker.update_state(total_loss)
        self.atel_auc.update_state(atel_gt, atel_pred)
        self.card_auc.update_state(card_gt, card_pred)
        self.cons_auc.update_state(cons_gt, cons_pred)
        self.edem_auc.update_state(edem_gt, edem_pred)
        self.plef_auc.update_state(plef_gt, plef_pred)

        return {
            "loss":self.loss_tracker.result(),
            "atel_auc":self.atel_auc.result(),
            "card_auc":self.card_auc.result(),
            "cons_auc":self.cons_auc.result(),
            "edem_auc":self.edem_auc.result(),
            "plef_auc":self.plef_auc.result(),
        }


    def initialize(self):
        self((tf.zeros((1, *self.configs.image_size, IMAGE_CHANNEL)), tf.zeros((1, 2))))
        self.augmentation(tf.zeros((1, *self.configs.image_size, IMAGE_CHANNEL)))


    def replace_value_onehot(self, array, value, classes):
        array = tf.where(array == tf.constant(-1.0, dtype=self.configs.tf_dtype), 
                           tf.constant(value, dtype=self.configs.tf_dtype), 
                           array) 
        array = tf.cast(array, dtype=tf.int32)
        array = tf.one_hot(array, depth=classes)

        return array

