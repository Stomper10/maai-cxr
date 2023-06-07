import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

# private module
from .pruning_modules import Augmentation, DenseNet, Classifier


class A2IModel(tf.keras.Model):
    def __init__(self, configs):
        super().__init__()
        """ CONFIGUTRATION """
        # generatl configuration
        self.configs = configs
        initializer = tf.keras.initializers.HeNormal(seed=self.configs.general.seed)
        regularizer = tf.keras.regularizers.l2(self.configs.model.regularization)
        img_input = layers.Input(shape=(*configs.dataset.image_size, configs.dataset.image_channels))

        # logger (tracker)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.atel_auc = tf.keras.metrics.AUC()
        self.card_auc = tf.keras.metrics.AUC()
        self.cons_auc = tf.keras.metrics.AUC()
        self.edem_auc = tf.keras.metrics.AUC()
        self.plef_auc = tf.keras.metrics.AUC()


        """ MODEL ARCHITECTURE """
        # architecture 0 : image augmentation (training data only)
        self.augmentation, augmentation_output_shape = Augmentation(
            model=self,
            translation_height_factor=configs.augmentation.translation_height_factor, 
            translation_width_factor=configs.augmentation.translation_width_factor, 
            rotation_factor=configs.augmentation.rotation_factor,
            zoom_height_factor=configs.augmentation.zoom_height_factor, 
            zoom_width_factor=configs.augmentation.zoom_width_factor,
            input_shape=(*configs.dataset.image_size, configs.dataset.image_channels),
            seed=configs.general.seed
        )


        # architecture 1 : feature extractor
        if configs.model.backbone == 'densenet':
            self.feature_extractor, feature_extractor_output_shape = DenseNet(
                model=self,
                input_shape=augmentation_output_shape,
                blocks=configs.model.densenet.blocks, # DenseNet 201 + additional blocks
                growth_rate=configs.model.densenet.growth_rate,
                num_classes=0, 
                activation=None,
                seed=configs.general.seed,
                reg=configs.model.regularization,
            )
        elif configs.model.backbone == 'convnext':
            raise NotImplementedError
        else:
            raise NotImplementedError


        # architecture 2 : expert (classifier)
        # atel : ones (label '-1' as '1')
        # [0,1] binary classification
        self.atel_classifier, _ = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            target_name='atel',
            num_classes=2,
            activation='softmax',
            add_expert=configs.model.classifier.add_expert,
            expert_filters=configs.model.classifier.expert_filters,
            seed=configs.general.seed,
            reg=configs.model.regularization,
        )

        # card : multi (label '-1' as '2') 
        # [0,1,2] multiclass classification 
        self.card_classifier, _ = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            target_name='card',
            num_classes=3,
            activation='softmax',
            add_expert=configs.model.classifier.add_expert,
            expert_filters=configs.model.classifier.expert_filters,
            seed=configs.general.seed,
            reg=configs.model.regularization,
        )

        # cons : ignore (label '-1' as '0', but no loss included) 
        # [0,1] binary classification
        self.cons_classifier, _ = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            target_name='cons',
            num_classes=2,
            activation='softmax',
            add_expert=configs.model.classifier.add_expert,
            expert_filters=configs.model.classifier.expert_filters,
            seed=configs.general.seed,
            reg=configs.model.regularization,
        )

        # edem : ones (label '-1' as '1')
        # [0, 1] binary classification 
        self.edem_classifier, _ = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            target_name='edem',
            num_classes=2,
            activation='softmax',
            add_expert=configs.model.classifier.add_expert,
            expert_filters=configs.model.classifier.expert_filters,
            seed=configs.general.seed,
            reg=configs.model.regularization,
        )

        # plef : multi (label '-1' as '2')
        # [0,1,2] multiclass classification 
        self.plef_classifier, _ = Classifier(
            model=self, 
            input_shape=feature_extractor_output_shape,
            target_name='plef',
            num_classes=3,
            activation='softmax',
            add_expert=configs.model.classifier.add_expert,
            expert_filters=configs.model.classifier.expert_filters,
            seed=configs.general.seed,
            reg=configs.model.regularization,
        )
        
        if self.configs.model.use_aux_information:
            self.auxiliary_layer = None


    def extract_feature(self, x, training=False):
        """image feature extraction"""
        for idx, layer_name in enumerate(self.augmentation):
            if idx == 0:
                feature = getattr(self, layer_name)(x, training=training)
            else:
                feature = getattr(self, layer_name)(feature, training=training)      

        for idx, layer_name in enumerate(self.feature_extractor):
            if ('densenet_denseblock' in layer_name) & ('bn1' in layer_name):
                feature_for_skip = feature
            elif ('densenet_denseblock' in layer_name) & ('concatenate' in layer_name):
                feature = getattr(self, layer_name)([feature, feature_for_skip], training=training) # concatenate
                feature_for_skip = None
            else:
                feature = getattr(self, layer_name)(feature, training=training)      

        return feature


    def classify(self, feature, classifier, training=False):
        """classifier forward"""
        for idx, layer_name in enumerate(classifier):
            if idx == 0:
                pred = getattr(self, layer_name)(feature, training=training)
            else:
                pred = getattr(self, layer_name)(pred, training=training)

        return pred


    def call(self, X, training=False):
        # (x, x_aux) = X
        x = X
        """image forward"""
        feature = self.extract_feature(x, training=training)

        """label classification features"""
        atel_pred = self.classify(feature, self.atel_classifier, training=training)
        card_pred = self.classify(feature, self.card_classifier, training=training)
        cons_pred = self.classify(feature, self.cons_classifier, training=training)
        edem_pred = self.classify(feature, self.edem_classifier, training=training)
        plef_pred = self.classify(feature, self.plef_classifier, training=training)
        
        # if self.use_aux_information:
        #     information = self.auxiliary_layer(x_aug)
        #     out = out + information
        return [atel_pred, card_pred, cons_pred, edem_pred, plef_pred]


    def label_processing(self, y):
        """label (Y) setting"""
        # atel : ones (replace -1 with 1)
        atel_gt = y[:, 0]
        atel_gt = tf.where(atel_gt == tf.constant(-1.0, dtype=self.configs.general.tf_dtype), 
                           tf.constant(1.0, dtype=self.configs.general.tf_dtype), 
                           atel_gt) # float values needed !
        atel_gt = tf.cast(atel_gt, dtype=tf.int32) # onehot : integer needed
        atel_gt = tf.one_hot(atel_gt, depth=2)
        
        # card : multi (replace -1 with 2)
        card_gt = y[:, 1]
        card_gt = tf.where(card_gt == tf.constant(-1.0, dtype=self.configs.general.tf_dtype), 
                           tf.constant(2.0, dtype=self.configs.general.tf_dtype), 
                           card_gt) 
        card_gt = tf.cast(card_gt, dtype=tf.int32) # onehot : integer needed
        card_gt = tf.one_hot(card_gt, depth=3)

        # cons : ignore
        cons_gt = y[:, 2]
        cons_indices = tf.where(cons_gt != tf.constant(-1.0, dtype=self.configs.general.tf_dtype)) # ignore -1 rows
        cons_indices = tf.reshape(cons_indices, [-1])
        cons_gt = tf.gather(cons_gt, cons_indices)
        cons_gt = tf.cast(cons_gt, dtype=tf.int32) # onehot : integer needed
        cons_gt = tf.one_hot(cons_gt, depth=2)
        
        # edem : ones
        edem_gt = y[:, 3]
        edem_gt = tf.where(edem_gt == tf.constant(-1.0, dtype=self.configs.general.tf_dtype), 
                           tf.constant(1.0, dtype=self.configs.general.tf_dtype), 
                           edem_gt)
        edem_gt = tf.cast(edem_gt, dtype=tf.int32) # onehot : integer needed
        edem_gt = tf.one_hot(edem_gt, depth=2)
        
        # plef : multi
        plef_gt = y[:, 4]
        plef_gt = tf.where(plef_gt == tf.constant(-1.0, dtype=self.configs.general.tf_dtype), 
                           tf.constant(2.0, dtype=self.configs.general.tf_dtype), 
                           plef_gt) 
        plef_gt = tf.cast(plef_gt, dtype=tf.int32)
        plef_gt = tf.one_hot(plef_gt, depth=3)

        output = (
            (atel_gt, None), 
            (card_gt, None), 
            (cons_gt, cons_indices), 
            (edem_gt, None), 
            (plef_gt, None)
        )
        return output


    # def forward(self, x, x_aux, processed_y, training=False):
    def forward(self, x, processed_y, training=False):
        """forward pass"""
        outputs = self(
            x,
            training=training
        )

        """loss calculation (each label)"""
        # atel : ones (replace -1 with 1)
        atel_loss = self.compiled_loss(processed_y[0][0], outputs[0])
        
        # card : multi
        card_loss = self.compiled_loss(processed_y[1][0], outputs[1])
        
        # cons : ignore
        outputs[2] = tf.gather(outputs[2], processed_y[2][1]) # ignore some predictions
        cons_loss = self.compiled_loss(processed_y[2][0], outputs[2])
        
        # edem : ones
        edem_loss = self.compiled_loss(processed_y[3][0], outputs[3])

        # plef : multi
        plef_loss = self.compiled_loss(processed_y[4][0], outputs[4])

        # TODO: Decide whether to use weighted sum or simple averaging for combining losses
        total_loss = atel_loss + card_loss + cons_loss + edem_loss + plef_loss 

        # logging
        self.loss_tracker.update_state(total_loss)
        self.atel_auc.update_state(processed_y[0][0], outputs[0])
        self.card_auc.update_state(processed_y[1][0], outputs[1])
        self.cons_auc.update_state(processed_y[2][0], outputs[2])
        self.edem_auc.update_state(processed_y[3][0], outputs[3])
        self.plef_auc.update_state(processed_y[4][0], outputs[4])

        return total_loss


    def train_step(self, data):
        # (img, aux_info), y = data
        img, y = data
        # label processing
        processed_y = self.label_processing(y)

        # calculate loss & logging
        with tf.GradientTape() as tape:
            if self.configs.general.distributed:
                total_loss = tf.reduce_sum(
                    self.forward(
                        x=img, 
                        # x_aux=aux_info, 
                        processed_y=processed_y, 
                        training=True
                )) 
            else:
                total_loss = self.forward(
                    x=img, 
                    # x_aux=aux_info, 
                    processed_y=processed_y, 
                    training=True
                )

        # model weight update (backward)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss":self.loss_tracker.result(),
            "atel_auc":self.atel_auc.result(),
            "card_auc":self.card_auc.result(),
            "cons_auc":self.cons_auc.result(),
            "edem_auc":self.edem_auc.result(),
            "plef_auc":self.plef_auc.result(),
        }


    def test_step(self, data):
        # (img, aux_info), y = data
        img, y = data
        # label processing
        processed_y = self.label_processing(y)

        # calculate loss & logging
        if self.configs.general.distributed:
            total_loss = tf.reduce_sum(
                self.forward(
                    x=img, 
                    # x_aux=aux_info, 
                    processed_y=processed_y, 
                    training=True
            )) 
        else:
            total_loss = self.forward(
                x=img, 
                # x_aux=aux_info, 
                processed_y=processed_y, 
                training=True
            )

        return {
            "loss":self.loss_tracker.result(),
            "atel_auc":self.atel_auc.result(),
            "card_auc":self.card_auc.result(),
            "cons_auc":self.cons_auc.result(),
            "edem_auc":self.edem_auc.result(),
            "plef_auc":self.plef_auc.result(),
        }

    # for test
    def prediction(self, atel_output, card_output, cons_output, edem_output, plef_output):
        """
        Function to process the model's predictions.
        This function assumes that the predictions are logits over labels,
        and converts them into final label predictions according to the same scheme as in validation.
        """

        # Assuming atel_output, cons_output, etc. are the outputs of your model for different classes
        # Only consider the first logit for 'card' and 'plef', ignore the rest
        atel_pred = tf.argmax(atel_output[..., :1], axis=-1)
        cons_pred = tf.argmax(cons_output[..., :1], axis=-1)
        card_pred = tf.argmax(card_output[..., :1], axis=-1)
        edem_pred = tf.argmax(edem_output[..., :1], axis=-1)
        plef_pred = tf.argmax(plef_output[..., :1], axis=-1)

        return atel_pred, card_pred, cons_pred, edem_pred, plef_pred


    def initialize(self):
        img_sample = tf.zeros((1, *self.configs.dataset.image_size, self.configs.dataset.image_channels))
        # aux_sample = tf.zeros((1,2))
        self(img_sample, , training=False)


