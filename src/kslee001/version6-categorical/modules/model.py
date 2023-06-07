import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

# private module
from .submodel import DenseNet, ExpertClassifier, LinearClassifier


class A2IModel(tf.keras.Model):
    def __init__(self, configs):
        super().__init__()
        """ CONFIGUTRATION """
        # generatl configuration
        self.configs = configs
        initializer = tf.keras.initializers.HeNormal(seed=self.configs.general.seed)
        regularizer = tf.keras.regularizers.l2(self.configs.model.regularization)
        img_input = layers.Input(shape=(*configs.dataset.image_size, configs.dataset.image_channels))

        # head configuration
        if configs.model.classifier.add_expert == True:
            Classifier = ExpertClassifier # conv filter added
        else:
            Classifier = LinearClassifier # linear head only

        # logger (tracker)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.atel_auc = tf.keras.metrics.AUC()
        self.card_auc = tf.keras.metrics.AUC()
        self.cons_auc = tf.keras.metrics.AUC()
        self.edem_auc = tf.keras.metrics.AUC()
        self.plef_auc = tf.keras.metrics.AUC()

        # loss functions
        self.criterion_atel = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=self.configs.model.label_smoothing,
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.criterion_card = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=self.configs.model.label_smoothing,
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.criterion_cons = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=self.configs.model.label_smoothing,
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.criterion_edem = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=self.configs.model.label_smoothing,
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.criterion_plef = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            from_logits=False, 
            label_smoothing=self.configs.model.label_smoothing,
            reduction=tf.keras.losses.Reduction.SUM
        )


        """ MODEL ARCHITECTURE """
        # architecture 0 : image augmentation (training data only)
        self.augmentation = tf.keras.Sequential([
            # random translation
            tf.keras.layers.RandomTranslation(
                height_factor=self.configs.augmentation.translation_height_factor,
                width_factor =self.configs.augmentation.translation_width_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.general.seed,
                fill_value=0.0,
            ),
            # random rotation
            tf.keras.layers.RandomRotation(
                factor=self.configs.augmentation.rotation_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.general.seed,
                fill_value=0.0,
            ),
            # random scaling -> scaling tensor values?? scaling tensor size ???
            tf.keras.layers.RandomZoom(
                height_factor=self.configs.augmentation.zoom_height_factor,
                width_factor=self.configs.augmentation.zoom_width_factor,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=self.configs.general.seed,
                fill_value=0.0,
            )
        ])
        img_input = self.augmentation(img_input)

        # architecture 1 : feature extractor
        if configs.model.backbone == 'densenet':
            self.feature_extractor = DenseNet(
                blocks=configs.model.densenet.blocks, 
                num_classes=0,
                image_size=configs.dataset.image_size,
                image_channels=configs.dataset.image_channels,
                seed=configs.general.seed,
                reg=configs.model.regularization,
            )
            
        elif configs.model.backbone == 'convnext':
            self.feature_extractor = tf.keras.applications.convnext.ConvNeXtBase(
                model_name='convnext_base',
                include_top=False,
                include_preprocessing=False,
                weights=None,
                input_tensor=None,
                input_shape=(*(configs.dataset.image_size), configs.dataset.image_channels),
                pooling=None,
                classes=0,
                classifier_activation=None,
            )
            
        img_input = self.feature_extractor(img_input)

        # architecture 2 : expert (classifier)
        # atel : ones (label '-1' as '1')
        # [0,1] binary classification
        self.atel_classifier = Classifier(num_classes=2, activation='sigmoid',
        filter_sizes=self.configs.model.classifier.filter_sizes, seed=self.configs.general.seed, reg=self.configs.model.regularization)
        atel_out = self.atel_classifier(img_input)

        # card : multi (label '-1' as '2') 
        # [0,1,2] multiclass classification 
        self.card_classifier = Classifier(num_classes=3, activation='softmax',
        filter_sizes=self.configs.model.classifier.filter_sizes, seed=self.configs.general.seed, reg=self.configs.model.regularization)
        card_out = self.card_classifier(img_input)

        # cons : ignore (label '-1' as '0', but no loss included) 
        # [0,1] binary classification
        self.cons_classifier = Classifier(num_classes=2, activation='sigmoid',
        filter_sizes=self.configs.model.classifier.filter_sizes, seed=self.configs.general.seed, reg=self.configs.model.regularization)
        cons_out = self.cons_classifier(img_input)

        # edem : ones (label '-1' as '1')
        # [0, 1] binary classification 
        self.edem_classifier = Classifier(num_classes=2, activation='sigmoid',
        filter_sizes=self.configs.model.classifier.filter_sizes, seed=self.configs.general.seed, reg=self.configs.model.regularization)
        edem_out = self.edem_classifier(img_input)

        # plef : multi (label '-1' as '2')
        # [0,1,2] multiclass classification 
        self.plef_classifier = Classifier(num_classes=3, activation='softmax',
        filter_sizes=self.configs.model.classifier.filter_sizes, seed=self.configs.general.seed, reg=self.configs.model.regularization)
        plef_out = self.plef_classifier(img_input)
        
        if self.configs.model.use_aux_information:
            self.auxiliary_layer = None

        self.initialize()


    def call(self, inputs, training=False):
        x, x_aux = inputs
        """augmentation during training"""
        x = self.augmentation(x, training=training)

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

        return atel_gt, card_gt, cons_gt, edem_gt, plef_gt, cons_indices


    # def distributed_train_step(self, dataset_inputs):
    #     per_replica_losses = strategy.run(self.train_step, args=(dataset_inputs,))
    #     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    # def distributed_test_step(self, dataset_inputs):
    #     return strategy.run(self.test_step, args=(dataset_inputs,))

    def forward(self, img, aux_info, processed_y, training=False):
        atel_gt, card_gt, cons_gt, edem_gt, plef_gt, cons_indices = processed_y

        """forward pass"""
        atel_pred, card_pred, cons_pred, edem_pred, plef_pred = self(inputs=(img, aux_info), training=training)

        """loss calculation (each label)"""
        # atel : ones (replace -1 with 1)
        atel_loss = self.criterion_atel(atel_gt, atel_pred)
        
        # card : multi
        card_loss = self.criterion_card(card_gt, card_pred)
        
        # cons : ignore
        cons_pred = tf.gather(cons_pred, cons_indices) # ignore some predictions
        cons_loss = self.criterion_cons(cons_gt, cons_pred)
        
        # edem : ones
        edem_loss = self.criterion_edem(edem_gt, edem_pred)

        # plef : multi
        plef_loss = self.criterion_plef(plef_gt, plef_pred)

        # TODO: Decide whether to use weighted sum or simple averaging for combining losses
        total_loss = atel_loss + card_loss + cons_loss + edem_loss + plef_loss 

        # logging
        self.loss_tracker.update_state(total_loss)
        self.atel_auc.update_state(atel_gt, atel_pred)
        self.card_auc.update_state(card_gt, card_pred)
        self.cons_auc.update_state(cons_gt, cons_pred)
        self.edem_auc.update_state(edem_gt, edem_pred)
        self.plef_auc.update_state(plef_gt, plef_pred)

        return total_loss



    def train_step(self, data):
        (img, aux_info), y = data
        # label processing
        processed_y = self.label_processing(y)

        # calculate loss & logging
        with tf.GradientTape() as tape:
            if self.configs.general.distributed:
                total_loss = tf.reduce_sum(self.forward(img, aux_info, processed_y, training=True)) 
            else:
                total_loss = self.forward(img, aux_info, processed_y, training=True)

        # model weight update
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
        (img, aux_info), y = data
        # label processing
        processed_y = self.label_processing(y)

        # calculate loss & logging
        if self.configs.general.distributed:
            total_loss = tf.reduce_sum(self.forward(img, aux_info, processed_y, training=False)) 
        else:
            total_loss = self.forward(img, aux_info, processed_y, training=False)

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

        # Convert binary classification outputs to probabilities and then to discrete labels (0 or 1)
        atel_pred = tf.cast((atel_output) > 0.5, dtype=self.configs.general.tf_dtype)
        cons_pred = tf.cast((cons_output) > 0.5, dtype=self.configs.general.tf_dtype)
        edem_pred = tf.cast((edem_output) > 0.5, dtype=self.configs.general.tf_dtype)

        # Only consider the first logit for 'card' and 'plef', ignore the rest
        card_pred = tf.cast((card_output[..., :1]) > 0.5, dtype=self.configs.general.tf_dtype)
        plef_pred = tf.cast((plef_output[..., :1]) > 0.5, dtype=self.configs.general.tf_dtype)

        return atel_pred, card_pred, cons_pred, edem_pred, plef_pred


    def initialize(self):
        self((tf.zeros((1, *self.configs.dataset.image_size, self.configs.dataset.image_channels)), tf.zeros((1, 2))))


