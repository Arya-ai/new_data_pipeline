import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten

class MultiModel():

    def __init__(self, options):
        self.model = None
        self.input_shapes = None
        self.output_shapes = None
        self.generator = None
        self.n_samples = None
        self.batch_size = int(options['batch_size'])
        self.nb_epochs = int(options['epochs'])


    def train(self, options):
        self.input_shapes = options['input_shapes']
        self.output_shapes = options['output_shapes']
        self.generator = options['generator']
        self.n_samples = options['n_samples']
        self.batch_size = options['batch_size']

        input_layers = []
        inter_layers = []
        flat_layers = []
        output_layers = []

        # print self.input_shapes
        # print self.output_shapes

        _input_layer = Input(shape=(224,224,3))
        Flat = Flatten()(_input_layer)
        flat_layers.append(Flat)

        top_model = Model(_input_layer, Flat)

        # # inputs
        for shape in self.input_shapes:
            input_layer = Input(shape=shape)
            input_layers.append(input_layer)
            inter_layer = top_model(input_layer)
            inter_layers.append(inter_layer)

        merged_layer = keras.layers.concatenate([inter_layer for inter_layer in inter_layers], axis=-1)

        # outputs
        for shape in self.output_shapes:
            total_dim = 1
            for dim in shape:
                total_dim *= dim

            y = Dense(total_dim, activation='sigmoid')(merged_layer)
            output_layer = Reshape(shape)(y)
            output_layers.append(output_layer)

        self.model = Model(inputs=input_layers, outputs=output_layers)

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        print "Model compiled."

        self.n_batches = self.n_samples // self.batch_size

        self.model.fit_generator(self.generator, steps_per_epoch=self.n_batches, epochs=self.nb_epochs, verbose=2)

        print "Done training."
