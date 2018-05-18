import numpy as np


class DataGenerator(object):

    def __init__(self, args):
        super(DataGenerator, self).__init__()

        ''' Keep Reference of args '''

        self.args = args
        self.batch_size = args.batch_size
        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes = 1

        if args.datasource == "sinusoid":
            self.generate = self.generate_sinusoid_batch
            self.amp_range = [0.1, 5.0]
            self.phase_range = [0, np.pi]
            self.input_range = [-5.0, 5.0]
            self.dim_input = 1
            self.dim_output = 1

    def generate_sinusoid_batch(self, input_idx=None):
        # input_idx is used during qualitative testing --the number of examples
        # used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])

        print(init_inputs, outputs, amp, phase)

        return init_inputs, outputs, amp, phase
