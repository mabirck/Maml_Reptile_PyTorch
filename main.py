from args import get_args
from DataLoader import DataGenerator
from maml import MAML


def train(model, exp_string, data_gen, resume_itr=0, args=None):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if args.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    prelosses, postlosses = [], []
    num_classes = DataGenerator.num_classes  # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for i in range(resume_itr, args.pre_iter + args.meta_iter):
        batch_x, batch_y, amp, phase = DataGenerator.generate()

        ''' Oracle adds extra info into the model '''
        if(args.oracle):
            batch_x[i, :, 1] = amp[i]
            batch_y[i, :, 2] = phase[i]

        ''' Get train batch '''
        input_train = batch_x[:, :num_classes * args.update_batch_size, :]
        label_train = batch_y[:, :num_classes * args.update_batch_size, :]

        ''' Get test batch '''
        input_test = batch_x[:, num_classes * args.update_batch_size:, :]
        label_test = batch_y[:, num_classes * args.update_batch_size:, :]
        
        if i < args.pre_iter:
            pass
            ''' NOT SURE WHAT TODO '''

        if (i % SUMMARY_INTERVAL == 0 or i % PRINT_INTERVAL == 0):
            pass
            ''' NOT SURE WHAT TODO '''

        result = model(input)


def test(model, exp_string, data_gen, test_num_updates=None, args=None):
    assert "Not Implemented Yet"


def main():
    ''' Arguments which dictate how training is gonna be! '''
    args = get_args()

    print(args)

    if args.dataset == 'sinusoid':
        if args.train:
            test_num_updates = 5
        else:
            test_num_updates = 10

    if args.test:
        orig_meta_batch_size = args.meta_batch_size
        # always use meta batch size of 1 when testing.
        args.meta_batch_size = 1

    if args.datasource == 'sinusoid':
        data_generator = DataGenerator(args.update_batch_size*2, args.meta_batch_size)

    dim_output = data_generator.dim_output

    if args.oracle == 'oracle':
        assert args.datasource == 'sinusoid'
        dim_input = 3
        args.pretrain_iterations += args.metatrain_iterations
        args.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)

    if not args.test:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')

    # TODO check this out, summop, saver
    # model.summ_op = tf.summary.merge_all()
    # saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE\
    # _VARIABLES), max_to_keep=10)
    # sess = tf.InteractiveSession()

    if args.test:
        # change to original meta batch size when loading model.
        args.meta_batch_size = orig_meta_batch_size

    if args.train_update_batch_size == -1:
        args.train_update_batch_size = args.update_batch_size
    if args.train_update_lr == -1:
        args.train_update_lr = args.update_lr

    resume_itr = 0
    model_file = None

    exp_string = ""

    # tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    if args.resume or args.test:
        # model_file = tf.train.latest_checkpoint(args.logdir + '/' + exp_string)
        assert "should load model with pytorch"
        if args.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' \
                         + str(args.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            # saver.restore(sess, model_file)

    if args.test:
        test(model, exp_string, data_generator, test_num_updates, args)
    else:
        train(model, exp_string, data_generator, resume_itr, args)


if __name__ == '__main__':
    main()
