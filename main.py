import argparse



# define arguments

parser = argparse.ArgumentParser(description='experiment entry point')

# common arguments

parser.add_argument('--exp_type', type=str, default='Test', help='experiment mode',

                    choices=['Train', 'Test'])

parser.add_argument('--Data_name', type=str, default='TaxiBJ', help='dataset name',

                    choices=['TaxiBJ', 'TaxiCQ', 'BikeNYC',

                             'TrafficZZ', 'TrafficNJ',

                             'TrafficCD', 'TrafficSH', 'TrafficTJ', 'TrafficHZ', 'TaxiBJ_CS',

                             'TDrive', 'TaxiNYC2', 'BikeNYC2',

                             'PEMS03','PEMS04', 'PEMS07', 'PEMS08'])

parser.add_argument('--Model_name', type=str, default='OurModel', help='model name',

                    choices=['OurModel'])

parser.add_argument('--model_variants', type=str, default='OurModel', help='ablation variant',

                    choices=['OurModel','OurModel_Att','OurModel_FFN','OurModel_Expert','OurModel_T','OurModel_S'])

parser.add_argument('--LossFunction', type=str, default='MSE', help='loss function',choices=['MSE','MAE','HuberLoss'])

parser.add_argument('--Epochs', type=int, default=50, help='training epochs')

parser.add_argument('--Batch_size', type=int, default=16, help='training batch size')

parser.add_argument('--Lr', type=float, default=0.01, help='learning rate')

parser.add_argument('--Continue_train', type=bool, default=False, help='whether to resume training')

# other arguments

parser.add_argument('--Checkpoint_path', type=str, default='./Checkpoints', help='checkpoint save path')

parser.add_argument('--Data_path', type=str, default='./Data', help='dataset root path')

parser.add_argument('--Device', type=str, default='cuda:0', help='training device')

parser.add_argument('--lr_lists', type=float, default=[0.005,0.001,0.0005,0.0005], help='learning-rate schedule')

args = parser.parse_args()





def create_experiment(args):

    """Create and return the experiment instance"""

    Exp = allocate_model(args)

    exp = Exp(Data_path=args.Data_path, Data_name=args.Data_name,

              Batch_size=args.Batch_size, Device=args.Device,

              Checkpoint_path=args.Checkpoint_path, Model_name=args.Model_name,

              Lr=args.Lr, Epochs=args.Epochs, Continue_train=args.Continue_train,

              model_variants=args.model_variants, LossFunction=args.LossFunction)

    return exp



def allocate_model(config):

    """Select the model from the configuration"""

    if config.Model_name == "OurModel":

        if config.exp_type == "Train":

            from Train.train_OurModel import train_exp as exp

        elif config.exp_type == "Test":

            from Test.test_OurModel import test_exp as exp

    else:

        raise ValueError(f"Unsupported model name: {config.Model_name}")

    return exp



if __name__ == '__main__':

    # batch size 1 in test mode

    if args.exp_type == 'Test':

        args.Batch_size = 1

        # create the experiment and run it

        exp = create_experiment(args)

        exp.run()

    elif args.exp_type == 'Train':

        # initial training

        exp = create_experiment(args)

        exp.run()

        # learning-rate decay training

        for lr in args.lr_lists:

            args.Lr = lr

            args.Continue_train = True

            exp = create_experiment(args)

            exp.run()

    else:

        print(f"Unsupported experiment type: {args.exp_type}")