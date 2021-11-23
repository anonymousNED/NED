from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume training')
        self.parser.add_argument('--finetune', action='store_true', help='Whether to perform regular training or finetuning')
        self.parser.add_argument('--finetune_epoch', type=int, default=20, help='Epoch to load checkpoint for finetuning')

        self.parser.add_argument('--niter', type=int, default=20, help='# of total epochs')
        self.parser.add_argument('--niter_decay', type=int, default=10, help='# of epochs for decaying lr')

        self.parser.add_argument('--beta1', type=float, default=0.0, help='decay rate for 1st moment of Adam')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='decay rate for 2nd moment of Adam')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--f_lr', type=float, default=1e-4, help='initial learning rate for mapping network')

        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cycle consistency loss')
        self.parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
        self.parser.add_argument('--lambda_mouth', type=float, default=1, help='Weight for mouth loss')

        self.parser.add_argument('--database', type=str, choices=['aff-wild2', 'MEAD'], required=True, help='Which of the 2 databases to use')
        self.parser.add_argument('--train_root', type=str, default='/gpu-data3/filby/MEAD', help='Directory containing (reconstructed) MEAD/aff-wild2')
        self.parser.add_argument('--ann_path', type=str, default='/gpu-data/fpar/aff-wild2/expression_set/annotations/Train_Set',
                                 help='Directory containing annotations for affwild2-expression set')
        self.parser.add_argument('--selected_actors', type=str, nargs='+', help='Subset (or all) of the MEAD actors', default=['M003','M009','W029'])
        self.parser.add_argument('--selected_actors_val', type=str, nargs='+', help='Subset (or all) of the MEAD actors', default=['M023'])

        self.isTrain = True
