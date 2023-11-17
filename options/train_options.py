from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_epoch', type=int, default=51, help='total epoch for training')
        self._parser.add_argument('--save_interval', type=int, default=5, help='interval for saving models')
        self._parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate') # 1e-4
        self._parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size') # default = 4
        self._parser.add_argument('--val_freq', type=int, default=1, help='validation frequency') #default = 1
        self._parser.add_argument('--T_max', type=int, default=50, help="cosine learning rate period (iteration)") # defualt = 50
        self._parser.add_argument('--eta_min', type=int, default=0, help="mininum learning rate") # defualt =0

        # For Testing in 0419
        self._parser.add_argument('--test_file_name', type=str, default='test_results_score_0524.txt', help='txt path to save results')
        self._parser.add_argument('--test_mode', type=bool, default=False, help='Test mode for pred. only. If False, we can calc. PLCC score between pred and mos')
        self._parser.add_argument('--n_ensemble', type=int, default=20,help='crop method for test: five points crop or nine points crop or random crop for several times')

        self.is_train = True