from .base_options import BaseOptions
import datetime
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--test_ref_path', type=str, default='dataa/PIPAL/Train_images/val/Ref', help='path to test images') #dataa/PIPAL/Train_images/train/Ref
        self._parser.add_argument('--test_dst_path', type=str, default='dataa/PIPAL/Train_images/val/Dis', help='path to test images') #dataa/PIPAL/Train_images/train/Dis
        self._parser.add_argument('--test_list', type=str, default='PIPAL_val.txt', help='testing data list')
        #self._parser.add_argument('--test_ref_path', type=str, default='dataa/PIPAL/Train_images/val/Ref', help='path to test images')
        #self._parser.add_argument('--test_dst_path', type=str, default='dataa/PIPAL/Train_images/val/Dis', help='path to test images')
        #self._parser.add_argument('--test_list', type=str, default='PIPAL_val.txt', help='testing data list')
        self._parser.add_argument('--test_mode', type=bool, default=True, help='Test mode for pred. only. If False, we can calc. PLCC score between pred and mos')
        self._parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self._parser.add_argument('--test_file_name', type=str, default='test_results_AHIQ_Label.txt', help='txt path to save results')
        self._parser.add_argument('--n_ensemble', type=int, default=20, help='crop method for test: five points crop or nine points crop or random crop for several times')
        self._parser.add_argument('--flip', type=bool, default=False, help='if flip images when testing')
        self._parser.add_argument('--resize', type=bool, default=False, help='if resize images when testing')
        self._parser.add_argument('--size', type=int, default=224, help='the resize shape')
        self.is_train = False
