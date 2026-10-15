import pandas as pd

from tqdm import tqdm

from Utils.tools import *

from Models.Our.model import HASTFormer as Model

from Models.Our.model_Att import HASTFormer as Model_Att

from Models.Our.model_FFN import HASTFormer as Model_FFN

from Models.Our.model_Expert import HASTFormer as Model_Expert

from Models.Our.model_S import HASTFormer as Model_S

from Models.Our.model_T import HASTFormer as Model_T

from Utils.configs import all_configs

import time

import datetime

from openpyxl import load_workbook



class train_exp:

    def __init__(self, Data_path, Data_name, Batch_size, Epochs, Device, Checkpoint_path, Model_name, Lr,

                 Continue_train=False,  model_variants='OurModel',LossFunction='MSE'):

        self.Data_path = Data_path  # dataset path

        self.Data_name = Data_name   # dataset name

        self.Batch_size = Batch_size    # training batch size

        self.Epochs = Epochs    # training epochs

        self.Device = Device        # device

        self.Checkpoint_path = Checkpoint_path  # checkpoint save path

        self.Model_name = Model_name     # model name

        self.Lr = Lr     # learning rate

        self.Continue_train = Continue_train # whether to resume training

        self.LossFunction = LossFunction # loss function

        

        

        self.exp_name = 'val'

        if self.Data_name == 'BikeNYC' or self.Data_name == 'TaxiBJ' or self.Data_name == 'TaxiCQ':

            self.exp_name = 'test'

            

        # initialize the adj_matrix attribute

        self.adj_matrix = None  # ensure the attribute is defined in all cases

         

        # adapt to different data

        if self.Data_name in ['PEMS03','PEMS04', 'PEMS07', 'PEMS08']: 

            self.adj_matrix = None

            self.patch_size = 1

        else:

            self.patch_size = 2   

        # prepare data

        self._prepare_data()

        # initialize the model

        self._init_model(model_variants)

        # initialize the optimizer and loss function

        self._init_optimizer()

        # create the record file

        self._create_record_files()

        self.epoch_train_times = []  # records the training time of each epoch

    

    def _prepare_data(self):

        """Prepare data loaders"""

        self.train_loader = get_data_loader_all(self.Data_path, self.Data_name, self.Batch_size, mode='train')

        self.val_loader = get_data_loader_all(self.Data_path, self.Data_name, self.Batch_size, mode=self.exp_name)

        

        # define arguments

        self.config = all_configs[self.Model_name][self.Data_name]  # select the model parameters

        self.History_len = self.config['seq_in']

        self.Pred_len = self.config['seq_out']

        self.Input_channel = self.config['in_channel']

        self.Hight = self.config['hight']

        self.Width = self.config['width']

        self.Hidden_channel = self.config['hidden_channel']

        self.Main_layers = self.config['main_layers']

        self.data_type = self.config['data_type']

        self.nodes = self.config['nodes']

        self.patch_size = self.config['patch_size']

    

    def _init_model(self, model_variants):

        """Initialize the model"""

        # model class mapping

        model_classes = {

            'OurModel': Model,

            'OurModel_Att': Model_Att,

            'OurModel_FFN': Model_FFN,

            'OurModel_Expert': Model_Expert,

            'OurModel_S': Model_S,

            'OurModel_T': Model_T,

        }

        # model name mapping

        model_names = {

            'OurModel': 'HASTFormer',

            'OurModel_Att': 'HASTFormer_Att',

            'OurModel_FFN': 'HASTFormer_FFN',

            'OurModel_Expert': 'HASTFormer_Expert',

            'OurModel_S': 'HASTFormer_S',

            'OurModel_T': 'HASTFormer_T',

        }

        

        # get the model class and name

        model_class = model_classes.get(model_variants)

        if model_class is None:

            raise ValueError('model variants error')

            

        self.Model_name = model_names[model_variants]  

        # initialize the model

        self.model = model_class(

            in_channel=self.Input_channel,

            d_model=self.Hidden_channel,

            his_len=self.History_len,

            pre_len=self.Pred_len,

            num_heads=4,

            num_layers=self.Main_layers,

            in_h=self.Hight,

            in_w=self.Width,

            patch_size=self.patch_size,

            nodes=self.nodes,

            data_type=self.data_type

        ).to(device=self.Device)



        # print model hyperparameters

        print("Model:",self.Model_name)     

        print('his_len:', self.History_len)

        print('pre_len:', self.Pred_len)

        print('in_channel:', self.Input_channel)

        print('hidden_channel:', self.Hidden_channel)

        print('main_layers:', self.Main_layers)

        print('hight:', self.Hight)

        print('width:', self.Width)

        print('nodes:', self.nodes)

        print('patch_size:', self.patch_size)

        print('data_type:', self.data_type)

        print('loss function:', self.LossFunction)

        

        # print the number of parameters

        print('Total params: {:.2f}M'.format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))

    

    def _init_optimizer(self):

        """Initialize the optimizer and loss function"""

        # print the learning rate

        print('Learning rate: {}'.format(self.Lr))

        # define the optimizer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.Lr)

        # define the learning-rate scheduler

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(

            self.optimizer, 

            max_lr=self.Lr,

            steps_per_epoch=len(self.train_loader),

            epochs=self.Epochs

        )

        # define the loss function

        self.criterion = torch.nn.MSELoss()  # MSE loss by default

        if self.LossFunction == 'MSE':

            self.criterion = torch.nn.MSELoss()

        elif self.LossFunction == 'MAE':

            self.criterion = torch.nn.L1Loss()

        elif self.LossFunction == 'HuberLoss':

            self.criterion = torch.nn.HuberLoss()

        self.val_loss_min = np.inf

        self.prev_val_loss = np.inf  # record of the previous-epoch validation loss

    

    def _create_record_files(self):

        """Create the record file"""

        # create the log directory

        log_dir = os.path.join(self.Checkpoint_path, 'logs')

        os.makedirs(log_dir, exist_ok=True)

        

        # build the log file path for the model+dataset pair

        self.log_file_path = os.path.join(log_dir, f'{self.Model_name}_{self.Data_name}.xlsx')

        

        

        # initialize the record lists

        self.record_loss_train = []

        self.record_loss_val = []

        

    

    def _save_model(self, val_loss, epoch):

        """Save the model"""

        if val_loss < self.val_loss_min:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))

            self.val_loss_min = val_loss

            

            # ensure the directory exists

            save_dir = os.path.join(self.Checkpoint_path, 'results', self.Data_name)

            os.makedirs(save_dir, exist_ok=True)

            

            # save the model

            torch.save(

                self.model.state_dict(),

                os.path.join(save_dir, '{}_{}_{}.pth'.format(

                    self.Model_name, self.History_len, self.Pred_len

                ))

            )

    

    def _validate(self):

        """Validate the model"""

        val_loss = 0

        val_loader_tqdm = tqdm(self.val_loader, desc='Val Loss ')

        with torch.no_grad():

            self.model.eval()

            for i, data in enumerate(val_loader_tqdm):

                x, y = split_batch(data, self.History_len, self.Pred_len, self.Data_name)

                x = x.to(device=self.Device)

                y = y.to(device=self.Device)

                output = self.model(x)

                loss = self.criterion(output, y)

                val_loss += loss.item()

                # update the progress-bar description

                val_loader_tqdm.set_description(f'Val Loss {val_loss / (i + 1):.6f}')

        return val_loss / len(self.val_loader)

    

    def _train_epoch(self, epoch):

        """Train one epoch"""

        

        self.train_loss = 0

        train_loader_tqdm = tqdm(self.train_loader, desc='Epoch {} '.format(epoch + 1))

        self.model.train()

        

        epoch_start_time = time.time()  # record the epoch start time

        for i, data in enumerate(train_loader_tqdm):

            x, y = split_batch(data, self.History_len, self.Pred_len, self.Data_name)

            self.optimizer.zero_grad()

            x = x.to(device=self.Device)

            y = y.to(device=self.Device)

            output = self.model(x)

            loss = self.criterion(output, y)

            loss.backward()

            self.optimizer.step()  # update the parameters

            self.scheduler.step()  # update the learning rate

            self.train_loss += loss.item()

            # update the progress-bar description

            train_loader_tqdm.set_description(f'Epoch {epoch + 1} Loss {self.train_loss / (i + 1):.6f}')

        self.epoch_train_times.append(time.time() - epoch_start_time)  # record the pure training time



    



    def run(self):

        """Run the training procedure"""

        # load the previous model when resuming training

        if self.Continue_train:

            model_path = os.path.join(self.Checkpoint_path, 'results', self.Data_name,

                                        '{}_{}_{}.pth'.format(self.Model_name, self.History_len, self.Pred_len))

            if os.path.exists(model_path):

                self.model.load_state_dict(torch.load(model_path))

                print('Resuming training: model loaded successfully!')

                val_loss = self._validate()

                self.prev_val_loss = val_loss  # initialize the previous-epoch validation loss

                # save the best model

                if val_loss < self.val_loss_min:

                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))

                    self.val_loss_min = val_loss

                print('Resuming training!')

            else:

                print(f"Model file {model_path} not found; training from scratch.")

        

        

        # record the training start time

        self.start_time = time.time()

        self.start_time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")

        

        # train the model

        for epoch in range(self.Epochs):

            # train one epoch

            self._train_epoch(epoch)

            

            # validate the model

            val_loss = self._validate()

            

            # check whether the validation loss is more than 1.5x the previous epoch

            if epoch > 0 and val_loss > 2 * self.prev_val_loss:

                print(f'Validation loss increased by more than 2x ({self.prev_val_loss:.6f} --> {val_loss:.6f}). Early stopping.')

                break

                

            self.prev_val_loss = val_loss

            

            # save the best model

            self._save_model(val_loss, epoch)

            

            print('Model: {}, dataset: {}, Epoch: {}, Train Loss: {:.6f}, Val Loss: {:.6f}\n'.

                    format('{}_{}_{}'.format(self.Model_name, self.History_len, self.Pred_len), 

                            self.Data_name, epoch + 1,

                            self.train_loss / len(self.train_loader),

                            val_loss))

            

            # record the loss of each epoch

            self.record_loss_train.append(self.train_loss / len(self.train_loader))  # record the training loss

            self.record_loss_val.append(val_loss)  # record the validation loss

        

        # compute the training time

        self.end_time = time.time()

        self.total_time = self.end_time - self.start_time

        

        # save the training log

        self._save_training_log()

        

    def _save_training_log(self):

        """Save the training log to an Excel file"""

        # get the model parameter count

        total_params = sum(p.numel() for p in self.model.parameters()) / 1000000.0

        

        # create the experiment info dict

        exp_info = {

            "StartTime": self.start_time_str,

            "Batch": self.Batch_size,

            "Lr": self.Lr,

            "Epochs": self.Epochs,

            "Continue": self.Continue_train,

            "His": self.History_len,

            "Pred": self.Pred_len,

            "Params": f"{total_params:.2f}M",

            "Hidden": self.Hidden_channel,

            "Layers": self.Main_layers,

            "TrainTime": f"{sum(self.epoch_train_times)/60:.2f}min",

            "TotalTime": f"{self.total_time/60:.2f}min",

            "MinValLoss": f"{self.val_loss_min:.6f}",

            "LossFunction": self.LossFunction

        }

        

        # check whether the file exists

        if os.path.exists(self.log_file_path):

            try:

                # load the existing workbook if the file exists

                book = load_workbook(self.log_file_path)

                

                # ensure at least one sheet is visible

                if 'Training Log' not in book.sheetnames:

                    book.create_sheet('Training Log')

                

                # get the worksheet

                sheet = book['Training Log']

                

                # find the last row

                last_row = sheet.max_row

                if last_row == 0:

                    last_row = 0

                

                # import font settings

                from openpyxl.styles import Font

                bold_font = Font(bold=True)

                

                # part 1: experiment parameters (keys and values on separate rows)

                # write the key row in bold

                for col_idx, key in enumerate(exp_info.keys()):

                    cell = sheet.cell(row=last_row+1, column=col_idx+1, value=key)

                    cell.font = bold_font

                

                # write the value row

                for col_idx, value in enumerate(exp_info.values()):

                    sheet.cell(row=last_row+2, column=col_idx+1, value=value)

                

                # part 2: loss information

                # write the training-loss row

                for col_idx, loss in enumerate(self.record_loss_train):

                    sheet.cell(row=last_row+3, column=col_idx+1, value=f"{loss:.6f}")

                

                # write the validation-loss row

                for col_idx, loss in enumerate(self.record_loss_val):

                    sheet.cell(row=last_row+4, column=col_idx+1, value=f"{loss:.6f}")

                

                # save the workbook

                book.save(self.log_file_path)

                

            except Exception as e:

                print(f"Error updating the existing Excel file: {e}")

                print("Creating a new Excel file...")

                self._create_new_excel_log(exp_info)

        else:

            # create a new workbook if the file does not exist

            self._create_new_excel_log(exp_info)

        

        print(f"Training log saved to: {self.log_file_path}")

    

    def _create_new_excel_log(self, exp_info):

        """Create a new Excel log file"""

        # prepare data

        data = []

        

        # row 1: parameter keys

        data.append(list(exp_info.keys()))

        

        # row 2: parameter values

        data.append(list(exp_info.values()))

        

        # row 3: training loss

        data.append([f"{loss:.6f}" for loss in self.record_loss_train])

        

        # row 4: validation loss

        data.append([f"{loss:.6f}" for loss in self.record_loss_val])

        

        # build the dataframe and save it

        df = pd.DataFrame(data)

        df.to_excel(self.log_file_path, index=False, header=False, sheet_name='Training Log')

        

        # open the saved file and set the key row to bold

        try:

            from openpyxl.styles import Font

            book = load_workbook(self.log_file_path)

            sheet = book['Training Log']

            bold_font = Font(bold=True)

            

            # set the first row (keys) to bold

            for col_idx in range(1, len(exp_info.keys()) + 1):

                cell = sheet.cell(row=1, column=col_idx)

                cell.font = bold_font

                

            # save the modified workbook

            book.save(self.log_file_path)

        except Exception as e:

            print(f"Error setting bold style: {e}")