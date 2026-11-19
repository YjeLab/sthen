all_configs = {'OurModel': {}}

# OurModel configs

all_configs['OurModel']['TaxiBJ_CS'] = {'seq_in': 4, 'seq_out': 1, 'in_channel': 2, 'main_layers': 4, 'hidden_channel': 64,

                                     'patch_size': 2, 'hight': 32, 'width': 32, 'nodes': None, 'data_type': 'Grid'}  # case study only (4->1)

all_configs['OurModel']['TaxiBJ'] = {'seq_in': 6, 'seq_out': 1, 'in_channel': 2, 'main_layers': 6, 'hidden_channel': 128,

                                     'patch_size': 2, 'hight': 32, 'width': 32, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TaxiCQ'] = {'seq_in': 6, 'seq_out': 1, 'in_channel': 2,'main_layers': 4, 'hidden_channel': 64, 

                                     'patch_size': 2, 'hight': 20, 'width': 25, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['BikeNYC'] = {'seq_in': 6, 'seq_out': 1, 'in_channel': 2, 'main_layers': 4, 'hidden_channel': 64,

                                     'patch_size': 2, 'hight': 16, 'width': 8, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficSH'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64,

                                        'patch_size': 2, 'hight': 28, 'width': 26, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficCD'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1,  'main_layers': 4,'hidden_channel': 64,

                                        'patch_size': 2, 'hight': 28, 'width': 26,'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficNJ'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64, 

                                        'patch_size': 2, 'hight': 32, 'width': 24, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficZZ'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64, 

                                        'patch_size': 2, 'hight': 26, 'width': 26, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficTJ'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1,'main_layers': 4, 'hidden_channel': 64,

                                        'patch_size': 2, 'hight': 24, 'width': 30,'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TrafficHZ'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1,  'main_layers': 4,'hidden_channel': 64,

                                        'patch_size': 2, 'hight': 28, 'width': 24, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TDrive'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64, 

                                     'patch_size': 2, 'hight': 32, 'width': 32, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['BikeNYC2'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64, 

                                       'patch_size': 2, 'hight': 16, 'width': 8, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['TaxiNYC2'] = {'seq_in': 6, 'seq_out': 6, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64, 

                                       'patch_size': 2, 'hight': 16, 'width': 12, 'nodes': None, 'data_type': 'Grid'}

all_configs['OurModel']['PEMS03'] = {'seq_in': 12, 'seq_out': 12, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64,

                                    'patch_size':4,'hight': None, 'width': None,'nodes':358, 'data_type':'Topology'}

all_configs['OurModel']['PEMS04'] = {'seq_in': 12, 'seq_out': 12, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64,

                                    'patch_size':4,'hight': None, 'width': None,'nodes':307, 'data_type':'Topology'}

all_configs['OurModel']['PEMS07'] = {'seq_in': 12, 'seq_out': 12, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64,

                                    'patch_size':4,'hight': None, 'width': None,'nodes':883, 'data_type':'Topology'}

all_configs['OurModel']['PEMS08'] = {'seq_in': 12, 'seq_out': 12, 'in_channel': 1, 'main_layers': 4,'hidden_channel': 64,

                                    'patch_size':4,'hight': None, 'width': None,'nodes':170, 'data_type':'Topology'}