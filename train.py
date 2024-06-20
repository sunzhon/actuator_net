from utils import Train
from glob import glob
import os
import pandas as pd
import pickle


class DataProcess():
    def __init__(self, 
            raw_filepath=None,
            start_index =3100, # valid of data row start
            end_index=8600, # valid of data row end
            valid_motors=[2,3,4] # valid of data column group
            ):
        self.raw_filepath = raw_filepath
        # checking data file path
        if os.path.isdir(raw_filepath):
            self.pd_data = pd.read_csv(os.path.join(raw_filepath,"controlfile_data.csv"), sep="\t",index_col=0, header=0)
            self.datafile_dir = raw_filepath
        else:
            if(os.path.exists(raw_filepath)):
                self.pd_data = pd.read_csv(os.path.join(raw_filepath), sep="\t",index_col=0, header=0)
                self.datafile_dir = os.path.dirname(raw_filepath)
            else:
                raise "Data file does not exist, please check the data file path!"

        self.data_row_num = self.pd_data.shape[0]
        self.start_index = start_index
        self.end_index = end_index
        self.valid_motors = valid_motors


    def process_data(self):
        """
        Process data
        """
        processed_data =[] 
        for step_idx in range(self.start_index, min(self.data_row_num,self.end_index) - self.start_index):
            processed_data.append({
            "motor_pos_target":[self.pd_data["jcm_"+str(joint_idx)][step_idx] for joint_idx in self.valid_motors],
            "motor_pos":[self.pd_data["jointPosition_"+str(joint_idx)][step_idx] for joint_idx in self.valid_motors],
            "motor_vel":[self.pd_data["jointVelocity_"+str(joint_idx)][step_idx] for joint_idx in self.valid_motors],
            "motor_tor":[(self.pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in self.valid_motors],
            #"torques":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in valid_motors],
            #"tau_est":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in valid_motors], #mA to A, to Nm
            #"tau_est":[0.0 for joint_idx in valid_motors],
            })

        result_datas = {'motor_data':[processed_data]}

        with open(os.path.join(self.datafile_dir,'motor_data.pkl'), 'wb') as f:
            pickle.dump(result_datas, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Process ambot motor data, data file saved at: {:}".format(os.path.join(self.datafile_dir,'motor_data.pkl')))





if __name__=="__main__":
    load_pretrained_model = True
    datafile_dir = "./app/"
    dp = DataProcess(datafile_dir,
                    start_index=3000,
                    end_index=10000,
                    valid_motors=[2,3,4]
                    )
    dp.process_data()
    training = Train(
            motor_num=3,
            data_sample_freq=100,
            datafile_dir = datafile_dir,
            load_pretrained_model = load_pretrained_model
            )
    training.load_data()
    training.training_model()
    training.eval_model()

