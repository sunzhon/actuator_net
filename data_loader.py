from train import Train
from glob import glob
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler



class DataProcess():
    def __init__(self, 
            raw_filepath=None,
            data_start =3100, # valid of data row start
            data_end=8600, # valid of data row end
            motors=[2,3,5], # valid of data column group
            input_data_name = ["jcm","jointPosition"],
            output_data_name = ["jointCurrent"],
            scaler="standard"
            ):
        self.raw_filepath = raw_filepath
        # checking data file path
        if os.path.isdir(raw_filepath):
            self.pd_data = pd.read_csv(
                    os.path.join(raw_filepath,"controlfile_data.csv"), 
                    sep="\t",index_col=0, header=0)
            self.datafile_dir = raw_filepath
        else:
            if(os.path.exists(raw_filepath)):
                self.pd_data = pd.read_csv(os.path.join(raw_filepath), sep="\t",index_col=0, header=0)
                self.datafile_dir = os.path.dirname(raw_filepath)
            else:
                raise "Data file does not exist, please check the data file path!"

        self.data_row_num = self.pd_data.shape[0]
        self.data_start = data_start
        self.data_end = data_end
        self.motors = motors
        self.input_data_name = input_data_name
        self.output_data_name = output_data_name
        self.scaler = scaler


    def process_data(self):
        """
        Process data
        """
        #1) select dataset column
        input_data_list =[]
        output_data_list =[]
        for idx in self.motors:
            input_data_list.append(self.pd_data.loc[:,[tmp+"_"+str(idx) for tmp in self.input_data_name]])
            output_data_list.append(self.pd_data.loc[:,[tmp+"_"+str(idx) for tmp in self.output_data_name]])

        input_data = np.concatenate([value.values for value in input_data_list],axis=0)
        output_data = np.concatenate([value.values for value in output_data_list],axis=0)

        raw_dataset = np.concatenate([input_data,output_data],axis=1)

        #2) normalizate data
        #i) normalization method
        if self.scaler=='standard':
            scaler=StandardScaler()
        if self.scaler=='minmax':
            scaler=MinMaxScaler()
        if self.scaler=='robust':
            scaler=RobustScaler()

        try:
            scaler.fit(raw_dataset)
            scaled_raw_dataset = scaler.transform(raw_dataset.astype(np.float32))
        except Exception as e:
            print(e)

        scaled_input_data = scaled_raw_dataset[:,:-1]
        scaled_output_data = scaled_raw_dataset[:,-1:]
        processed_data = {"input_data": scaled_input_data, "output_data": scaled_output_data}

        with open(os.path.join(self.datafile_dir,'motor_data.pkl'), 'wb') as f:
            pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.datafile_dir,'scaler.pkl'),'wb') as f:
            pickle.dump(scaler,f)

        print("Process ambot motor data, data file saved at: {:}".format(os.path.join(self.datafile_dir,'motor_data.pkl')))
        print("Scaler file saved at: {:}".format(os.path.join(self.datafile_dir,'scaler.pkl')))

        """
        processed_data =[] 
        for step_idx in range(self.data_start, min(self.data_row_num,self.data_end) - self.data_start):
            processed_data.append({
            "motor_pos_target":[self.pd_data["jcm_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_pos":[self.pd_data["jointPosition_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_vel":[self.pd_data["jointVelocity_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_tor":[(self.pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in self.motors],
            #"torques":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in motors],
            #"tau_est":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in motors], #mA to A, to Nm
            #"tau_est":[0.0 for joint_idx in motors],
            })

        result_datas = {'motor_data':[processed_data]}

        with open(os.path.join(self.datafile_dir,'motor_data.pkl'), 'wb') as f:
            pickle.dump(result_datas, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Process ambot motor data, data file saved at: {:}".format(os.path.join(self.datafile_dir,'motor_data.pkl')))

        """





if __name__=="__main__":
    load_pretrained_model = False
    datafile_dir = "./app/"
    dp = DataProcess(datafile_dir,
                    data_start=3000,
                    data_end=10000,
                    motors=[2,3,5, 6,7,9, 10,11,13, 14,15,17],
                    input_data_name=["jcm","jointPosition"],
                    output_data_name=["jointCurrent"],
                    )
    dp.process_data()
