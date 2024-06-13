from utils import train_actuator_network_and_plot_predictions
from glob import glob
import os
import pandas as pd

log_dir_root = "../../../logs/"

#log_dir = "example_experiment/2022/11_01/16_01_50_0"
log_dir = "example_experiment/ambotv1"

# Evaluates the existing actuator network by default
# load_pretrained_model = True
# actuator_network_path = "../../resources/actuator_nets/unitree_go1.pt"

# Uncomment these lines to train a new actuator network
load_pretrained_model = False
#actuator_network_path = "../../resources/actuator_nets/unitree_go1_new.pt"
actuator_network_path = "../../../resources/actuator_nets/actuator.pt"


log_dirs = glob(f"{log_dir_root}{log_dir}/", recursive=True)

if len(log_dirs) == 0: raise FileNotFoundError(f"No log files found in {log_dir_root}{log_dir}/")

def process_actuator_data(raw_filepath):
    """
    Process the raw data collected from actuators, and then save them in a log.pkl file

    """
    pd_data = pd.read_csv(os.path.join(raw_filepath,"controlfile_data.csv"), sep="\t",index_col=0, header=0)

    processed_data =[] 
    start_index = 3100
    end_index = 8600
    joint_indexs = [2, 3, 4, 5,  6, 7, 8, 9,  10, 11, 12, 13] # 12 joints
    for step_idx in range(start_index, pd_data.shape[0]-start_index):
        processed_data.append({
        "joint_pos_target":[pd_data["jcm_"+str(joint_idx)][step_idx] for joint_idx in joint_indexs],
        "joint_pos":[pd_data["jointPosition_"+str(joint_idx)][step_idx] for joint_idx in joint_indexs],
        "joint_vel":[pd_data["jointVelocity_"+str(joint_idx)][step_idx] for joint_idx in joint_indexs],
        "torques":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in joint_indexs],
        "tau_est":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in joint_indexs], #mA to A, to Nm
        #"tau_est":[0.0 for joint_idx in joint_indexs],
        })

    result_datas = {'hardware_closed_loop':[0,processed_data]}

    import pickle

    with open(os.path.join(raw_filepath,'log.pkl'), 'wb') as handle:
        pickle.dump(result_datas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Process ambot motor data, data file saved at: {:}".format(os.path.join(raw_filepath,'log.pkl')))



process_actuator_data(log_dirs[0])

for log_dir in log_dirs:
    try:
        train_actuator_network_and_plot_predictions(log_dir[:11], log_dir[11:], actuator_network_path=actuator_network_path, load_pretrained_model=load_pretrained_model)
    except FileNotFoundError:
        print(f"Couldn't find log.pkl in {log_dir}")
    except EOFError:
        print(f"Incomplete log.pkl in {log_dir}")


