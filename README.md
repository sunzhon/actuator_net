# Welcome to actuator_net 

actuator_net is a project to model actuators using  artificial neural network with deep-learning algorithm.

## Repository Structure

- train.py contains the code to process dataset collected from actuators
- utils.py contains the code to build a MLP model and train the model
- eval.py contains the code to evaluate the trained model

## Installation

1. Prepare a python virtual env with python3.8 or later
2. install pytorch and pandas by `pip install pytorch pandas`



## Usage

1. Clone this repository
2. Collected labeled dataset from actuators/motors
3. Process dataset
4. Training Model
5. Evaluation model



### Raw data storage

The raw data collected from actuators are format as a table and saved at a csv file. The file name is controlfile_data.csv.

Each column of the table represents a variable, such as a joint position, joint velocity, etc. each row indicates a step. For instance,

```python
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


```







