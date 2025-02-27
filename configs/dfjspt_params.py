

time_window_size = 300
current_window = 0

# training params
as_test = False
framework = "torch"
local_mode = False
use_tune = True
use_custom_loss = True
il_loss_weight = 10.0
stop_iters = 5000
stop_timesteps = 100000000000
stop_reward = 2

max_n_machines = 12
min_prcs_time = 1
max_prcs_time = 99
n_machines_is_fixed = True
# n_machines = 15
n_machines = max_n_machines
is_fully_flexible = False
min_compatible_machines = 1
time_for_compatible_machines_are_same = False
time_viration_range = 5

max_n_transbots = 6
min_tspt_time = 2
max_tspt_time = 4
loaded_transport_time_scale = 1.5
n_transbots_is_fixed = True
# n_transbots = 3
n_transbots = max_n_transbots

all_machines_are_perfect = True
min_quality = 1

max_n_jobs = 20
n_jobs_is_fixed = True
# n_jobs = 15
n_jobs = max_n_jobs
n_operations_is_n_machines = False
min_n_operations = int(n_machines*0.8)
max_n_operations = int(n_machines*1.2)
consider_job_insert = True
new_arrival_jobs = 3
earliest_arrive_time = 30
latest_arrive_time = 300

normalized_scale = max_n_operations * max_prcs_time

n_instances = 1200
n_instances_for_training = 1000
n_instances_for_evaluation = 100
n_instances_for_testing = 100
instance_generator_seed = 1000
layout_seed = 0

# env params
perform_left_shift_if_possible = True

# instance selection params
randomly_select_instance = False
current_instance_id = 0
imitation_env_count = 0
env_count = 0


# render params
JobAsAction = True
gantt_y_axis = "nJob"
drawMachineToPrcsEdges = True
default_visualisations = None




