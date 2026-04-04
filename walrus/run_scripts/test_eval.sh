export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN


# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
# export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
# module load cuda/12.4 cudnn/9.1.0.70-cuda12 nccl/2.21.5-1+cuda12.4
# source /mnt/home/mmccabe/venvs/mamba_well/bin/activate
module load conda
conda activate walrus

# Launch the training script
# Folder structures defined by the train script can enter validation just by pointing to a config and weight folder. The rest of the settings are telling the run to validate differently than during training.
python3 train.py --config-path="/pscratch/sd/k/khegazy/projects/pde/walrus/experiments/test/" --config-name="extended_config.yaml" ++distribution.distribution_type=local ++validation_mode=True ++folder_override="/pscratch/sd/k/khegazy/projects/pde/walrus/experiments/test/" ++checkpoint_override="/pscratch/sd/k/khegazy/projects/pde/walrus/experiments/test/checkpoints" "++trainer.validation_suite=[{_target_:the_well.benchmark.metrics.NRMSE},{_target_:the_well.benchmark.metrics.VRMSE},{_target_:the_well.benchmark.metrics.PearsonR}]"   "++trainer.validation_trajectory_metrics=[]"   "++trainer.batch_aggregation_fns=[torch.mean,torch.median,torch.std]" "++data.module_parameters.max_rollout_steps=20"  "++data.module_parameters.start_rollout_valid_output_at_t=17" "++trainer.max_rollout_steps=20" "++data.well_base_path=/global/cfs/cdirs/m4790/Data/well/datasets/" "++trainer.dump_prediction_to_disk=True"
