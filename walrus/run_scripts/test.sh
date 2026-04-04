module load conda
conda activate walrus

# Launch the training script

torchrun --nnodes=1 --nproc_per_node=4 train.py server=local distribution=fsdp 

# model=isotropic_model name=Walrus_ft_example trainer=globalnorm trainer.grad_acc_steps=1 server=gpuxl optimizer=adam optimizer.lr=1e-4 logger.wandb_project_name="walrus_Finetuning_Runs" \
# trainer.enable_amp=False model.gradient_checkpointing_freq=2 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=2 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1   \
# model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=1408 model.groups=16 model.processor_blocks=40 model.drop_path=0.0 \
# model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 ++trainer.epsilon=1e-8 \
# model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=20 trainer.max_rollout_steps=60 \
# lr_scheduler=inv_sqrt_w_sqrt_ramps_longer trainer.val_frequency=5 trainer.rollout_val_frequency=5 data.module_parameters.min_dt_stride=1 data.module_parameters.max_dt_stride=1 \
# trainer.prediction_type="delta" data=euler_multi_quadrants_openBC trainer.max_epoch=51 data_workers=10 model.override_dimensionality=0 auto_resume=True \
# checkpoint=finetune experiment=finetune_example ++model.use_periodic_fixed_jitter=True ++model.input_field_drop=0 ++trainer.skip_spectral_metrics=True \
# finetuning_mods=all ++experiment_dir=/mnt/home/polymathic/ceph/walrus_logging/runs ++data.module_parameters.start_rollout_valid_output_at_t=17 \
