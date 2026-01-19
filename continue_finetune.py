#!/usr/bin/env python3
import os
import sys
import torch

# Add the repo to path
sys.path.insert(0, '/root/repo/openvla-oft')
sys.path.insert(0, '/root/repo/openvla-oft/vla-scripts')

# Import the finetune module
import finetune as finetune_module
from finetune import *

# Paths to your existing checkpoints
ADAPTER_PATH = "/root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_20260117-122720/vla_cdpr_adapter"
ACTION_HEAD_PATH = "/root/repo/VLA_CDPR/oft_cdpr_ckpts/cdpr_finetune_20260117-122720/action_head_cdpr.pt"

# Save the original load_checkpoint function
original_load_checkpoint = load_checkpoint

def patched_load_checkpoint(module_name: str, vla_path: str, resume_step: Optional[int] = None) -> dict:
    """
    Patched version of load_checkpoint that handles missing checkpoints gracefully.
    """
    # For action_head, load from our custom path
    if module_name == "action_head" and os.path.exists(ACTION_HEAD_PATH):
        print(f"[PATCH] Loading action head from custom path: {ACTION_HEAD_PATH}")
        return torch.load(ACTION_HEAD_PATH, map_location="cpu")
    
    # For other modules, try to load from original location
    try:
        return original_load_checkpoint(module_name, vla_path, resume_step)
    except FileNotFoundError as e:
        # If checkpoint doesn't exist, return empty dict
        print(f"[INFO] No checkpoint found for {module_name}, using random initialization")
        return {}

# Replace the original function
finetune_module.load_checkpoint = patched_load_checkpoint

# Monkey-patch the finetune function
original_finetune = finetune_module.finetune

def log_metrics_to_tensorboard(metrics, prefix, step, tb_writer) -> None:
    """
    Log metrics to TensorBoard only.
    """
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability
        if name == "loss_value":
            tb_writer.add_scalar(f"{prefix}/Loss", value, step)
        # Keep other metrics as is
        else:
            tb_writer.add_scalar(f"{prefix}/{name.replace('_', ' ').title()}", value, step)
    
    tb_writer.flush()  # Ensure data is written
    
    # Print to console for debugging
    console_str = f"[Step {step}] {prefix}: "
    console_items = []
    for name, value in metrics.items():
        if name == "loss_value":
            console_items.append(f"Loss: {value:.4f}")
        else:
            console_items.append(f"{name.replace('_', ' ').title()}: {value:.4f}")
    console_str += ", ".join(console_items)
    print(console_str, flush=True)

@draccus.wrap()
def patched_finetune(cfg: FinetuneConfig) -> None:
    print("=" * 60)
    print("CONTINUING TRAINING FROM EXISTING CHECKPOINT")
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Action head path: {ACTION_HEAD_PATH}")
    print(f"New learning rate: {cfg.learning_rate}")
    print("=" * 60)
    cfg.resume = False
    cfg.resume_step = None
    
    # Run the original function but patch the LoRA loading
    nvml_handle = None
    if is_rank0():
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            print("[NVML] enabled", flush=True)
        except Exception as e:
            print(f"[NVML] disabled: {e}", flush=True)
            nvml_handle = None
    
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    print(
        f"[rank={distributed_state.process_index} local_rank={distributed_state.local_process_index}] "
        f"cuda.current_device={torch.cuda.current_device()} "
        f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True
    )
    torch.cuda.empty_cache()

    class DummyLogger:
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
        
    if distributed_state.is_main_process:
        os.environ["WANDB_DISABLED"] = "true"
        wandb = DummyLogger()
    else:
        wandb = DummyLogger()

    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    vla = vla.to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup with adapter loading
    if cfg.use_lora and os.path.exists(ADAPTER_PATH):
        print(f"[LOAD] Loading LoRA adapter from: {ADAPTER_PATH}")
        vla = PeftModel.from_pretrained(vla, ADAPTER_PATH, is_trainable=True)
        vla = vla.to(device_id)
        vla.print_trainable_parameters()
    else:
        raise RuntimeError(f"Adapter path not found: {ADAPTER_PATH}")

    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            try:
                state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
                load_state_dict_skip_action_head(vla, state_dict)
            except Exception as e:
                print(f"[WARNING] Could not load vision_backbone checkpoint: {e}")
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    vla = wrap_ddp(vla, device_id, find_unused=True)

    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
            },
            to_bf16=True,
        )
        if os.path.exists(ACTION_HEAD_PATH):
            print(f"[LOAD] Loading action head from: {ACTION_HEAD_PATH}")
            sd = torch.load(ACTION_HEAD_PATH, map_location="cpu")
            action_head.module.load_state_dict(sd, strict=True)
        else:
            raise RuntimeError(f"Action head path not found: {ACTION_HEAD_PATH}")


    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )
        if os.path.exists(ACTION_HEAD_PATH):
            print(f"[LOAD] Loading action head from: {ACTION_HEAD_PATH}")
            sd = torch.load(ACTION_HEAD_PATH, map_location="cpu")
            action_head.module.load_state_dict(sd, strict=True)
        else:
            raise RuntimeError(f"Action head path not found: {ACTION_HEAD_PATH}")


    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    if distributed_state.is_main_process:
        print(f"==========================================")
        print(f"TensorBoard logs directory: {run_dir}")
        print(f"To view logs: tensorboard --logdir={run_dir}")
        print(f"==========================================")
        
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    tb_writer = None
    if is_rank0():
        tb_logdir = Path(run_dir)
        tb_logdir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_logdir), flush_secs=10)
        print(f"[TensorBoard] Writing events to: {tb_logdir}", flush=True)
        
    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=not is_rank0()) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            
            log_step = gradient_step_idx if not cfg.resume else (cfg.resume_step or 0) + gradient_step_idx
            
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

                if is_rank0() and tb_writer is not None and gradient_step_idx % cfg.wandb_log_freq == 0:
                    # Log loss + other metrics
                    log_metrics_to_tensorboard(smoothened_metrics, "VLA Train", log_step, tb_writer)

                    # Log LR too (keep what you already had)
                    # tb_writer.add_scalar("VLA Train/Learning Rate", scheduler.get_last_lr()[0], log_step)
                    allocated_gb = torch.cuda.memory_allocated() / 1024**3
                    reserved_gb  = torch.cuda.memory_reserved() / 1024**3
                    max_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3

                    tb_writer.add_scalar("cuda/mem_allocated_gb", allocated_gb, log_step)
                    tb_writer.add_scalar("cuda/mem_reserved_gb",  reserved_gb,  log_step)
                    tb_writer.add_scalar("cuda/max_allocated_gb", max_alloc_gb, log_step)
                    tb_writer.flush()

            if gradient_step_idx >= cfg.max_steps:
                break
            
            if is_rank0() and gradient_step_idx % cfg.save_freq == 0:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ckpt_dir = Path(cfg.run_root_dir) / f"cdpr_finetune_step{gradient_step_idx}_{timestamp}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                vla_adapter_dir = ckpt_dir / "vla_cdpr_adapter"
                print(f"[SAVE] Saving VLA adapters to {vla_adapter_dir}", flush=True)
                vla.module.save_pretrained(vla_adapter_dir)

                if cfg.use_l1_regression:
                    ah_ckpt_path = ckpt_dir / "action_head_cdpr.pt"
                    print(f"[SAVE] Saving action head weights to {ah_ckpt_path}", flush=True)
                    torch.save(action_head.module.state_dict(), ah_ckpt_path)

            
    # Create a unique subdir under run_root_dir using timestamp
    if is_rank0():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_dir = Path(cfg.run_root_dir) / f"cdpr_finetune_{timestamp}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if tb_writer is not None:
            tb_writer.close()
            print("[TensorBoard] Writer closed")

        vla_adapter_dir = ckpt_dir / "vla_cdpr_adapter"
        print(f"[SAVE] Saving VLA adapters to {vla_adapter_dir}", flush=True)
        vla.module.save_pretrained(vla_adapter_dir)

        if cfg.use_l1_regression:
            ah_ckpt_path = ckpt_dir / "action_head_cdpr.pt"
            print(f"[SAVE] Saving action head weights to {ah_ckpt_path}", flush=True)
            torch.save(action_head.module.state_dict(), ah_ckpt_path)
    else:
        # non-rank0 still should close writer if it exists, but don't write files
        if tb_writer is not None:
            tb_writer.close()
            
    if tb_writer is not None:
        tb_writer.close()

    wandb.finish()

# Replace the original finetune function with our patched version
finetune_module.finetune = patched_finetune

if __name__ == "__main__":
    # Run the patched function
    patched_finetune()