from src.configs.models import Model, RunConfig, Step, StepRevision, StepRevisionPool

model = Model.gemini_2_5

gemini_config_prod = RunConfig(
    final_follow_model=model,
    final_follow_times=5,
    max_concurrent_tasks=120,
    steps=[
        Step(
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=10,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)

# Alternative config with flash lite model for faster/cheaper runs
model_flash = Model.gemini_2_5_flash_lite

gemini_config_flash = RunConfig(
    final_follow_model=model_flash,
    final_follow_times=3,
    max_concurrent_tasks=120,
    steps=[
        Step(
            instruction_model=model_flash,
            follow_model=model_flash,
            times=3,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model_flash,
            follow_model=model_flash,
            times=3,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model_flash,
            follow_model=model_flash,
            times=10,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=3,
            instruction_model=model_flash,
            follow_model=model_flash,
            times=3,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)

