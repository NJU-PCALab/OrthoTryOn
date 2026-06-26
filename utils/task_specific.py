TASKS = ("vton", "vtoff", "pose")
TASK_IDS = {task: index for index, task in enumerate(TASKS)}
CONDITION_ORDER = ("ref_image", "cloth_image", "pose_image")
CONDITION_MODALITY_IDS = {"ref_image": 2, "cloth_image": 3, "pose_image": 4}
TASK_CONDITIONS = {
    "vton": ("ref_image", "cloth_image", "pose_image"),
    "vtoff": ("ref_image", "cloth_image"),
    "pose": ("ref_image", "pose_image"),
}
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATES = {
    "vton": (
        "<|im_start|>system\n"
        "As an image editing expert, first analyze the content and attributes of the first input image (the model). "
        "Then, examine the second input image (the target garment) and the third input image (the target pose). "
        "Based on the user's editing instructions and these visual inputs, clearly and precisely determine how to modify the model image. "
        "Ensure that the garment is applied correctly and the model adopts the target pose, while keeping all other non-specified aspects consistent with the original.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
    ),
    "vtoff": (
        "<|im_start|>system\n"
        "As an image editing expert, first analyze the content and visual attributes of the garment worn by the model in the first input image."
        "Then, examine the second input image as a reference for flat-lay presentation style and background appearance."
        "Based on the user's editing instructions and these visual inputs, clearly and precisely determine how to reconstruct the garment as a standalone, flat-laid item."
        "Ensure that the reconstructed garment strictly preserves all original attributes from the first image, including texture, pattern, color, material, and structural details. "
        "The second image should be used only to guide background style, layout, lighting, and flat-lay cleanliness, and must not influence the garment's identity, shape, or appearance. "
        "Ensure the final reconstructed garment is free from body-induced distortions and occlusions, while keeping all non-specified aspects consistent with the source garment.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
    ),
    "pose": (
        "<|im_start|>system\n"
        "As an image editing expert, first analyze the content and attributes of the first input image (the source appearance/model image). "
        "Then, based on the user's editing instructions and the second input image (the target pose), clearly and precisely determine how to generate a new image. "
        "The goal is to apply the pose from the second image to the person in the first image, ensuring that the person's identity, clothing, and background remain consistent with the first original image.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
    ),
}
DEFAULT_PROMPTS = {
    "vton": "Transfer the target garment onto the model.",
    "vtoff": "Change the worn garment to a flat-laid garment.",
    "pose": "Transfer the person to the target pose.",
}


def normalize_task(task):
    task = str(task).strip().lower()
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Expected one of {TASKS}.")
    return task


def get_task_id(task):
    return TASK_IDS[normalize_task(task)]


def get_task_conditions(task):
    return TASK_CONDITIONS[normalize_task(task)]


def get_prompt_template(task):
    return PROMPT_TEMPLATES[normalize_task(task)]


def get_default_prompt(task):
    return DEFAULT_PROMPTS[normalize_task(task)]


def resolve_negative_task(task, sampler="cfg", negative_task=None):
    normalize_task(task)
    if negative_task is not None:
        return normalize_task(negative_task)
    sampler = str(sampler or "cfg").strip().lower()
    if sampler in {"", "cfg"}:
        return None
    if sampler.endswith("_cfg"):
        return normalize_task(sampler[:-4])
    raise ValueError("sampler must be 'cfg' or '<task>_cfg'.")


def required_conditions(*tasks):
    names = set()
    for task in tasks:
        if task is not None:
            names.update(get_task_conditions(task))
    return tuple(name for name in CONDITION_ORDER if name in names)
