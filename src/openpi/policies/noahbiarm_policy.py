import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_noahbiarm_example() -> dict:
    """Creates a random input example for the Maniskill policy."""
    QPOS_DIM = 9
    return {
        "observation/state": np.random.rand(QPOS_DIM),
        "observation/base_camera": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "observation/head_camera": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "observation/hand_camera": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NoahBiArmInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0


    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
       
        augmented_state = torch.cat(
            (
                torch.tensor(data["observation/state"]),
                torch.tensor(data["observation/tcp_pose"]),
                torch.tensor(data["observation/obj_pose"]),
                torch.tensor(data["observation/rack_pose"]),
            )
        )
        state = transforms.pad_to_dim(augmented_state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        
        # base_image = _parse_image(data["observation/base_camera"])
        hand_image = _parse_image(data["observation/hand_camera"])
        head_image = _parse_image(data["observation/head_camera"])


        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "right_wrist_0_rgb": hand_image,
                "left_wrist_0_rgb": np.zeros_like(head_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = data["actions"]              # (N, A)
            N = actions.shape[0]

            # Pose fields to append
            pose_keys = ["tcp_pose", "obj_pose", "rack_pose"]

            # Expand each pose tensor from (B1, B2, …) → (N, B1, B2, …)
            expanded = []
            for key in pose_keys:
                pose = torch.tensor(data[f"observation/{key}"])
                expanded.append(
                    pose.unsqueeze(0)                       # (1, B1, B2, …)
                        .expand(N, *pose.shape)             # (N, B1, B2, …)
                )

            # Concatenate along the feature axis
            augmented = torch.cat([actions, *expanded], dim=1)

            # Pad to fixed action_dim
            padded = transforms.pad_to_dim(augmented, self.action_dim)
            inputs["actions"] = padded

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        else:
            # @sajjad: we assume that the data_config.prompt_from_task=True (see config.py for details)
            raise NotImplementedError

        return inputs


@dataclasses.dataclass(frozen=True)
class NoahBiArmOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    robot_action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Maniskill, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.

        return {"actions": np.asarray(data["actions"][:, :self.robot_action_dim])}