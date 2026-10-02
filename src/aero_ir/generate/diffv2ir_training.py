"""Narrow compatibility adapter for pinned DiffV2IR and Lightning1.9.5.

This is not a training launcher or scientific authorization. The caller must
verify/import the pinned upstream in the isolated environment before creation.
No upstream source, model parameter names or diffusion objective are changed.
"""

from importlib.metadata import version


class Lightning19BatchHook:
    """Supply the old single-loader index omitted by Lightning1.9.5."""

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        return super().on_train_batch_start(batch, batch_idx, dataloader_idx)


def create_training_model(**params):
    """Config factory used only after official_modules() verifies upstream imports."""
    if version("pytorch-lightning") != "1.9.5":
        raise RuntimeError("DiffV2IR training adapter requires the verified Lightning1.9.5 runtime")
    from ldm.models.diffusion.ddpm_DiffV2IR import LatentDiffusion

    class Lightning19LatentDiffusion(Lightning19BatchHook, LatentDiffusion):
        pass

    return Lightning19LatentDiffusion(**params)
