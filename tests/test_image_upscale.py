from merceka_core.image import _fal_upscale_payload


def test_esrgan_payload_uses_requested_scale():
  payload = _fal_upscale_payload("fal-ai/esrgan", "data:image/png;base64,abc", 1.75)

  assert payload == {
    "image_url": "data:image/png;base64,abc",
    "scale": 1.75,
    "model": "RealESRGAN_x4plus",
    "output_format": "png",
  }


def test_aura_sr_payload_uses_fixed_4x_checkpoint():
  payload = _fal_upscale_payload("fal-ai/aura-sr", "data:image/png;base64,abc", 2.0)

  assert payload == {
    "image_url": "data:image/png;base64,abc",
    "upscale_factor": 4,
    "overlapping_tiles": False,
    "checkpoint": "v2",
  }
