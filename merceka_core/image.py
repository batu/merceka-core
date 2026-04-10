"""Image generation and editing via AI APIs."""

__all__ = ["generate_image", "edit_image", "inpaint"]

import base64
import io
import json
import os

import httpx
from PIL import Image


def _image_to_base64_uri(image: Image.Image) -> str:
  """Convert a PIL Image to a base64 PNG data URI."""
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  encoded = base64.b64encode(buf.getvalue()).decode("ascii")
  return f"data:image/png;base64,{encoded}"


def _base64_to_image(data_uri: str) -> Image.Image:
  """Decode a base64 data URI to a PIL Image."""
  _, encoded = data_uri.split(",", 1)
  return Image.open(io.BytesIO(base64.b64decode(encoded)))


def generate_image(
  prompt: str,
  *,
  model: str = "google/gemini-3.1-flash-image-preview",
  aspect_ratio: str = "1:1",
  image_size: str = "1K",
) -> Image.Image:
  """Generate an image from a text prompt via OpenRouter.

  Args:
    prompt: Text description of the image to generate.
    model: OpenRouter model ID.
    aspect_ratio: Aspect ratio string (e.g., "1:1", "1:2", "16:9").
    image_size: Resolution tier ("1K", "2K", "4K").

  Returns:
    PIL Image in RGB mode.
  """
  api_key = os.environ.get("OPENROUTER_API_KEY")
  if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")

  payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "modalities": ["image", "text"],
    "image_config": {
      "aspect_ratio": aspect_ratio,
      "image_size": image_size,
    },
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://openrouter.ai/api/v1/chat/completions",
      json=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    data = response.json()

  try:
    image_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenRouter response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e

  return _base64_to_image(image_url).convert("RGB")


def edit_image(
  image: Image.Image,
  prompt: str,
  *,
  model: str = "google/gemini-3.1-flash-image-preview",
) -> Image.Image:
  """Send one image + text prompt, get back a modified image.

  Unlike inpaint(), this sends a single image as context — no mask.
  The model sees the image and follows the text instructions.

  Args:
    image: Reference image (RGB).
    prompt: Instructions for what to do with the image.
    model: OpenRouter model ID.

  Returns:
    PIL Image in RGB mode.
  """
  api_key = os.environ.get("OPENROUTER_API_KEY")
  if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")

  image_uri = _image_to_base64_uri(image.convert("RGB"))
  original_size = image.size

  w, h = image.size
  if w == h:
    ar = "1:1"
  elif w > h:
    ar = "16:9" if w / h > 1.5 else "4:3"
  else:
    ar = "9:16" if h / w > 1.5 else "3:4"
  img_size = "1K" if max(w, h) <= 1024 else "2K"

  payload = {
    "model": model,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_uri}},
      ],
    }],
    "modalities": ["image", "text"],
    "image_config": {
      "aspect_ratio": ar,
      "image_size": img_size,
    },
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://openrouter.ai/api/v1/chat/completions",
      json=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text[:500]}")
    data = response.json()

  try:
    result_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenRouter response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e

  result = _base64_to_image(result_url).convert("RGB")
  if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result


def inpaint(
  image: Image.Image,
  mask: Image.Image,
  prompt: str,
  *,
  model: str = "fal-ai/flux-pro/v1/fill",
) -> Image.Image:
  """Fill masked region of image using inpainting.

  Supports two provider types:
  - fal.ai models (model starts with "fal-ai/"): true mask-based inpainting
  - OpenRouter models (model starts with "google/" or "openai/"): prompt-directed
    editing with image + mask sent as visual context

  Args:
    image: Source image (RGB).
    mask: Mask image — white = edit, black = preserve. Must match image dimensions.
    prompt: Description of what to generate in the masked area.
    model: Model identifier. fal.ai paths or OpenRouter model IDs.

  Returns:
    PIL Image in RGB mode with the masked region filled.
  """
  if image.size != mask.size:
    raise ValueError(f"Image size {image.size} doesn't match mask size {mask.size}")

  if model.startswith("fal-ai/"):
    return _inpaint_fal(image, mask, prompt, model)
  else:
    return _inpaint_openrouter(image, mask, prompt, model)


def _inpaint_fal(
  image: Image.Image, mask: Image.Image, prompt: str, model: str,
) -> Image.Image:
  """Inpaint via fal.ai (true mask-based)."""
  api_key = os.environ.get("FAL_KEY")
  if not api_key:
    raise RuntimeError("FAL_KEY not set in environment")

  image_uri = _image_to_base64_uri(image.convert("RGB"))
  mask_uri = _image_to_base64_uri(
    mask.convert("L").point(lambda x: 255 if x > 128 else 0).convert("RGB")
  )

  payload = {
    "image_url": image_uri,
    "mask_url": mask_uri,
    "prompt": prompt,
    "num_images": 1,
    "output_format": "png",
    "safety_tolerance": 5,
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      f"https://fal.run/{model}",
      json=payload,
      headers={
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
      },
    )
    if response.status_code != 200:
      raise RuntimeError(f"fal.ai API error {response.status_code}: {response.text[:500]}")
    result_data = response.json()

  try:
    result_image_url = result_data["images"][0]["url"]
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in fal.ai response: {e}\nResponse: {json.dumps(result_data, indent=2)[:500]}"
    ) from e

  with httpx.Client(timeout=60) as client:
    img_response = client.get(result_image_url)
    img_response.raise_for_status()

  result = Image.open(io.BytesIO(img_response.content)).convert("RGB")
  # Resize to match input dimensions (fal.ai may return different size)
  if result.size != image.size:
    result = result.resize(image.size, Image.Resampling.LANCZOS)
  return result


def _inpaint_openrouter(
  image: Image.Image, mask: Image.Image, prompt: str, model: str,
) -> Image.Image:
  """Inpaint via OpenRouter (prompt-directed with image + mask as visual context)."""
  api_key = os.environ.get("OPENROUTER_API_KEY")
  if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")

  image_uri = _image_to_base64_uri(image.convert("RGB"))
  mask_uri = _image_to_base64_uri(
    mask.convert("L").point(lambda x: 255 if x > 128 else 0).convert("RGB")
  )

  # Derive aspect ratio and size from input image
  w, h = image.size
  if w == h:
    ar = "1:1"
  elif w > h:
    ar = "16:9" if w / h > 1.5 else "4:3"
  else:
    ar = "9:16" if h / w > 1.5 else "3:4"
  img_size = "1K" if max(w, h) <= 1024 else "2K"
  original_size = image.size

  edit_prompt = (
    f"I am providing two images. The first is a scene. The second is a black and white mask — "
    f"the white areas indicate where to edit.\n\n"
    f"Edit the scene by adding the following in the white masked area, "
    f"keeping everything in the black area completely unchanged:\n\n{prompt}\n\n"
    f"The result must look natural and match the style of the original scene. "
    f"Return only the edited image."
  )

  payload = {
    "model": model,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": edit_prompt},
        {"type": "image_url", "image_url": {"url": image_uri}},
        {"type": "image_url", "image_url": {"url": mask_uri}},
      ],
    }],
    "modalities": ["image", "text"],
    "image_config": {
      "aspect_ratio": ar,
      "image_size": img_size,
    },
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://openrouter.ai/api/v1/chat/completions",
      json=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text[:500]}")
    data = response.json()

  try:
    result_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenRouter response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e

  result = _base64_to_image(result_url).convert("RGB")
  # Resize to match input dimensions (OpenRouter may return different size)
  if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result
