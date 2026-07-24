"""Image generation and editing via AI APIs."""

__all__ = [
  "generate_image",
  "edit_image",
  "inpaint",
  "upscale_image",
]

import base64
import io
import json
import os
import re

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


def _load_image_ref(image_ref: str) -> Image.Image:
  """Load an image from a data URI or an HTTPS URL."""
  if image_ref.startswith("data:image/"):
    return _base64_to_image(image_ref)
  if image_ref.startswith("http://") or image_ref.startswith("https://"):
    with httpx.Client(timeout=120) as client:
      response = client.get(image_ref)
      response.raise_for_status()
      return Image.open(io.BytesIO(response.content))
  raise ValueError(f"unsupported image reference: {image_ref[:80]}")


def _extract_openrouter_image_ref(data: dict) -> str:
  """Extract the first image reference from known OpenRouter response shapes.

  OpenRouter's image-capable chat endpoint has returned generated images both
  in `message.images[].image_url.url` and in content parts. Keep the parser
  tolerant so model/provider format drift doesn't turn a valid image response
  into a failed edit.
  """
  choices = data.get("choices")
  if not isinstance(choices, list) or not choices:
    raise KeyError("choices")
  message = choices[0].get("message") if isinstance(choices[0], dict) else None
  if not isinstance(message, dict):
    raise KeyError("message")

  for item in message.get("images") or []:
    if not isinstance(item, dict):
      continue
    image_url = item.get("image_url")
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
      return image_url["url"]
    if isinstance(item.get("url"), str):
      return item["url"]

  content = message.get("content")
  if isinstance(content, list):
    for part in content:
      if not isinstance(part, dict):
        continue
      image_url = part.get("image_url")
      if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return image_url["url"]
      if isinstance(part.get("url"), str):
        return part["url"]
      source = part.get("source")
      if isinstance(source, dict):
        data_b64 = source.get("data")
        media_type = source.get("media_type", "image/png")
        if isinstance(data_b64, str):
          return f"data:{media_type};base64,{data_b64}"

  if isinstance(content, str):
    match = re.search(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\n\r]+", content)
    if match:
      return match.group(0).replace("\n", "").replace("\r", "")

  raise KeyError("images")


def _openrouter_image_or_raise(data: dict, *, transparent: bool = False) -> Image.Image:
  try:
    image_ref = _extract_openrouter_image_ref(data)
  except (KeyError, IndexError, TypeError) as e:
    raise RuntimeError(
      f"No image in OpenRouter response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e
  image = _load_image_ref(image_ref)
  if transparent and _has_alpha(image):
    return image.convert("RGBA")
  return image.convert("RGB")


def _has_alpha(image: Image.Image) -> bool:
  """True when the image carries an alpha channel (or palette transparency)."""
  return (
    image.mode in ("RGBA", "LA", "PA")
    or (image.mode == "P" and "transparency" in image.info)
  )


_IMAGE_ONLY_SUFFIX = (
  "\n\nReturn an image using the image modality. Do not respond with text only."
)

_TRANSPARENT_SUFFIX = (
  "\n\nRender the subject on a fully transparent background (PNG alpha channel,"
  " no backdrop, no solid color fill behind the subject)."
)


_OPENAI_1K_SIZES = {
  "1:1": "1024x1024",
  "9:16": "1024x1536",
  "16:9": "1536x1024",
  "3:4": "1024x1536",   # closest OpenAI-supported size
  "4:3": "1536x1024",
}

_OPENAI_GPT_IMAGE_2_2K_SIZES = {
  "1:1": "2048x2048",
  "9:16": "1440x2560",
  "16:9": "2560x1440",
  "3:4": "1536x2048",
  "4:3": "2048x1536",
}

_OPENAI_GPT_IMAGE_2_4K_SIZES = {
  "1:1": "2880x2880",
  "9:16": "2160x3840",
  "16:9": "3840x2160",
  "3:4": "2160x2880",
  "4:3": "2880x2160",
}


def _openai_size(aspect_ratio: str, image_size: str, model: str = "") -> str:
  """Map our (aspect_ratio, image_size) pair onto OpenAI's single `size`
  parameter.

  GPT Image 2 accepts explicit larger dimensions within its max-edge and
  total-pixel budget. Older OpenAI image models stay on the conservative
  fixed 1K-ish sizes plus `auto` path.
  """
  normalized_size = image_size.strip().upper()
  normalized_model = model.strip().lower()
  if normalized_model == "gpt-image-2":
    if normalized_size == "2K":
      return _OPENAI_GPT_IMAGE_2_2K_SIZES.get(aspect_ratio, "2048x2048")
    if normalized_size == "4K":
      return _OPENAI_GPT_IMAGE_2_4K_SIZES.get(aspect_ratio, "2880x2880")
  if normalized_size in {"2K", "4K", "AUTO"}:
    return "auto"
  return _OPENAI_1K_SIZES.get(aspect_ratio, "1024x1024")


def _generate_openai(
  prompt: str,
  model: str,
  aspect_ratio: str,
  image_size: str,
  transparent: bool = False,
) -> Image.Image:
  """Generate an image via OpenAI's direct image API (e.g. gpt-image-2).

  OpenAI-side model id has no `openai/` prefix. The caller strips it
  before dispatching here.
  """
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

  size = _openai_size(aspect_ratio, image_size, model)
  payload = {
    "model": model,
    "prompt": prompt,
    "size": size,
    "n": 1,
  }
  if transparent:
    payload["background"] = "transparent"
  target_mode = "RGBA" if transparent else "RGB"

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://api.openai.com/v1/images/generations",
      json=payload,
      headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
      },
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text[:500]}")
    data = response.json()

  try:
    item = data["data"][0]
    if "b64_json" in item:
      raw = base64.b64decode(item["b64_json"])
      return Image.open(io.BytesIO(raw)).convert(target_mode)
    # Some responses use a URL instead of b64_json.
    url = item["url"]
    with httpx.Client(timeout=120) as client:
      r = client.get(url)
      r.raise_for_status()
      return Image.open(io.BytesIO(r.content)).convert(target_mode)
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenAI response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e


def _edit_openai(image: Image.Image, prompt: str, model: str) -> Image.Image:
  """Edit one image via OpenAI's image edits endpoint.

  Sends multipart/form-data with the image file + prompt. The returned
  image is resized back to the input's dimensions so downstream
  compositing (which assumes identical crop sizes) keeps working.
  """
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

  original_size = image.size
  # Encode input as PNG bytes for the multipart upload.
  buf = io.BytesIO()
  image.convert("RGBA").save(buf, format="PNG")
  buf.seek(0)

  w, h = original_size
  if w == h:
    ar = "1:1"
  elif w > h:
    ar = "16:9" if w / h > 1.5 else "4:3"
  else:
    ar = "9:16" if h / w > 1.5 else "3:4"
  size = _openai_size(ar, "1K", model)

  files = {"image": ("input.png", buf.getvalue(), "image/png")}
  form = {
    "model": model,
    "prompt": prompt,
    "size": size,
    "n": "1",
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://api.openai.com/v1/images/edits",
      files=files,
      data=form,
      headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenAI edit API error {response.status_code}: {response.text[:500]}")
    data = response.json()

  try:
    item = data["data"][0]
    if "b64_json" in item:
      raw = base64.b64decode(item["b64_json"])
      result = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
      url = item["url"]
      with httpx.Client(timeout=120) as client:
        r = client.get(url)
        r.raise_for_status()
        result = Image.open(io.BytesIO(r.content)).convert("RGB")
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenAI edit response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e

  if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result


def _mask_to_openai_alpha(mask: Image.Image) -> bytes:
  """Encode a grayscale edit mask as an RGBA PNG for OpenAI image edits.

  OpenAI edits use the alpha channel as the editable area: transparent pixels
  are changed, opaque pixels are preserved. Our internal masks use white for
  "edit this", so invert the alpha when building the API mask.
  """
  alpha = mask.convert("L").point(lambda x: 0 if x > 128 else 255)
  rgba = Image.new("RGBA", alpha.size, (255, 255, 255, 255))
  rgba.putalpha(alpha)
  buf = io.BytesIO()
  rgba.save(buf, format="PNG")
  return buf.getvalue()


def _inpaint_openai(
  image: Image.Image,
  mask: Image.Image,
  prompt: str,
  model: str,
) -> Image.Image:
  """Masked edit via OpenAI's Images API.

  This is the right path for localized edits. It sends an alpha mask instead
  of asking the model to infer the editable region from prose, and uses low
  quality + JPEG output because level-editor dog previews are latency-sensitive.
  """
  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

  original_size = image.size
  image_buf = io.BytesIO()
  image.convert("RGBA").save(image_buf, format="PNG")

  files = {
    "image[]": ("input.png", image_buf.getvalue(), "image/png"),
    "mask": ("mask.png", _mask_to_openai_alpha(mask), "image/png"),
  }
  form = {
    "model": model.removeprefix("openai/"),
    "prompt": prompt,
    "quality": "low",
    "output_format": "jpeg",
    "n": "1",
  }

  with httpx.Client(timeout=300) as client:
    response = client.post(
      "https://api.openai.com/v1/images/edits",
      files=files,
      data=form,
      headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code != 200:
      raise RuntimeError(f"OpenAI edit API error {response.status_code}: {response.text[:500]}")
    data = response.json()

  try:
    item = data["data"][0]
    if "b64_json" in item:
      raw = base64.b64decode(item["b64_json"])
      result = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
      url = item["url"]
      with httpx.Client(timeout=120) as client:
        r = client.get(url)
        r.raise_for_status()
        result = Image.open(io.BytesIO(r.content)).convert("RGB")
  except (KeyError, IndexError) as e:
    raise RuntimeError(
      f"No image in OpenAI edit response: {e}\nResponse: {json.dumps(data, indent=2)[:500]}"
    ) from e

  if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result



def _google_image_or_raise(data: dict) -> Image.Image:
  """Extract the first inline image from a Gemini generateContent response."""
  for cand in data.get("candidates", []):
    for part in cand.get("content", {}).get("parts", []):
      inline = part.get("inlineData") or part.get("inline_data")
      if inline and inline.get("data"):
        raw = base64.b64decode(inline["data"])
        return Image.open(io.BytesIO(raw))
  raise RuntimeError(f"Gemini API returned no image: {str(data)[:400]}")


def _generate_google(
  prompt: str,
  model: str,
  aspect_ratio: str,
  transparent: bool,
  input_images: list[Image.Image] | None = None,
) -> Image.Image:
  """Direct Google Gemini API image generation/editing (key-gated fallback
  when OpenRouter is unavailable). Same prompt contract as the OpenRouter
  path; alpha is prompt-requested only, so callers needing guaranteed alpha
  must check the result mode and fall back to matting."""
  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not set in environment")
  suffix = _IMAGE_ONLY_SUFFIX + (_TRANSPARENT_SUFFIX if transparent else "")
  parts: list[dict] = []
  for img in input_images or []:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    parts.append({
      "inlineData": {"mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
    })
  parts.append({"text": prompt + suffix})
  payload = {
    "contents": [{"parts": parts}],
    "generationConfig": {
      "responseModalities": ["IMAGE"],
      "imageConfig": {"aspectRatio": aspect_ratio},
    },
  }
  with httpx.Client(timeout=300) as client:
    response = client.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
      json=payload,
      headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    if response.status_code != 200:
      raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:400]}")
    data = response.json()
  return _google_image_or_raise(data)


def generate_image(
  prompt: str,
  *,
  model: str = "google/gemini-3.1-flash-image-preview",
  aspect_ratio: str = "1:1",
  image_size: str = "1K",
  transparent: bool = False,
) -> Image.Image:
  """Generate an image from a text prompt.

  Dispatches by model prefix:
  - `openai/...` → OpenAI's direct image API (gpt-image-2, etc).
  - Anything else → OpenRouter chat-completions with image modality.

  Args:
    prompt: Text description of the image to generate.
    model: OpenRouter model id OR `openai/<openai-model>`.
    aspect_ratio: Aspect ratio string (e.g., "1:1", "9:16", "16:9").
    image_size: Resolution tier ("1K", "2K", "4K") — OpenAI maps to WxH.
    transparent: Request a transparent background. OpenAI: native alpha via
      `background: "transparent"` (RGBA result). OpenRouter: prompt-requested;
      alpha is preserved only when the model actually returns it — callers
      needing guaranteed alpha must check the result mode and fall back to
      matting.

  Returns:
    PIL Image — RGBA when `transparent` produced real alpha, RGB otherwise.
  """
  if model.startswith("openai/") and os.environ.get("OPENAI_API_KEY"):
    return _generate_openai(
      prompt, model.removeprefix("openai/"), aspect_ratio, image_size, transparent
    )
  # Without an OpenAI key, `openai/...` ids fall through to OpenRouter, which
  # serves the same model ids (no native `background: transparent` there —
  # callers needing guaranteed alpha must check the result mode).
  if model.startswith("google/") and os.environ.get("GOOGLE_API_KEY") and not os.environ.get("MERCEKA_FORCE_OPENROUTER"):
    # Key-gated direct Gemini dispatch; OpenRouter remains the default when
    # only OPENROUTER_API_KEY is present.
    return _generate_google(prompt, model.removeprefix("google/"), aspect_ratio, transparent)

  api_key = os.environ.get("OPENROUTER_API_KEY")
  if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")

  suffix = _IMAGE_ONLY_SUFFIX + (_TRANSPARENT_SUFFIX if transparent else "")
  payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt + suffix}],
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

  return _openrouter_image_or_raise(data, transparent=transparent)


def edit_image(
  image: Image.Image,
  prompt: str,
  *,
  model: str = "google/gemini-3.1-flash-image-preview",
  resize_to_input: bool = True,
) -> Image.Image:
  """Send one image + text prompt, get back a modified image.

  Unlike inpaint(), this sends a single image as context — no mask.
  Dispatches by model prefix:
  - `openai/...` → OpenAI's `/v1/images/edits` multipart endpoint.
  - Anything else → OpenRouter chat-completions with image modality.

  Args:
    image: Reference image (RGB).
    prompt: Instructions for what to do with the image.
    model: OpenRouter model id OR `openai/<openai-model>`.

  Returns:
    PIL Image in RGB mode, resized to the input's original dimensions.
  """
  if model.startswith("openai/") and os.environ.get("OPENAI_API_KEY"):
    return _edit_openai(image, prompt, model.removeprefix("openai/"))
  if model.startswith("google/") and os.environ.get("GOOGLE_API_KEY") and not os.environ.get("MERCEKA_FORCE_OPENROUTER"):
    result = _generate_google(prompt, model.removeprefix("google/"), "1:1", False, input_images=[image])
    if resize_to_input and result.size != image.size:
      result = result.resize(image.size, Image.Resampling.LANCZOS)
    return result

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
        {"type": "text", "text": prompt + _IMAGE_ONLY_SUFFIX},
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

  result = _openrouter_image_or_raise(data)
  if resize_to_input and result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result


def _fal_upscale_payload(model: str, image_uri: str, scale: float) -> dict:
  """Build a fal.ai upscaler payload for the supported background models."""
  if scale < 1 or scale > 8:
    raise ValueError(f"scale must be between 1 and 8, got {scale}")

  if model == "fal-ai/esrgan":
    return {
      "image_url": image_uri,
      "scale": scale,
      "model": "RealESRGAN_x4plus",
      "output_format": "png",
    }

  if model == "fal-ai/aura-sr":
    return {
      "image_url": image_uri,
      "upscale_factor": 4,
      "overlapping_tiles": False,
      "checkpoint": "v2",
    }

  raise ValueError(f"unsupported upscaler model: {model}")


def _image_from_fal_response(result_data: dict) -> Image.Image:
  """Load the first image from fal.ai's common response shapes."""
  image_info = result_data.get("image")
  if isinstance(image_info, dict) and isinstance(image_info.get("url"), str):
    result_image_url = image_info["url"]
  else:
    try:
      result_image_url = result_data["images"][0]["url"]
    except (KeyError, IndexError, TypeError) as e:
      raise RuntimeError(
        f"No image in fal.ai response: {e}\nResponse: {json.dumps(result_data, indent=2)[:500]}"
      ) from e

  with httpx.Client(timeout=120) as client:
    img_response = client.get(result_image_url)
    img_response.raise_for_status()

  return Image.open(io.BytesIO(img_response.content)).convert("RGB")


def upscale_image(
  image: Image.Image,
  *,
  model: str = "fal-ai/esrgan",
  scale: float = 2.0,
) -> Image.Image:
  """Upscale an image via a dedicated fal.ai upscaler.

  This is intended for background masters, not dog insertion crops. It returns
  whatever resolution the provider produces; callers that need an exact long
  edge should resize the returned image after the model pass.
  """
  api_key = os.environ.get("FAL_KEY")
  if not api_key:
    raise RuntimeError("FAL_KEY not set in environment")

  image_uri = _image_to_base64_uri(image.convert("RGB"))
  payload = _fal_upscale_payload(model, image_uri, scale)

  with httpx.Client(timeout=600) as client:
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

  return _image_from_fal_response(result_data)


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
  if model.startswith("openai/"):
    return _inpaint_openai(image, mask, prompt, model)
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
    f"Return only the edited image using the image modality. Do not respond with text only."
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

  result = _openrouter_image_or_raise(data)
  # Resize to match input dimensions (OpenRouter may return different size)
  if result.size != original_size:
    result = result.resize(original_size, Image.Resampling.LANCZOS)
  return result
