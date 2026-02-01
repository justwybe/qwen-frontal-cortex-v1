"""
Gradio Web UI for Qwen 2.5 Omni — real-time multimodal conversation.

Supports:
  - Webcam image capture (sent as base64 image)
  - Microphone audio input (sent as base64 audio)
  - Text input
  - Streaming text output
  - Audio speech output (played in browser)

Requires vLLM-Omni server running with --omni flag on port 8000.
"""

import argparse
import base64
import io
import json
import struct
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from openai import OpenAI


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return a data-URI string."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(suffix, "image/png")
    return f"data:{mime};base64,{data}"


def encode_audio_to_base64(audio_path: str) -> str:
    """Read an audio file and return a data-URI string."""
    data, sr = sf.read(audio_path)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def numpy_audio_to_base64(sample_rate: int, audio_array: np.ndarray) -> str:
    """Convert Gradio's (sample_rate, numpy_array) audio to base64 data-URI."""
    audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio_array))
    if peak > 0:
        audio_array = audio_array / peak
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def decode_audio_response(audio_b64: str) -> tuple[int, np.ndarray] | None:
    """Decode a base64-encoded WAV from the model response into (sample_rate, numpy_array)."""
    try:
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        raw = base64.b64decode(audio_b64)
        buf = io.BytesIO(raw)
        data, sr = sf.read(buf)
        return sr, (data * 32767).astype(np.int16)
    except Exception:
        return None


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000,
                     num_channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    data_size = len(pcm_data)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, num_channels,
        sample_rate, sample_rate * num_channels * sample_width,
        num_channels * sample_width, sample_width * 8,
        b'data', data_size,
    )
    return header + pcm_data


def build_ui(vllm_base_url: str):
    """Construct and return the Gradio Blocks app."""

    client = OpenAI(base_url=f"{vllm_base_url}/v1", api_key="not-needed")

    # Detect the served model name once at startup
    try:
        models = client.models.list()
        model_id = models.data[0].id if models.data else "Qwen2.5-Omni-3B"
    except Exception:
        model_id = "Qwen2.5-Omni-3B"

    def chat(
        message: str,
        image: str | None,
        audio: tuple[int, np.ndarray] | None,
        history: list[dict],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ):
        """Send a multimodal message and stream the response."""

        # Build the user content list
        content: list[dict] = []

        # Attach image if provided (webcam snapshot)
        if image is not None:
            img_uri = encode_image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_uri},
            })

        # Attach audio if provided (microphone recording)
        if audio is not None:
            sr, arr = audio
            audio_uri = numpy_audio_to_base64(sr, arr)
            content.append({
                "type": "input_audio",
                "input_audio": {"data": audio_uri, "format": "wav"},
            })

        # Attach text (always present, even if empty — the model needs at least one text part)
        text = message.strip() if message else ""
        if not text and not content:
            yield history, None
            return
        if text:
            content.append({"type": "text", "text": text})
        elif not content:
            yield history, None
            return

        # Build messages array
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # Add conversation history
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})

        messages.append({"role": "user", "content": content})

        # Append user message to history for display
        display_parts = []
        if text:
            display_parts.append(text)
        if image:
            display_parts.append("[image attached]")
        if audio:
            display_parts.append("[audio attached]")
        user_display = " ".join(display_parts)

        history = history + [{"role": "user", "content": user_display}]

        # Call vLLM-Omni with streaming, requesting both text and audio
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                modalities=["text", "audio"],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            assistant_text = ""
            audio_data_b64 = ""

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Text content
                if delta.content:
                    assistant_text += delta.content
                    updated_history = history + [
                        {"role": "assistant", "content": assistant_text}
                    ]
                    yield updated_history, None

                # Audio content (may arrive in the delta or as audio field)
                if hasattr(delta, "audio") and delta.audio:
                    audio_chunk = delta.audio
                    if isinstance(audio_chunk, dict):
                        audio_data_b64 += audio_chunk.get("data", "")
                    elif isinstance(audio_chunk, str):
                        audio_data_b64 += audio_chunk

            # Final update with complete text
            if assistant_text:
                history = history + [{"role": "assistant", "content": assistant_text}]
            else:
                history = history + [{"role": "assistant", "content": "(audio-only response)"}]

            # Decode audio if present
            audio_output = None
            if audio_data_b64:
                audio_output = decode_audio_response(audio_data_b64)

            yield history, audio_output

        except Exception as e:
            error_msg = f"Error: {e}"
            history = history + [{"role": "assistant", "content": error_msg}]
            yield history, None

    def chat_non_stream(
        message: str,
        image: str | None,
        audio: tuple[int, np.ndarray] | None,
        history: list[dict],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ):
        """Non-streaming fallback — collects the full response then returns."""

        content: list[dict] = []

        if image is not None:
            img_uri = encode_image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_uri},
            })

        if audio is not None:
            sr, arr = audio
            audio_uri = numpy_audio_to_base64(sr, arr)
            content.append({
                "type": "input_audio",
                "input_audio": {"data": audio_uri, "format": "wav"},
            })

        text = message.strip() if message else ""
        if text:
            content.append({"type": "text", "text": text})
        elif not content:
            return history, None

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        for entry in history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": content})

        display_parts = []
        if text:
            display_parts.append(text)
        if image:
            display_parts.append("[image attached]")
        if audio:
            display_parts.append("[audio attached]")
        history = history + [{"role": "user", "content": " ".join(display_parts)}]

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                modalities=["text", "audio"],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            choice = response.choices[0]
            assistant_text = choice.message.content or ""
            history = history + [
                {"role": "assistant", "content": assistant_text or "(audio-only response)"}
            ]

            audio_output = None
            if hasattr(choice.message, "audio") and choice.message.audio:
                aud = choice.message.audio
                b64 = aud.get("data", "") if isinstance(aud, dict) else str(aud)
                if b64:
                    audio_output = decode_audio_response(b64)

            return history, audio_output

        except Exception as e:
            history = history + [{"role": "assistant", "content": f"Error: {e}"}]
            return history, None

    # ---- Gradio Layout ----
    with gr.Blocks(
        title="Qwen 2.5 Omni — Multimodal Conversation",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Qwen 2.5 Omni — Real-Time Multimodal Conversation\n"
            "Talk via **text**, **webcam**, or **microphone**. "
            "The model responds with both text and speech."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                )
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Message",
                        placeholder="Type a message or use webcam/mic below...",
                        lines=2,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    image_input = gr.Image(
                        label="Webcam / Image",
                        sources=["webcam", "upload"],
                        type="filepath",
                    )
                    audio_input = gr.Audio(
                        label="Microphone",
                        sources=["microphone", "upload"],
                        type="numpy",
                    )

                audio_output = gr.Audio(
                    label="Model Speech Output",
                    autoplay=True,
                    type="numpy",
                )

            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful assistant. Respond naturally in conversation.",
                    lines=4,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                )
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                )
                use_streaming = gr.Checkbox(
                    label="Stream responses",
                    value=True,
                )
                clear_btn = gr.Button("Clear Conversation")
                model_info = gr.Markdown(f"**Model:** `{model_id}`\n\n**Server:** `{vllm_base_url}`")

        # State for conversation history (kept as list of dicts)
        history_state = gr.State([])

        def on_send(message, image, audio, history, system_prompt, temperature, max_tokens, streaming):
            if streaming:
                for updated_history, audio_out in chat(
                    message, image, audio, history, system_prompt, temperature, max_tokens
                ):
                    yield updated_history, updated_history, audio_out, "", None, None
            else:
                updated_history, audio_out = chat_non_stream(
                    message, image, audio, history, system_prompt, temperature, max_tokens
                )
                yield updated_history, updated_history, audio_out, "", None, None

        def on_clear():
            return [], [], None

        send_btn.click(
            fn=on_send,
            inputs=[
                text_input, image_input, audio_input, history_state,
                system_prompt, temperature, max_tokens, use_streaming,
            ],
            outputs=[
                chatbot, history_state, audio_output,
                text_input, image_input, audio_input,
            ],
        )

        text_input.submit(
            fn=on_send,
            inputs=[
                text_input, image_input, audio_input, history_state,
                system_prompt, temperature, max_tokens, use_streaming,
            ],
            outputs=[
                chatbot, history_state, audio_output,
                text_input, image_input, audio_input,
            ],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, history_state, audio_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Qwen 2.5 Omni Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--vllm-host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    vllm_base_url = f"http://{args.vllm_host}:{args.vllm_port}"
    demo = build_ui(vllm_base_url)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
