"""
Gradio Web UI for Qwen 2.5 Omni — real-time multimodal conversation.

Supports:
  - Webcam image capture
  - Microphone audio input
  - Text input
  - Text + audio speech output (played in browser)

Uses direct transformers inference (no vLLM server).
"""

import argparse
import io
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Global model and processor (loaded once at startup)
model = None
processor = None

AUDIO_SAMPLE_RATE = 24000


def load_model(model_path: str):
    """Load the Qwen2.5-Omni model and processor."""
    global model, processor

    print(f"Loading model from {model_path}...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("Model loaded successfully.")


def numpy_audio_to_wav_path(sample_rate: int, audio_array: np.ndarray) -> str:
    """Save Gradio's (sample_rate, numpy_array) audio to a temporary WAV file and return the path."""
    audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    peak = np.max(np.abs(audio_array))
    if peak > 0:
        audio_array = audio_array / peak
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_array, sample_rate)
    return tmp.name


def build_ui():
    """Construct and return the Gradio Blocks app."""

    def generate_response(
        message: str,
        image: str | None,
        audio: tuple[int, np.ndarray] | None,
        history: list[dict],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        voice: str,
    ):
        """Run inference and return text + audio."""

        # Build user content parts for the Qwen chat template
        user_content = []

        if image is not None:
            user_content.append({"type": "image", "image": image})

        if audio is not None:
            sr, arr = audio
            wav_path = numpy_audio_to_wav_path(sr, arr)
            user_content.append({"type": "audio", "audio": wav_path})

        text = message.strip() if message else ""
        if not text and not user_content:
            return history, None
        if text:
            user_content.append({"type": "text", "text": text})
        elif not user_content:
            return history, None

        # Build display text for chat history
        display_parts = []
        if text:
            display_parts.append(text)
        if image:
            display_parts.append("[image attached]")
        if audio:
            display_parts.append("[audio attached]")

        history = history + [{"role": "user", "content": " ".join(display_parts)}]

        # Build messages for the model
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})

        # Rebuild conversation from history (text-only for prior turns)
        for entry in history[:-1]:  # exclude the latest user message we just added
            messages.append({"role": entry["role"], "content": [{"type": "text", "text": entry["content"]}]})

        # Add current user message with multimodal content
        messages.append({"role": "user", "content": user_content})

        try:
            # Process with Qwen's chat template
            text_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Extract multimodal inputs
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

            inputs = processor(
                text=text_prompt,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # Generate with audio output
            gen_kwargs = {
                "use_audio_in_video": True,
                "return_audio": True,
                "speaker": voice,
                "max_new_tokens": max_tokens,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                text_ids, audio_waveform = model.generate(**inputs, **gen_kwargs)

            # Decode text
            assistant_text = processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_text = assistant_text[0] if assistant_text else ""

            history = history + [{"role": "assistant", "content": response_text or "(audio-only response)"}]

            # Convert audio waveform to Gradio format (sample_rate, numpy_array)
            audio_output = None
            if audio_waveform is not None:
                if isinstance(audio_waveform, torch.Tensor):
                    audio_np = audio_waveform.cpu().float().numpy()
                else:
                    audio_np = np.array(audio_waveform, dtype=np.float32)

                # Flatten if needed
                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze()

                # Normalize to int16 range for playback
                peak = np.max(np.abs(audio_np))
                if peak > 0:
                    audio_np = audio_np / peak
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_output = (AUDIO_SAMPLE_RATE, audio_int16)

            return history, audio_output

        except Exception as e:
            error_msg = f"Error: {e}"
            import traceback
            traceback.print_exc()
            history = history + [{"role": "assistant", "content": error_msg}]
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
                voice_select = gr.Dropdown(
                    label="Voice",
                    choices=["Chelsie", "Ethan"],
                    value="Chelsie",
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
                clear_btn = gr.Button("Clear Conversation")
                gr.Markdown("**Inference:** Direct transformers (in-process)")

        # State for conversation history
        history_state = gr.State([])

        def on_send(message, image, audio, history, system_prompt, temperature, max_tokens, voice):
            updated_history, audio_out = generate_response(
                message, image, audio, history, system_prompt, temperature, max_tokens, voice
            )
            return updated_history, updated_history, audio_out, "", None, None

        def on_clear():
            return [], [], None

        send_btn.click(
            fn=on_send,
            inputs=[
                text_input, image_input, audio_input, history_state,
                system_prompt, temperature, max_tokens, voice_select,
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
                system_prompt, temperature, max_tokens, voice_select,
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
    parser.add_argument("--model-path", type=str, default="/workspace/models/Qwen2.5-Omni-3B",
                        help="Path to the model directory")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    load_model(args.model_path)

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
