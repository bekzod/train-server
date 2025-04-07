#!/usr/bin/env python
import argparse
import torch
import numpy as np
import sounddevice as sd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

def load_model(base_model, lora_checkpoint, device):
    """
    Loads the processor and base Whisper model, applies the LoRA adapter checkpoint,
    and moves the model to the specified device.
    """
    # Load the processor from the base model
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    
    # Choose the torch dtype based on device availability
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Load the base Whisper model
    base_model_obj = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    base_model_obj.config.pad_token_id = processor.tokenizer.pad_token_id

    # Create the LoRA configuration (must match training settings)
    config = LoraConfig(
        r=64,
        lora_alpha=8,
        use_rslora=True,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["model.embed_tokens"],
        lora_dropout=0.02,
        bias="none"
    )
    # Wrap the base model with the LoRA configuration
    model_with_lora = get_peft_model(base_model_obj, config)
    # Load the adapter checkpoint weights
    model_with_lora = PeftModel.from_pretrained(model_with_lora, lora_checkpoint)
    model_with_lora.to(device)
    return processor, model_with_lora

def transcribe_audio(processor, model, audio, device):
    """
    Transcribes a numpy audio array (expected sample rate: 16000Hz) using the processor and model.
    """
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

def transcribe_file(processor, model, file_path, device):
    """
    Loads an audio file (resampling it to 16000Hz), and transcribes its content.
    """
    audio, sr = librosa.load(file_path, sr=16000)
    transcription = transcribe_audio(processor, model, audio, device)
    return transcription

def realtime_transcription(processor, model, device, duration=5):
    """
    Continuously records audio (in fixed-length segments) from the microphone and prints transcriptions.
    """
    print("Starting real-time transcription (press Ctrl+C to stop)...")
    try:
        while True:
            print(f"\nRecording for {duration} seconds...")
            # Record mono audio at 16kHz
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            audio = np.squeeze(audio)
            transcription = transcribe_audio(processor, model, audio, device)
            print("Transcription:", transcription)
    except KeyboardInterrupt:
        print("\nReal-time transcription stopped.")

def main():
    parser = argparse.ArgumentParser(
        description="Test Whisper model with LoRA adapter checkpoint for transcription (real-time via microphone or for an audio file)."
    )
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3",
                        help="Base Whisper model name or path (default: openai/whisper-large-v3).")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to the LoRA adapter checkpoint (required).")
    parser.add_argument("--mode", type=str, choices=["mic", "file"], default="mic",
                        help="Transcription mode: 'mic' for real-time transcription or 'file' for audio file transcription.")
    parser.add_argument("--file_path", type=str,
                        help="Path to an audio file (required if mode is 'file').")
    parser.add_argument("--duration", type=float, default=5,
                        help="Duration (in seconds) for recording segments in real-time mode (default: 5 seconds).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Loading base model and applying LoRA adapter checkpoint...")
    processor, model = load_model(args.base_model, args.lora_checkpoint, device)
    print("Model loaded successfully.\n")

    if args.mode == "file":
        if not args.file_path:
            print("Error: --file_path must be specified when mode is 'file'.")
            return
        print(f"Transcribing audio file: {args.file_path}")
        transcription = transcribe_file(processor, model, args.file_path, device)
        print("\nTranscription:", transcription)
    else:
        realtime_transcription(processor, model, device, duration=args.duration)

if __name__ == "__main__":
    main()
