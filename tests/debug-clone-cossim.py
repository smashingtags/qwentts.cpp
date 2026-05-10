#!/usr/bin/env python3
"""Cossim debug : C++ qwen-tts vs Python Qwen3-TTS on the Voice Clone Mode B
(ICL) path.

Inputs (relative to CWD = tests/) :
    ../examples/prompt.txt       target text fed to both pipelines
    ../examples/freeman.wav      reference audio for cloning
    ../examples/freeman.txt      transcript of the reference audio

Default mode is greedy on both sides, non_streaming_mode=False which is
the ICL branch upstream : text + codec streams are aligned to the codec
length, the shorter one padded with tts_pad / truncated as needed.

Cote Python the speaker embedding is captured directly via
model.extract_speaker_embedding, and the reference codec frames via
model.speech_tokenizer.encode. Both intermediates land as speaker-emb.bin
and ref-codes.bin and are compared against the C++ side dumps emitted
by pipeline-tts.cpp when --ref-audio and --ref-text are set.

Dumps land in cpp/clone/ (C++) and python/clone/ (Python).
"""

import argparse
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

import cossim_common as cc

MODEL_T     = "../models/qwen-talker-1.7b-base-{q}.gguf"
MODEL_CDC_T = "../models/qwen-tokenizer-12hz-{q}.gguf"
CKPT        = "../checkpoints/Qwen3-TTS-12Hz-1.7B-Base"
DUMP_CPP    = "cpp/clone"
DUMP_PT     = "python/clone"

DEFAULT_REF_AUDIO = "../examples/freeman.wav"
DEFAULT_REF_TEXT  = "../examples/freeman.txt"

# Mode B adds two pre-talker stages to the standard list : the speaker
# embedding extracted from the reference audio (ECAPA forward, projected to
# talker hidden), and the reference codec frames at 12.5 Hz.
STAGES_CLONE = cc.STAGES_STANDARD + [
    ("SpeakerEmb", "speaker-emb.bin"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",         default="../examples/prompt.txt")
    ap.add_argument("--ref-audio",      default=DEFAULT_REF_AUDIO,
                    help="reference WAV path for voice cloning")
    ap.add_argument("--ref-text-file",  default=DEFAULT_REF_TEXT,
                    help="path to a UTF-8 file with the transcript of ref-audio")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--lang",           default="english")
    ap.add_argument("--quant",          default="F32",
                    help="GGUF quantization suffix (F32, BF16, Q8_0, Q4_K_M)")
    ap.add_argument("--out-pt",         default=os.path.join(DUMP_PT,  "clone-python.wav"))
    ap.add_argument("--out-cpp",        default=os.path.join(DUMP_CPP, "clone-cpp.wav"))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--trace",          action="store_true",
                    help="print per sample u and idx for the first 32 samples")
    args = ap.parse_args()

    cc.ensure_dir(DUMP_PT)
    cc.ensure_dir(DUMP_CPP)
    os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    with open(args.ref_text_file, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] RefAudio: {args.ref_audio}")
    print(f"[Input] RefText: {len(ref_text)} chars: {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}")
    print(f"[Input] Lang: {args.lang} Seed: {args.seed} MaxNewTokens: {args.max_new_tokens}")
    print(f"[Input] Mode: greedy ICL")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cc.set_trace(args.trace)

    cc.register_qwen3_tts()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Python] Device: {device}")
    model = cc.AutoModel.from_pretrained(
        CKPT,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    processor = cc.AutoProcessor.from_pretrained(CKPT, fix_mistral_regex=True)

    # Load reference WAV. Resample to 24 kHz if needed since both the speaker
    # encoder and the codec tokenizer expect 24 kHz mono input.
    ref_wav, ref_sr = sf.read(args.ref_audio, always_2d=False)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav[:, 0]
    ref_wav = ref_wav.astype(np.float32)
    target_sr = model.speaker_encoder_sample_rate
    if ref_sr != target_sr:
        ref_wav = librosa.resample(y=ref_wav, orig_sr=int(ref_sr), target_sr=int(target_sr))
        ref_sr = target_sr
    print(f"[Python] RefWav: {ref_wav.shape[0]} samples {ref_sr} Hz {ref_wav.shape[0]/ref_sr:.2f}s")

    # Extract speaker embedding via ECAPA forward, projected to talker hidden.
    spk_emb = model.extract_speaker_embedding(audio=ref_wav, sr=ref_sr)
    print(f"[Python] SpeakerEmb shape: {tuple(spk_emb.shape)} dtype: {spk_emb.dtype}")
    cc.save_dump(os.path.join(DUMP_PT, "speaker-emb.bin"), spk_emb)

    # Encode the reference audio to 16 codebook codes at 12.5 Hz. The encode
    # call returns shape [T_codec, K=16] after the internal transpose, while
    # the C++ side dumps [K=16, T_codec] row major. We transpose here for a
    # straight exact match comparison.
    enc         = model.speech_tokenizer.encode([ref_wav], sr=int(ref_sr))
    ref_code_pt = enc.audio_codes[0]
    ref_code_kt = ref_code_pt.transpose(0, 1).contiguous()
    print(f"[Python] RefCodes shape: {tuple(ref_code_kt.shape)} (K, T_codec)")
    cc.save_dump_i32(os.path.join(DUMP_PT, "ref-codes.bin"), ref_code_kt)

    # Tokenize the utterance and the reference text.
    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inp_utt        = processor(text=assistant_text, return_tensors="pt", padding=True)
    input_ids      = inp_utt["input_ids"].to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    print(f"[Python] InputIds shape: {tuple(input_ids.shape)}")
    cc.save_dump_i32(os.path.join(DUMP_PT, "prompt-ids.bin"), input_ids[0])

    ref_text_wrap = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
    inp_ref       = processor(text=ref_text_wrap, return_tensors="pt", padding=True)
    ref_ids       = inp_ref["input_ids"].to(device)
    if ref_ids.dim() == 1:
        ref_ids = ref_ids.unsqueeze(0)
    print(f"[Python] RefIds shape: {tuple(ref_ids.shape)}")
    cc.save_dump_i32(os.path.join(DUMP_PT, "ref-ids.bin"), ref_ids[0])

    cc.install_hooks(model, DUMP_PT)

    # Custom subtalker_* kwargs are forwarded to talker.forward but not
    # declared on GenerationMixin, so transformers 4.57 rejects them under
    # the strict validator. Disable it on the talker only.
    model.talker._validate_model_kwargs = lambda *a, **k: None

    gen_kwargs = dict(
        do_sample             = False,
        top_k                 = 1,
        top_p                 = 1.0,
        temperature           = 1.0,
        subtalker_dosample    = False,
        subtalker_top_k       = 1,
        subtalker_top_p       = 1.0,
        subtalker_temperature = 1.0,
        repetition_penalty    = 1.0,
    )

    # voice_clone_prompt dict mirrors what _prompt_items_to_voice_clone_prompt
    # builds for a single ICL prompt item : ref_code is the [T_codec, K]
    # tensor, ref_spk_embedding is the [hidden] tensor, x_vector_only=False
    # and icl_mode=True together select the mode B branch upstream.
    voice_clone_prompt_dict = dict(
        ref_code           = [ref_code_pt],
        ref_spk_embedding  = [spk_emb],
        x_vector_only_mode = [False],
        icl_mode           = [True],
    )

    talker_codes_list, _ = model.generate(
        input_ids=[input_ids],
        ref_ids=[ref_ids],
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[args.lang],
        non_streaming_mode=False,
        max_new_tokens=args.max_new_tokens,
        **gen_kwargs,
    )
    codes = talker_codes_list[0]
    print(f"[Python] Codes shape: {tuple(codes.shape)} (T_frames, num_code_groups)")
    cc.save_dump_i32(os.path.join(DUMP_PT, "codes-full.bin"),  codes)
    cc.save_dump_i32(os.path.join(DUMP_PT, "codes-step0.bin"), codes[0])

    # The decode path prepends the reference codes and cuts the matching
    # audio prefix afterwards, mirroring generate_voice_clone exactly so the
    # produced WAV only covers the freshly generated portion.
    cat_codes = torch.cat([ref_code_pt.to(codes.device), codes], dim=0)
    wavs, fs  = model.speech_tokenizer.decode([{"audio_codes": cat_codes}])
    full_wav  = np.asarray(wavs[0], dtype=np.float32)
    ref_len   = int(ref_code_pt.shape[0])
    total_len = int(cat_codes.shape[0])
    cut       = int(ref_len / max(total_len, 1) * full_wav.shape[0])
    audio_pt  = full_wav[cut:]
    sf.write(args.out_pt, audio_pt, fs, subtype="FLOAT")
    cc.save_dump(os.path.join(DUMP_PT, "output-audio.bin"), audio_pt)
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {fs} Hz {audio_pt.shape[0]/fs:.2f}s -> {args.out_pt}")

    if not os.path.isfile(cc.BIN):
        print(f"[Cossim] FATAL: {cc.BIN} not found, build qwen-tts first")
        sys.exit(1)
    model_lm  = MODEL_T.format(q=args.quant)
    model_cdc = MODEL_CDC_T.format(q=args.quant)
    for p in (model_lm, model_cdc):
        if not os.path.isfile(p):
            print(f"[Cossim] FATAL: GGUF not found: {p}")
            sys.exit(1)
    print(f"[Quant] {args.quant} -> {model_lm} + {model_cdc}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    cmd = [
        cc.BIN,
        "--model",     model_lm,
        "--codec",     model_cdc,
        "--seed",      str(args.seed),
        "--text",      text,
        "--ref-audio", args.ref_audio,
        "--ref-text",  ref_text,
        "--lang",      args.lang,
        "--max-new",   str(args.max_new_tokens),
        "--dump",      DUMP_CPP,
        "-o",          args.out_cpp,
        "--greedy",
    ]
    print(f"[GGML] Cmd: {' '.join(cmd[:6])} --text [...] --ref-audio {args.ref_audio} --ref-text [...] --lang {args.lang} --max-new {args.max_new_tokens} --dump {DUMP_CPP} -o {args.out_cpp} --greedy")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

    audio_cpp, sr = sf.read(args.out_cpp)
    if audio_cpp.ndim > 1:
        audio_cpp = audio_cpp[:, 0]
    audio_cpp = audio_cpp.astype(np.float32)
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0]/sr:.2f}s -> {args.out_cpp}")

    cc.compare_exact_i32("prompt-ids.bin", DUMP_CPP, DUMP_PT, "PromptIDs")
    cc.compare_exact_i32("ref-codes.bin",  DUMP_CPP, DUMP_PT, "RefCodes")
    cc.compare_stages(STAGES_CLONE, DUMP_CPP, DUMP_PT)
    cc.compare_exact_i32("codes-full.bin", DUMP_CPP, DUMP_PT, "CodesFull")

    aa, ab = cc.pair("output-audio.bin", DUMP_CPP, DUMP_PT)
    print(f"[Cossim] Audio cos: {cc.cos(aa, ab):.6f}")

    n = min(aa.size, ab.size)
    print(f"[Cossim] WAV stft_cos: {cc.stft_cos(aa.ravel()[:n], ab.ravel()[:n]):.6f} samples: {n}")

if __name__ == "__main__":
    main()
