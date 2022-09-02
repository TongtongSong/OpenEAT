

if len(value) == 3:
    start_frame = int(float(value[1]) * sample_rate)
    end_frame = int(float(value[2]) * sample_rate)
    waveform, sample_rate = torchaudio.load(
        filepath=wav_path,
        num_frames=end_frame - start_frame,
        frame_offset=start_frame)
else:
    waveform, sample_rate = torchaudio.load(wav_path)
waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])