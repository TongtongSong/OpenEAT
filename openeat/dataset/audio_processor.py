import random

import torchaudio
torchaudio.set_audio_backend("sox_io")
def _speed_perturb(waveform, sample_rate, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    assert speeds[1] > speeds[0], 'speeds is wrong !'
    if speeds is None:
        speeds = [0.9, 1.1, 0.1]
    else:
        speeds = [float(s) for s in speeds]
    if speeds[2]!=0:
        speed = random.randrange(int(speeds[0]/speeds[2]),int(speeds[0]/speeds[2])+1)
        speed *= speeds[2]
    else:
        speed = speeds[0] + random.random()*(speeds[1]-speeds[0])
    if speed != 1.0:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
    return waveform
