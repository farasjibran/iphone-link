"""Parity test: handset.aac_darwin.AudioToolboxAacDecoder vs pymobiledevice3's
native macOS AACELDDecoder, fed the SAME encoded Access Units.

Encodes a 440 Hz tone with the AudioToolbox AAC encoder (av 'aac_at'), then
decodes it with both decoders and compares byte-for-byte + RMS/peak.

Usage: python3 tools/audio_decoder_parity.py
"""

import fractions
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from handset.aac_darwin import (
    AudioToolboxAacDecoder,
    asc_audio_object_type,
    asc_frames_per_packet,
    asc_sample_rate,
)
from pymobiledevice3.remote.core_device.aac_eld import (
    AAC_ELD_ASC_48K_STEREO_480,
    AACELDDecoder,
)


def rms_dbfs(pcm: bytes) -> float:
    if not pcm:
        return -96.0
    arr = np.frombuffer(pcm[: len(pcm) & ~3], dtype="<i2").astype(np.float64)
    if arr.size == 0:
        return -96.0
    rms = float(np.sqrt((arr * arr).mean()))
    return -96.0 if rms <= 0 else round(20.0 * np.log10(rms / 32768.0), 1)


def peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    arr = np.frombuffer(pcm[: len(pcm) & ~3], dtype="<i2")
    return int(np.abs(arr).max()) if arr.size else 0


def main() -> int:
    import av

    sr = 48000
    enc = av.CodecContext.create("aac_at", "w")
    enc.sample_rate = sr
    enc.layout = "stereo"
    enc.format = "s16"
    enc.bit_rate = 128000
    enc.time_base = fractions.Fraction(1, sr)
    enc.open()

    t = np.arange(sr) / sr
    sig = (0.35 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    stereo = np.repeat(sig, 2)

    packets = []
    for i in range(0, len(stereo) // 2 - 1024, 1024):
        frame = av.AudioFrame.from_ndarray(
            stereo[i * 2 : (i + 1024) * 2].reshape(1, -1), format="s16", layout="stereo"
        )
        frame.sample_rate = sr
        frame.pts = i
        packets.extend(enc.encode(frame))
    packets.extend(enc.encode(None))

    asc = bytes(enc.extradata)
    print(f"encoded {len(packets)} AUs")
    print(f"asc={asc.hex()} aot={asc_audio_object_type(asc)} "
          f"rate={asc_sample_rate(asc)} frames/pkt={asc_frames_per_packet(asc)}")

    # Reference decoder: pymobiledevice3's native macOS AAC-ELD decoder.
    ref = AACELDDecoder(AAC_ELD_ASC_48K_STEREO_480)
    # Ours: the module under test, driven with the encoder's own ASC.
    ours = AudioToolboxAacDecoder(asc, frames_per_packet=1024)

    ref_out = bytearray()
    our_out = bytearray()
    for p in packets:
        data = bytes(p)
        rp = ref.decode(data)
        op = ours.decode(data)
        if rp:
            ref_out += rp
        if op:
            our_out += op

    print(f"\nreference (pmd3 native) : {len(ref_out)} bytes  rms={rms_dbfs(bytes(ref_out))} dBFS  peak={peak(bytes(ref_out))}")
    print(f"ours     (aac_darwin)    : {len(our_out)} bytes  rms={rms_dbfs(bytes(our_out))} dBFS  peak={peak(bytes(our_out))}")

    n = min(len(ref_out), len(our_out))
    if n == 0:
        print("FAIL: both decoders produced no PCM")
        return 1
    a = np.frombuffer(bytes(ref_out[:n]), dtype="<i2").astype(np.int64)
    b = np.frombuffer(bytes(our_out[:n]), dtype="<i2").astype(np.int64)
    diff = np.abs(a - b)
    max_abs = int(diff.max())
    # Ignore the first AU: the reference decoder primes with a converter delay.
    tail = slice(4096, n)
    tail_max = int(diff[tail].max()) if n > tail.stop else max_abs
    print(f"\ncompared {n} bytes; max |diff| = {max_abs}, tail max |diff| = {tail_max}")

    ok = tail_max <= 2
    print("PARITY OK" if ok else "PARITY DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
