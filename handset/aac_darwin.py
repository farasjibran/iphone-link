"""AAC / AAC-ELD decoding on macOS through the system AudioToolbox converter.

Windows decodes with the bundled LGPL ``zlaac.dll`` (see :mod:`handset.aac`).
macOS ships no FFmpeg binary in this tree, so decoding goes through Apple's own
``AudioConverter`` -- the very decoder CoreAudio uses for AAC-ELD.

One Access Unit in, one block of interleaved signed 16-bit little-endian PCM out.
"""

from __future__ import annotations

import ctypes
from typing import Optional

_AUDIO_TOOLBOX = "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"

# Core Audio format IDs. AAC-ELD ('aace') and plain AAC ('aac ') are distinct
# stream formats to AudioToolbox even though both are MPEG-4 AAC.
_FORMAT_AAC = 0x61616320  # 'aac '
_FORMAT_AAC_ELD = 0x61616365  # 'aace'

# AAC-ELD Access Unit == 480 samples (10 ms @ 48 kHz). The converter must be
# told this explicitly: left at the LPCM default of 1024 it reads every AU as a
# longer block and emits clipped garbage.
_ELD_FRAMES_PER_PACKET = 480

# MPEG-4 Audio samplingFrequencyIndex table (ISO/IEC 14496-3 Table 1.18).
_ASC_SAMPLE_RATES = (
    96000, 88200, 64000, 48000, 44100, 32000, 24000,
    22050, 16000, 12000, 11025, 8000, 7350,
)

# Audio Object Type 39 == "ER AAC ELD" (ISO/IEC 23003-5). AAC-ELD packs
# 480 samples per Access Unit; plain AAC-LC packs 1024.
_AOT_ELD = 39
_AOT_ELD_FRAMES_PER_PACKET = 480
_AOT_LC_FRAMES_PER_PACKET = 1024


class AacDarwinError(RuntimeError):
    """Raised when the AudioToolbox converter cannot be created or fed."""


def _fourcc(tag: str) -> int:
    return int.from_bytes(tag.encode("ascii"), "big")


def _asc_bits(asc: bytes) -> int:
    """First 32 bits of an AudioSpecificConfig, big-endian."""
    return int.from_bytes(asc[:4].ljust(4, b"\x00"), "big")


def asc_audio_object_type(asc: bytes) -> Optional[int]:
    """Return the Audio Object Type of an AudioSpecificConfig, or ``None``.

    ``audioObjectType`` is 5 bits; the value 31 escapes into 6 more bits, so
    the real type is ``32 + escape``.
    """
    if not asc or len(asc) < 2:
        return None
    bits = _asc_bits(asc)
    head = (bits >> 27) & 0x1F
    if head != 31:
        return head
    return 32 + ((bits >> 21) & 0x3F)


def asc_sample_rate(asc: bytes) -> Optional[int]:
    """Return the sample rate advertised by an AudioSpecificConfig, or ``None``.

    Index 15 means an explicit 24-bit rate follows, which this decoder does
    not support.
    """
    if not asc or len(asc) < 2:
        return None
    bits = _asc_bits(asc)
    head = (bits >> 27) & 0x1F
    pos = 5 if head != 31 else 11
    index = (bits >> (32 - pos - 4)) & 0x0F
    if index == 0x0F or index >= len(_ASC_SAMPLE_RATES):
        return None
    return _ASC_SAMPLE_RATES[index]


def asc_frames_per_packet(asc: bytes) -> int:
    """Samples per Access Unit implied by the AudioSpecificConfig."""
    aot = asc_audio_object_type(asc)
    return _AOT_ELD_FRAMES_PER_PACKET if aot == _AOT_ELD else _AOT_LC_FRAMES_PER_PACKET


class _AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ("mSampleRate", ctypes.c_double),
        ("mFormatID", ctypes.c_uint32),
        ("mFormatFlags", ctypes.c_uint32),
        ("mBytesPerPacket", ctypes.c_uint32),
        ("mFramesPerPacket", ctypes.c_uint32),
        ("mBytesPerFrame", ctypes.c_uint32),
        ("mChannelsPerFrame", ctypes.c_uint32),
        ("mBitsPerChannel", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


class _AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class _AudioBufferList(ctypes.Structure):
    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", _AudioBuffer * 1),
    ]


class _AudioStreamPacketDescription(ctypes.Structure):
    _fields_ = [
        ("mStartOffset", ctypes.c_int64),
        ("mVariableFramesInPacket", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
    ]


class AudioToolboxAacDecoder:
    """AAC / AAC-ELD Access Unit -> s16le PCM via an AudioToolbox ``AudioConverter``.

    The constructor signature matches pymobiledevice3's ``AACELDDecoder`` so it
    can stand in for it directly (``AACELDDecoder(magic_cookie)``).
    """

    def __init__(
        self,
        asc: bytes,
        sample_rate: int = 48000,
        channels: int = 2,
        out_rate: int = 48000,
        frames_per_packet: Optional[int] = None,
    ) -> None:
        asc_bytes = bytes(asc or b"")
        if not asc_bytes:
            raise AacDarwinError("AudioSpecificConfig (asc) must not be empty")

        self.asc = asc_bytes
        self.channels = int(channels)
        self.out_rate = int(out_rate)
        # The ASC decides the packet size: AAC-ELD packs 480 samples per AU,
        # plain AAC-LC packs 1024. Left at the LPCM default the converter reads
        # every AU as a longer block and emits clipped garbage.
        self._frames_per_packet = int(frames_per_packet or asc_frames_per_packet(asc_bytes))
        self._pcm_bytes = self._frames_per_packet * self.channels * 2

        # The ASC is the authority on the incoming rate; the caller may pass a
        # stale default (airplay hands us a 44.1 kHz cookie with 48 kHz out).
        self.sample_rate = int(asc_sample_rate(asc_bytes) or sample_rate)

        self._AT = ctypes.CDLL(_AUDIO_TOOLBOX)
        self._BufferList = _AudioBufferList
        self._APD = _AudioStreamPacketDescription

        # Source: AAC-ELD or plain AAC, mFramesPerPacket pinned to the AU size.
        source_format = _FORMAT_AAC_ELD if asc_audio_object_type(asc_bytes) == _AOT_ELD else _FORMAT_AAC
        src = _AudioStreamBasicDescription(
            float(self.sample_rate), source_format, 0, 0,
            self._frames_per_packet, 0, self.channels, 0, 0,
        )
        # Destination: packed signed 16-bit interleaved PCM.
        dst = _AudioStreamBasicDescription(
            float(self.out_rate), _fourcc("lpcm"), (1 << 2) | (1 << 3),
            self.channels * 2, 1, self.channels * 2, self.channels, 16, 0,
        )

        converter = ctypes.c_void_p()
        status = self._AT.AudioConverterNew(
            ctypes.byref(src), ctypes.byref(dst), ctypes.byref(converter)
        )
        if status != 0:
            raise AacDarwinError(f"AudioConverterNew failed: 0x{status:x}")
        self._converter = converter

        # "dmgc" == the decompression magic cookie, i.e. the AudioSpecificConfig.
        self._AT.AudioConverterSetProperty(
            converter, _fourcc("dmgc"), len(asc_bytes), asc_bytes
        )

        self._InputCB = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_AudioBufferList),
            ctypes.POINTER(ctypes.POINTER(_AudioStreamPacketDescription)),
            ctypes.c_void_p,
        )
        self._pending: Optional[bytes] = None
        self._pending_buf = None
        self._pending_pd = _AudioStreamPacketDescription(0, 0, 0)

        def _input_proc(
            converter_ref: object,
            io_packets: object,
            io_data: object,
            out_pd: object,
            _user: object,
        ) -> int:
            au = self._pending
            if au is None:
                io_packets[0] = 0
                return 0
            self._pending = None  # one AU per FillComplexBuffer call
            self._pending_buf = ctypes.create_string_buffer(au)
            io_packets[0] = 1
            io_data[0].mNumberBuffers = 1
            io_data[0].mBuffers[0].mNumberChannels = self.channels
            io_data[0].mBuffers[0].mDataByteSize = len(au)
            io_data[0].mBuffers[0].mData = ctypes.cast(self._pending_buf, ctypes.c_void_p)
            self._pending_pd = _AudioStreamPacketDescription(0, 0, len(au))
            if out_pd:
                out_pd[0] = ctypes.cast(ctypes.pointer(self._pending_pd), ctypes.POINTER(_AudioStreamPacketDescription))
            return 0

        self._callback = self._InputCB(_input_proc)  # keep the thunk alive
        self._AT.AudioConverterFillComplexBuffer.argtypes = [
            ctypes.c_void_p,
            self._InputCB,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_AudioBufferList),
            ctypes.c_void_p,
        ]

        self._out_buf = (ctypes.c_uint8 * self._pcm_bytes)()

    def decode(self, au: bytes) -> bytes:
        """Decode one Access Unit into interleaved s16le PCM.

        Returns an empty buffer for the priming call the converter makes before
        it has any output to give.
        """
        if not au:
            return b""
        self._pending = bytes(au)
        packets = ctypes.c_uint32(self._frames_per_packet)
        out = self._BufferList()
        out.mNumberBuffers = 1
        out.mBuffers[0].mNumberChannels = self.channels
        out.mBuffers[0].mDataByteSize = self._pcm_bytes
        out.mBuffers[0].mData = ctypes.cast(self._out_buf, ctypes.c_void_p)
        status = self._AT.AudioConverterFillComplexBuffer(
            self._converter, self._callback, None, ctypes.byref(packets), ctypes.byref(out), None
        )
        if status != 0:
            raise AacDarwinError(f"AudioConverterFillComplexBuffer: 0x{status:x}")
        size = int(out.mBuffers[0].mDataByteSize)
        return bytes(self._out_buf[:size]) if size else b""

    def close(self) -> None:
        converter = getattr(self, "_converter", None)
        self._converter = None
        if converter:
            self._AT.AudioConverterDispose(converter)

    def __enter__(self) -> "AudioToolboxAacDecoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
