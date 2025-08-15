import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch, detrend
from scipy.integrate import simpson
import matplotlib.pyplot as plt

def hr_to_hrv_freq_features(hr_bpm, sampling_rate_hz=1/1.5, interp_rate_hz=4.0, plot_psd=True):
    """
    Convert HR signal to approximate HRV frequency-domain features (VLF, LF, HF).

    Parameters:
        hr_bpm: list or np.array of heart rate values in bpm
        sampling_rate_hz: sampling rate of the HR signal (default 1 Hz)
        interp_rate_hz: target sampling rate for interpolation (default 4 Hz)
        plot_psd: if True, plots power spectral density with HRV bands

    Returns:
        dict: {'VLF': ..., 'LF': ..., 'HF': ..., 'LF/HF': ..., 'VLF_rel': ..., 'LF_rel': ..., 'HF_rel': ...}
    """
    if len(hr_bpm) < sampling_rate_hz * 240:
        raise ValueError(f"Need at least 160 seconds of HR data for reliable frequency-domain HRV. "
                     f"Required: {sampling_rate_hz * 240}, Found: {len(hr_bpm)}")


    hr_bpm = np.array(hr_bpm)
    rr_ms = 60000 / hr_bpm  # Convert HR to RR in ms
    times = np.arange(len(rr_ms)) / sampling_rate_hz

    # Interpolate RR to a uniform sampling rate
    interp_func = interp1d(times, rr_ms, kind='cubic', fill_value='extrapolate')
    resampled_times = np.arange(times[0], times[-1], 1.0 / interp_rate_hz)
    resampled_rr = interp_func(resampled_times)

    # Remove low-frequency trends
    resampled_rr_detrended = detrend(resampled_rr)

    # Welch PSD estimate
    freqs, psd = welch(resampled_rr_detrended, fs=interp_rate_hz, nperseg=min(256, len(resampled_rr)))

    # Define HRV bands
    vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
    lf_band  = (freqs >= 0.04)   & (freqs < 0.15)
    hf_band  = (freqs >= 0.15)   & (freqs < 0.4)

    # Integrate PSD in each band
    vlf_power = simpson(psd[vlf_band], freqs[vlf_band])
    lf_power  = simpson(psd[lf_band], freqs[lf_band])
    hf_power  = simpson(psd[hf_band], freqs[hf_band])

    total_power = vlf_power + lf_power + hf_power
    lf_norm = lf_power / total_power
    hf_norm = hf_power / total_power
    vlf_norm = vlf_power / total_power

    return {
        "HF": hf_norm,
        "LF":lf_norm,
        "VLF": vlf_norm,
    }
