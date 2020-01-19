import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.fftpack import dct

# If you want to see the spectrogram picture
import matplotlib
# matplotlib.use('Agg')


def plot_spectrogram(spec, note, file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


# preemphasis config
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=16kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function


def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """

    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i*frame_shift:i*frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win

    return frames


def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum


def melFilterBank(fs=16000, fft_len=fft_len, num_filter=num_filter):
    """Create melFilterBank 在梅尔域上平均取点转回频域做中心点
        :param fs: sample frequency
        :param fft_len: Enframe config, default 512
        :param num_filter: mel filters number, default 23
        :returns: fbank , a fft_len/2 + 1 by num_filter array 
        :returns: fcenters, a num_filter array 
    """

    def hz2mel(f):
        return 2595 * np.log(f / 700.0 + 1.0)

    def mel2hz(m):
        return 700 * (np.exp(m / 2595) - 1.0)

    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = int(fft_len / 2 + 1)
    df = fs / fft_len
    dmel = melmax / (num_filter + 1)
    melcenters = np.arange(1, num_filter + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:num_filter - 1]))
    indexstop = np.hstack((indexcenter[1:num_filter], [nmax]))
    filterbank = np.zeros((num_filter, nmax))
    # print(indexstop)
    for c in range(0, num_filter):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    for c in np.arange(0, num_filter):
        plt.plot(np.arange(0, fft_len / 2 + 1) * df, filterbank[c])
    plt.show()
    return filterbank, fcenters


def fbank(spectrum, num_filter=num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    fbank, _ = melFilterBank(fft_len=fft_len, num_filter=num_filter)
    feats = np.dot(spectrum, fbank.T)
    feats = 10 * np.log10(feats)
    print(feats.shape)

    return feats


def mfcc(fbank, num_mfcc=num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    # feats = np.zeros((fbank.shape[0], num_mfcc))
    feats = dct(fbank)
    feats = feats[:, :num_mfcc]
    return feats


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f = open(file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(fbank_feats, 'Filter Bank', 'fbank.png')
    write_file(fbank_feats, './test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC', 'mfcc.png')
    write_file(mfcc_feats, './test.mfcc')
    # librosa里的mfcc，用于做比较
    # mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=num_mfcc,
    #                              win_length=frame_len, hop_length=frame_shift, n_mels=23, n_fft=512)
    # plot_spectrogram(mfccs, 'MFCC', 'mfccs.png')
    # write_file(mfccs, './tests.mfcc')


if __name__ == '__main__':
    main()
