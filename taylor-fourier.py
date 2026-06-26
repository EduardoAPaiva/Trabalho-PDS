import numpy as np
import matplotlib.pyplot as plt

def taylor_fourier_transform(
        signal,
        fs,
        f0,
        cycles=2,
        order=2):

    signal = np.asarray(signal)

    # Window length in samples
    window_size = int(round(cycles * fs / f0))

    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2

    # Local normalized time axis
    n = np.arange(-half, half + 1)
    tau = n / half

    omega = 2 * np.pi * f0 / fs

    carrier_pos = np.exp(1j * omega * n)
    carrier_neg = np.exp(-1j * omega * n)

    # Basis matrix
    B = np.zeros(
        (window_size, 2 * (order + 1)),
        dtype=complex
    )

    # Positive-frequency columns
    for k in range(order + 1):
        B[:, k] = (tau ** k) * carrier_pos

    # Negative-frequency columns    
    offset = order + 1
    for k in range(order + 1):
        B[:, offset + k] = (
            (tau ** k) * carrier_neg
        )

    B_pinv = np.linalg.pinv(B)

    centers = []
    coeffs = []

    for c in range(half, len(signal) - half):

        window = signal[
            c - half:
            c + half + 1
        ]

        p = B_pinv @ window

        centers.append(c)
        coeffs.append(p)

    return (
        np.asarray(centers),
        np.asarray(coeffs)
    )


ordem = 4
ciclos = 4

with open("sinais/sinal_4_semruido.csv") as f:
    array = [float(linha.strip()) for linha in f]

time = np.arange(0, len(array))/(60.0 * 128.0)

# plt.plot(time, array)
# plt.show()

centers, coeffs = taylor_fourier_transform(array, 60*128, 60, order=ordem, cycles=ciclos)
tf_time = time[centers[0]: centers[-1] + 1]

c0 = np.abs(coeffs[:,0])
c1 = np.abs(coeffs[:,1])

maior_ind = 0
maior_val = np.abs(c1[0])
for i in range(0, len(c1)):
    if maior_val < np.abs(c1[i]):
        maior_val = np.abs(c1[i])
        maior_ind = i

tempo = time[centers[maior_ind]]
print(f"Maior c1 encontrado em {tempo}s")

plt.plot(tf_time, c0)
plt.title(f"Valor absoluto de C0 para ordem {ordem} e {ciclos} períodos")
plt.show()

plt.plot(tf_time, c1)
plt.title(f"Valor absoluto de C1 para ordem {ordem} e {ciclos} períodos")
plt.show()
