import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import ArgumentParser
from tqdm import tqdm
from audioplayer import WavePlayerLoop


parser = ArgumentParser()
parser.add_argument("file", help="Input .wav file path")
parser.add_argument("-ds", "--derivativesamples", default=10, type=int, help="Number of samples to use when calculating derivatives")
parser.add_argument("-vs", "--valuesamples", default=10, type=int, help="Number of samples to use when calculating values")
parser.add_argument("-dt", "--dtolerance", default=0.005, type=float, help="Tolerance when matching derivatives")
parser.add_argument("-vt", "--vtolerance", default=0.01, type=float, help="Tolerance when matching values")
parser.add_argument("-st", "--step", default=1, type=int, help="Step size in samples")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-o", "--output", help="Output file. Defaults to input file with '_loop' postfix.")

args = parser.parse_args()

seams = []

plotWidth = max(args.valuesamples, args.derivativesamples) * 5

def verbosePrint(string):
    if args.verbose:
        print(string)

def readFile(filePath):
    verbosePrint(f"Reading wav file: {filePath}")
    sampleRate, data = wavfile.read(filePath)
    verbosePrint(f"Sample rate: {sampleRate} Hz")
    verbosePrint(f"Length: {data.shape[0]} samples")
    verbosePrint(f"Channels: {data.shape[1]}")
    verbosePrint(f"Datatype: {data.dtype}")
    return sampleRate, data

def plotSeam(data, sampleRate, seams=None):

    plt.figure(figsize=(10, 4))

    for channelIndex in range(data.shape[1]):
        channel = data[:, channelIndex]
        start = channel[:int(plotWidth/2)]
        end = channel[int(len(channel) - plotWidth/2):]
        endTime = np.linspace(0., len(end) / sampleRate, len(end))
        startTime = np.linspace(endTime[-1], endTime[-1] + len(start) / sampleRate, len(start))
        plt.plot(startTime, start, label='Audio waveform')
        plt.plot(endTime, end, label='Audio waveform')
    # visualize derivative sample amount
    plt.axvline(endTime[-1] - args.derivativesamples / sampleRate, 0, 0.1)
    plt.axvline(endTime[-1] - args.derivativesamples / sampleRate, 0.9, 1)
    plt.axvline(endTime[-1] + args.derivativesamples / sampleRate, 0, 0.1)
    plt.axvline(endTime[-1] + args.derivativesamples / sampleRate, 0.9, 1)

    dataTypeMaxValue = np.iinfo(data.dtype).max
    plt.ylim([-dataTypeMaxValue, dataTypeMaxValue])

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform at the looping seam')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def trySeam(data, seamIndex):
    
    if len(data) < seamIndex + max(args.derivativesamples, args.valuesamples):
        raise ValueError("Seam index too close to file end!")
    
    dataTypeMaxValue = np.iinfo(data.dtype).max
    derivativeTolerance = args.dtolerance * dataTypeMaxValue
    valueTolerance = args.vtolerance * dataTypeMaxValue

    valueStart = value(data, 0, args.derivativesamples)
    valueEnd = value(data, seamIndex, args.derivativesamples)
    derivativeStart = derivative(data, 0, args.derivativesamples)
    derivativeEnd = derivative(data, seamIndex, args.derivativesamples)

    diffs = []
    passes = True

    for channelIndex in range(data.shape[1]):
        valueDiff = abs(np.int64(valueStart[channelIndex]) - np.int64(valueEnd[channelIndex]))
        derivativeDiff = abs(derivativeStart[channelIndex] - derivativeEnd[channelIndex])
        if valueDiff > valueTolerance:
            passes = False
        if derivativeDiff > derivativeTolerance:
            passes = False
        diffs.append(np.float64(valueDiff) / dataTypeMaxValue)
        diffs.append(np.float64(derivativeDiff) / dataTypeMaxValue)
    return (passes, diffs)

def derivative(data, index, samples):
    s1 = np.float64(data[index])
    s2 = np.float64(data[index + samples])
    return (s2 - s1) / samples

def value(data, index, samples):
    return np.average(data[index:index + samples], 0)

def findSeam(data):
    step = args.step
    index = len(data) - max(args.valuesamples, args.derivativesamples) - 1
    for i in tqdm (range(math.floor(len(data)/step)), disable=(args.verbose != True), desc="Finding seams..."):
        passes, diffs = trySeam(data, index)
        if passes:
            seams.append((index, diffs))
            return index
        index = index - step
    return 0

def writeResult(data, sampleRate):
    fileName = None
    if args.output:
        fileName = args.output
    else:
        fileName = args.file[:-4] + "_loop.wav"
    print(f"Writing result: {fileName}")
    wavfile.write(fileName, sampleRate, data)
    return fileName


sampleRate, data = readFile(args.file)

audioPlayer = WavePlayerLoop(args.file)
audioPlayer.play()

if args.plot:
    plotSeam(data, sampleRate)

audioPlayer.stop()

seamIndex = findSeam(data)

if seamIndex == 0:
    print(f"Failed to find seam with given tolerances: Value tolerance {args.vtolerance}, Derivative tolerance {args.dtolerance}")
else:
    print(f"Seam found at {seamIndex}, {data.shape[0]-seamIndex} samples or {(data.shape[0]-seamIndex) / sampleRate:.4f}s from the end of the file")
    verbosePrint(f"Seam normalized value difference: {seams[0][1][0]:.3f} {seams[0][1][2]:.3f}")
    verbosePrint(f"Seam normalized derivative difference: {seams[0][1][1]:.3f} {seams[0][1][3]:.3f}")
    outputFile = writeResult(data[:seamIndex + 1], sampleRate)
    audioPlayer = WavePlayerLoop(outputFile)
    audioPlayer.play()
    if args.plot:
        plotSeam(data[:seamIndex + 1], sampleRate)
    audioPlayer.stop()

