import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import ArgumentParser
from tqdm import tqdm
from audioplayer import WavePlayerLoop


class SeamFinder():

    def __init__(self,
                 inputFileName:str,
                 dSamples:int = 10,
                 dTolerance:float = 0.001,
                 vSamples:int = 10,
                 vTolerance:float = 0.001,
                 candidateStep:int = 1,
                 validateStep:int = 10,
                 validateN:int = 10,
                 verbose:bool = False,
                 outputFileName:str = None
                 ):
        self.inputFileName: str = inputFileName
        self.dSamples: int = dSamples
        self.dTolerance: float = dTolerance
        self.vSamples: int = vSamples
        self.vTolerance: float = vTolerance
        self.candStep: int = candidateStep
        self.validateStep: int = validateStep
        self.validateN: int = validateN
        self.verbose: bool = verbose
        self.outputFileName: str = outputFileName

        self.minimumOverhead: int = max(self.dSamples, self.vSamples, self.validateN * self.validateStep)
        self.plotWidth: int = max(self.dSamples, self.vSamples) * 10
        self.seams: list[tuple[int,list[list[np.float64,np.float64]]]] = []
        self.audioPlayer = None

        sampleRate, data = self.readFile(self.inputFileName)
        self.data: np.ndarray = data
        self.sampleRate: int  = sampleRate

        if not self.outputFileName:
            self.outputFileName = self.inputFileName[:-4] + "_loop.wav"

    def verbosePrint(self, string):
        if self.verbose:
            print(string)

    def readFile(self, filePath):
        self.verbosePrint(f"Reading wav file: {filePath}")
        sampleRate, data = wavfile.read(filePath)
        self.verbosePrint(f"Sample rate: {sampleRate} Hz")
        self.verbosePrint(f"Length: {data.shape[0]} samples")
        self.verbosePrint(f"Channels: {data.shape[1]}")
        self.verbosePrint(f"Datatype: {data.dtype}")
        return sampleRate, data

    def startAudioLoop(self, fileName):
        self.audioPlayer = WavePlayerLoop(fileName)
        self.audioPlayer.play()

    def stopAudioLoop(self):
        if self.audioPlayer:
            self.audioPlayer.stop()

    def plotSeam(self, startIndex=0, endIndex=None):

        plt.figure(figsize=(10, 4))

        if not endIndex:
            endIndex = len(self.data)

        for channelIndex in range(self.data.shape[1]):

            sliced = self.data[startIndex:endIndex, channelIndex]

            plotHalf = int(self.plotWidth/2)

            end = sliced[len(sliced) - plotHalf:]
            endTime = np.linspace(0., len(end) / self.sampleRate, len(end))

            start = sliced[:plotHalf]
            startTime = np.linspace(endTime[-1], endTime[-1] + len(start) / self.sampleRate, len(start))
            
            plt.plot(startTime, start, label='Audio waveform at the start')
            plt.plot(endTime, end, label='Audio waveform at the end')
        
        # visualize derivative sample amount
        plt.axvline(endTime[-1] - max(self.dSamples, self.vSamples) / self.sampleRate, 0, 0.1)
        plt.axvline(endTime[-1] - max(self.dSamples, self.vSamples) / self.sampleRate, 0.9, 1)
        plt.axvline(endTime[-1] + max(self.dSamples, self.vSamples) / self.sampleRate, 0, 0.1)
        plt.axvline(endTime[-1] + max(self.dSamples, self.vSamples) / self.sampleRate, 0.9, 1)

        dataTypeMaxValue = np.iinfo(self.data.dtype).max
        plt.ylim([-dataTypeMaxValue, dataTypeMaxValue])

        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Waveform at the looping seam')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def validateSeam(self, seamIndex):
        validationResult = [0, 0, 0]
        for i in range(self.validateN):
            passed, diffs = self.trySeam(startIndex= i * self.validateStep, endIndex=seamIndex + i * self.validateStep)
            if passed:
                validationResult[0] += 1
            validationResult[1] += max(diffs[0][0], diffs[0][1])
            validationResult[2] += max(diffs[1][0], diffs[1][1])
        return validationResult

    def trySeam(self, startIndex=0, endIndex=None):
        
        if len(self.data) < endIndex + self.minimumOverhead:
            raise ValueError("Seam index too close to file end!")
        
        dataTypeMaxValue = np.iinfo(self.data.dtype).max
        derivativeTolerance = self.dTolerance * dataTypeMaxValue
        valueTolerance = self.vTolerance * dataTypeMaxValue

        valueStart = self.audioValue(startIndex)
        valueEnd = self.audioValue(endIndex)
        derivativeStart = self.audioDerivative(startIndex)
        derivativeEnd = self.audioDerivative(endIndex)

        diffs = []
        passes = True

        for channelIndex in range(self.data.shape[1]):
            valueDiff = abs(np.int64(valueStart[channelIndex]) - np.int64(valueEnd[channelIndex]))
            derivativeDiff = abs(derivativeStart[channelIndex] - derivativeEnd[channelIndex])
            if valueDiff > valueTolerance:
                passes = False
            if derivativeDiff > derivativeTolerance:
                passes = False
            diffs.append(np.float64(valueDiff) / dataTypeMaxValue)
            diffs.append(np.float64(derivativeDiff) / dataTypeMaxValue)
        return (passes, diffs)

    def audioDerivative(self, index):
        s1 = np.float64(self.data[index])
        s2 = np.float64(self.data[index + self.dSamples - 1])
        return (s2 - s1) / self.dSamples

    def audioValue(self, index):
        return np.average(self.data[index:index + self.vSamples], 0)

    def findSeam(self):
        index = len(self.data) - self.minimumOverhead
        for i in tqdm (range(math.floor(len(self.data)/self.candStep)), disable=(self.verbose != True), desc="Finding seams..."):
            passes, diffs = self.trySeam(endIndex=index)
            if passes:
                self.seams.append((index, diffs))
                return index
            index = index - self.candStep
        return 0

    def writeResult(self, cutIndex):
        self.verbosePrint(f"Writing result: {self.outputFileName}")
        wavfile.write(self.outputFileName, self.sampleRate, self.data[:cutIndex])


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("file", help="Input .wav file path")
    parser.add_argument("-ds", "--derivativesamples", default=10, type=int, help="Number of samples to use when calculating derivatives")
    parser.add_argument("-vs", "--valuesamples", default=10, type=int, help="Number of samples to use when calculating values")
    parser.add_argument("-dt", "--derivativetolerance", default=0.005, type=float, help="Tolerance when matching derivatives")
    parser.add_argument("-vt", "--valuetolerance", default=0.01, type=float, help="Tolerance when matching values")
    parser.add_argument("-cst", "--candidatestep", default=1, type=int, help="Step size in samples to use when finding initial points as seam candidates.")
    parser.add_argument("-vst", "--validatestep", default=10, type=int, help="Step size in samples to use when evaluating seam candidates.")
    parser.add_argument("-vn", "--validaten", default=10, type=int, help="Number of evaluation points to use.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the seam before and after adjustment.")
    parser.add_argument("-pa", "--plotaudio", action="store_true", help="Plot the seam and play the audio loop before and after adjustment.")
    parser.add_argument("-o", "--output", help="Output file. Defaults to input file with '_loop' postfix.")
    clargs = parser.parse_args()
    
    finder = SeamFinder(
        inputFileName=clargs.file,
        dSamples=clargs.derivativesamples,
        dTolerance=clargs.derivativetolerance,
        vSamples=clargs.valuesamples,
        vTolerance=clargs.valuetolerance,
        candidateStep=clargs.candidatestep,
        validateStep=clargs.validatestep,
        validateN=clargs.validaten,
        verbose=clargs.verbose,
        outputFileName=clargs.output
        )
    
    if clargs.plotaudio:
        finder.startAudioLoop(finder.inputFileName)
    if clargs.plot or clargs.plotaudio:
        finder.plotSeam()
    if clargs.plotaudio:
        finder.stopAudioLoop()

    seamIndex = finder.findSeam()

    if seamIndex == 0:
        print(f"Failed to find seam with given tolerances: Value tolerance {clargs.valuetolerance}, Derivative tolerance {clargs.derivativetolerance}")
    else:
        print(f"Seam found at {seamIndex}, {finder.data.shape[0]-seamIndex} samples or {(finder.data.shape[0]-seamIndex) / finder.sampleRate:.4f}s from the end of the file")
        print(f"Seam normalized value difference: {finder.seams[0][1][0]:.3f} {finder.seams[0][1][2]:.3f}")
        print(f"Seam normalized derivative difference: {finder.seams[0][1][1]:.3f} {finder.seams[0][1][3]:.3f}")

        finder.writeResult(seamIndex)

        if clargs.plotaudio:
            finder.startAudioLoop(finder.outputFileName)
        if clargs.plot or clargs.plotaudio:
            finder.plotSeam(endIndex=seamIndex)
        if clargs.plotaudio:
            finder.stopAudioLoop()
