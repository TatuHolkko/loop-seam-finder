import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import ArgumentParser
from tqdm import tqdm
from functools import cmp_to_key

from interactive import InteractiveResultViewer

class SeamFinder():

    def __init__(self,
                 inputFileName:str,
                 outputFileName:str = None,
                 dSamples:int = 10,
                 dTolerance:float = 0.001,
                 vSamples:int = 10,
                 vTolerance:float = 0.001,
                 candidateStep:int = 1,
                 candidateMax: float = 0.5,
                 validateStep:int = 10,
                 validateN:int = 10,
                 plotScale: float = 10,
                 verbose:bool = False
                 ):
        self.inputFileName: str = inputFileName
        self.outputFileName: str = outputFileName
        self.dSamples: int = dSamples
        self.dTolerance: float = dTolerance
        self.vSamples: int = vSamples
        self.vTolerance: float = vTolerance
        self.candStep: int = candidateStep
        self.candMax: float = candidateMax
        self.validateStep: int = validateStep
        self.validateN: int = validateN
        self.verbose: bool = verbose
        self.plotScale: float =  plotScale

        self.minimumOverhead: int = max(self.dSamples, self.vSamples) + self.validateN * self.validateStep
        self.plotWidth: int = max(self.dSamples, self.vSamples) * self.plotScale
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


    def score(self, seamIndex):
        seamScore = [0, 0, 0]
        for i in range(self.validateN):
            passed, diffs = self.trySeam(startIndex= i * self.validateStep, endIndex=seamIndex + i * self.validateStep)
            if passed:
                seamScore[0] += 1
            seamScore[1] += max(diffs[0], diffs[2])
            seamScore[2] += max(diffs[1], diffs[3])
        return seamScore

    def compareSeamScores(self, score1, score2):
        if score1[0] > score2[0]:
            return -1
        elif score1[0] < score2[0]:
            return 1
        return (score1[1] + score1[2]) - (score2[1] + score2[2])

    def trySeam(self, startIndex=0, endIndex=None):
        if len(self.data) < endIndex + max(self.vSamples, self.dSamples):
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
        for i in tqdm(range(math.floor(self.candMax * len(self.data)/self.candStep)), disable=(self.verbose != True), desc="Finding seams..."):
            passes, diffs = self.trySeam(endIndex=index)
            if passes:
                self.seams.append((index, diffs))
            index = index - self.candStep
        self.verbosePrint(f"Found {len(self.seams)} seam candidates.")
        if not self.seams:
            return 0
        scoredSeams = []
        for i in tqdm(range(len(self.seams)), disable=(self.verbose != True), desc="Evaluating candidates..."):
            seamScore = self.score(self.seams[i][0])
            scoredSeams.append((self.seams[i][0], seamScore, self.seams[i][1]))
        sortedSeams = sorted(scoredSeams, key=cmp_to_key(lambda scoredSeam1, scoredSeam2: self.compareSeamScores(scoredSeam1[1], scoredSeam2[1])))
        return sortedSeams

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
    parser.add_argument("-cm", "--candidatemax", default=0.5, type=float, help="Maximum proportional [0,1] distance from the end to look for candidates.")
    parser.add_argument("-vst", "--validatestep", default=10, type=int, help="Step size in samples to use when evaluating seam candidates.")
    parser.add_argument("-vn", "--validaten", default=10, type=int, help="Number of evaluation points to use.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the seam before and after adjustment.")
    parser.add_argument("-ps", "--plotscale", default=10, type=float, help="Plot viewing scale.")
    parser.add_argument("-pa", "--plotaudio", action="store_true", help="Plot the seam and play the audio loop before and after adjustment.")
    parser.add_argument("-o", "--output", help="Output file. Defaults to input file with '_loop' postfix.")
    clargs = parser.parse_args()
    
    finder = SeamFinder(
        inputFileName=clargs.file,
        outputFileName=clargs.output,
        dSamples=clargs.derivativesamples,
        dTolerance=clargs.derivativetolerance,
        vSamples=clargs.valuesamples,
        vTolerance=clargs.valuetolerance,
        candidateStep=clargs.candidatestep,
        candidateMax=clargs.candidatemax,
        validateStep=clargs.validatestep,
        validateN=clargs.validaten,
        verbose=clargs.verbose,
        plotScale=clargs.plotscale
        )
    


    results = finder.findSeam()

    if not results:
        print(f"Failed to find seam with given tolerances: Value tolerance {clargs.valuetolerance}, Derivative tolerance {clargs.derivativetolerance}")
    else:
        viewer = InteractiveResultViewer(finder, results)
        viewer.inputLoop()
