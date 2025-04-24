import matplotlib.pyplot as plt
import numpy as np
from audioplayer import WavePlayerLoop
from scipy.io import wavfile


class InteractiveResultViewer():

    def __init__(self, finder, results):
        self.results = results
        self.data: np.ndarray = finder.data
        self.sampleRate = finder.sampleRate
        self.plotWidth = finder.plotWidth
        self.vSamples = finder.vSamples
        self.dSamples = finder.dSamples
        self.volume = 100

        self.audioPlayer = None
        self.audioPlaying = False

    def inputLoop(self):
        currentCutIndex = None
        resultIndex = 0
        while True:
            print(f"Viewing result {resultIndex + 1}/{len(self.results)}")
            self.printDetails(self.results[resultIndex])
            currentCutIndex = self.results[resultIndex][0]
            print("Type h for list of commands")
            userInput = input(">")
            if userInput == "h":
                print("h: show this list")
                print("q: quit")
                print("l: start listening result")
                print("s: stop listening result")
                print("v: view plot of result")
                print("volume [0-100]: set volume")
                print("n: next")
            elif userInput == "q":
                self.stopAudioLoop()
                break
            elif userInput == "l":
                self.restartAudio(currentCutIndex)
            elif userInput == "s":
                self.stopAudioLoop()
            elif userInput == "v":
                self.plotStackedSeams(startIndex=0, endIndex=self.results[resultIndex][0])
            elif userInput == "n":
                resultIndex += 1
                if self.audioPlaying:
                    self.restartAudio(currentCutIndex)
                if resultIndex >= len(self.results):
                    self.stopAudioLoop()
                    break
            elif userInput.startswith("volume"):
                self.volume = max(0,min(100, int(userInput[7:])))
                if self.audioPlaying:
                    self.restartAudio(currentCutIndex)

    def restartAudio(self, endIndex):
        self.stopAudioLoop()
        self.writeFile("audio/temp.wav", startIndex=0, endIndex=endIndex)
        self.startAudioLoop("audio/temp.wav")

    def printDetails(self, result):
        seamIndex = result[0]
        score = result[1]
        diffs = result[2]
        print(f"Seam location {seamIndex}/{self.data.shape[0]} samples, {seamIndex/self.sampleRate:0.3f}/{self.data.shape[0]/self.sampleRate:0.3f} seconds")
        print(f"Seam normalized value difference: {diffs[0]:.3f} {diffs[2]:.3f}")
        print(f"Seam normalized derivative difference: {diffs[1]:.3f} {diffs[3]:.3f}")
        print(f"Seam score: {score[0]}, {score[1]:.3f}, {score[2]:.3f}")


    def setPlotScale(self, scale):
        self.plotScale = min(max(1, scale), 10000)

    def startAudioLoop(self, fileName):
        if self.audioPlaying:
            self.stopAudioLoop()
        self.audioPlayer = WavePlayerLoop(fileName)
        self.audioPlayer.play()
        self.audioPlaying = True

    def stopAudioLoop(self):
        if self.audioPlaying:
            self.audioPlayer.stop()
            self.audioPlayer = None
            self.audioPlaying = False
    
    def writeFile(self, fileName, startIndex=0, endIndex=None):
        if not endIndex:
            endIndex = len(self.data)
        dType = self.data.dtype
        wavfile.write(fileName, self.sampleRate, (self.data[startIndex:endIndex] * (self.volume/100)).astype(dType))

    def plotCombinedSeam(self, plt, startSlice, endSlice):
        sr = self.sampleRate
        startTime = np.linspace(endSlice[1] / sr, (endSlice[1] + (startSlice[1] - startSlice[0])) / sr, (startSlice[1] - startSlice[0]))
        endTime = np.linspace(endSlice[0] / sr, endSlice[1] / sr, (endSlice[1] - endSlice[0]))
        channelColors = ["royalblue", "violet"]
        for channelIndex in range(self.data.shape[1]):
            startAudioData = self.data[startSlice[0]:startSlice[1], channelIndex]
            endAudioData = self.data[endSlice[0]:endSlice[1], channelIndex]
            plt.plot(endTime, endAudioData, color=channelColors[channelIndex])
            plt.plot(startTime, startAudioData, color=channelColors[channelIndex])
        dataTypeMaxValue = np.iinfo(self.data.dtype).max
        plt.set_ylim([-dataTypeMaxValue, dataTypeMaxValue])
        plt.set_xlim([(endSlice[1] - self.plotWidth / 2) / sr,
                  (endSlice[1] + self.plotWidth / 2) / sr])
        plt.get_yaxis().set_visible(False)
    
    def plotSingleSeamEnd(self, plt, startIndex, endIndex, seamIndex):
        sr = self.sampleRate
        time = np.linspace(startIndex / sr, endIndex / sr, (endIndex - startIndex))

        channelColors = ["royalblue", "violet"]
        for channelIndex in range(self.data.shape[1]):
            audioData = self.data[startIndex:endIndex, channelIndex]
            plt.plot(time, audioData, color=channelColors[channelIndex])

        plt.axvline(seamIndex / sr, 0, 1, color="red")
        plt.axvline((seamIndex - self.vSamples) / sr, 0, 0.1, color="cyan")
        plt.axvline((seamIndex + self.vSamples) / sr, 0, 0.1, color="cyan")
        plt.axvline((seamIndex - self.dSamples) / sr, 0.9, 1, color="dodgerblue")
        plt.axvline((seamIndex + self.dSamples) / sr, 0.9, 1, color="dodgerblue")
        dataTypeMaxValue = np.iinfo(self.data.dtype).max
        plt.set_ylim([-dataTypeMaxValue, dataTypeMaxValue])
        plt.set_xlim([(seamIndex - self.plotWidth / 2) / sr,
                  (seamIndex + self.plotWidth / 2) / sr])
        plt.get_yaxis().set_visible(False)


    def plotStackedSeams(self, startIndex=0, endIndex=None):

        fig, axs = plt.subplots(3)
        fig.suptitle('Waveforms at the looping seam')

        if not endIndex:
            endIndex = len(self.data)

        plotHalf = int(self.plotWidth/2)
        startSlice = [max(startIndex - plotHalf, 0), min(startIndex + plotHalf, len(self.data))]
        endSlice = [max(endIndex - plotHalf, 0), min(endIndex + plotHalf, len(self.data))]

        padding = 10
        self.plotSingleSeamEnd(axs[0], startSlice[0], startSlice[1], startIndex)
        axs[0].set_title("Waveform at the start", pad=padding)
        self.plotSingleSeamEnd(axs[1], endSlice[0], endSlice[1], endIndex)
        axs[1].set_title("Waveform at the end", pad=padding)
        self.plotCombinedSeam(axs[2], startSlice, endSlice)
        axs[2].set_title("Combined seam result", pad=padding)

        fig.tight_layout()
        fig.show()
