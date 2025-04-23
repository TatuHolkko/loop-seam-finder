import os
import wave
import threading
import pyaudio

class WavePlayerLoop(threading.Thread):

  CHUNK = 1024

  def __init__(self,filepath,loop=True):
    super(WavePlayerLoop, self).__init__()
    self.filepath = os.path.abspath(filepath)
    self.loop = loop
    

  def run(self):
    
    wf = wave.open(self.filepath, 'rb')

    player = pyaudio.PyAudio()

    stream = player.open(format = player.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True)

    data = wf.readframes(self.CHUNK)
    while self.loop :
      stream.write(data)
      data = wf.readframes(self.CHUNK)
      if data == b'' : # If file is over then rewind.
        wf.rewind()
        data = wf.readframes(self.CHUNK)

    wf.close()
    stream.close()
    player.terminate()


  def play(self) :
    """
    Just another name for self.start()
    """
    self.start()

  def stop(self) :
    """
    Stop playback. 
    """
    self.loop = False