from pydub import AudioSegment

src = "example.mp3"
dst = "example.wav"

#convert mp3 to wav

audSeg = AudioSegment.from_mp3(src)
audSeg.export(dst, format="wav")

#Install ffmpeg for Ubuntu 20.04
""""
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
"""