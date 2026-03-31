**#BreathClock**

Detects your breathing rate in real time using only your laptop microphone.

What it does??

1)Captures mic input and computes RMS amplitude per audio chunk
2)Runs a Butterworth low-pass filter to isolate breath signal from noise
3)Visualizes raw amplitude and filtered envelope on a live dark-theme plot
4)Detects each breath and displays rolling BPM

The microphone is an analog sensor - it converts air pressure into digital samples. The Butterworth filter strips everything above 0.5 Hz, leaving only the slow rise and fall of breathing. Same principle used in medical respiration monitors.
