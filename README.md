# Diarize Project

This project is a Python application for audio diarization.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Add Hugging Face API key to `.env`
- Install `ffmpeg`

## Installing Diarize

To install Diarize, follow these steps:

1. Clone the repository
2. Navigate to the project directory
3. Run `pip install -r requirements.txt`

## Using Diarize

To use Diarize, run:

```bash
python diarize.py input.wav -d cpu
```

Sample print output:

```
~/diarizer$ ./diarize.py data/output1.wav --print
torchvision is not available - cannot save figures
INFO:lib.diarizer:Transcribing
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6000/6000 [00:09<00:00, 627.19frames/s]
INFO:lib.diarizer:Aligning Segments
Progress: 53.33%...
Progress: 56.67%...
Progress: 60.00%...
Progress: 63.33%...
Progress: 66.67%...
Progress: 70.00%...
Progress: 73.33%...
Progress: 76.67%...
Progress: 80.00%...
Progress: 83.33%...
Progress: 86.67%...
Progress: 90.00%...
Progress: 93.33%...
Progress: 96.67%...
Progress: 100.00%...
INFO:lib.diarizer:Diarizing
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmppbacq_y2
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmppbacq_y2/_remote_module_non_scriptable.py
INFO:lib.diarizer:Assigning Speakers
Segment 1:
Start time: 0.58
End time: 6.62
Speaker: SPEAKER_01
Transcript:  I will say, David, I would love to have NVIDIA's full production team every episode.

Segment 2:
Start time: 7.06
End time: 12.98
Speaker: SPEAKER_01
Transcript:  It was nice not having to worry about turning the cameras on and off and making sure that nothing bad happened to myself while we were recording this.

Segment 3:
Start time: 13.70
End time: 15.19
Speaker: SPEAKER_02
Transcript:  Yeah, just the gear.

Segment 4:
Start time: 15.19
End time: 17.98
Speaker: SPEAKER_02
Transcript: I mean, the drives that came out of the camera.

Segment 5:
Start time: 18.00
End time: 22.14
Speaker: SPEAKER_01
Transcript:  All right, red cameras for the home studio starting next episode.

Segment 6:
Start time: 23.22
End time: 23.86
Speaker: SPEAKER_01
Transcript:  Yeah, great.

Segment 7:
Start time: 24.00
End time: 25.00
Speaker: SPEAKER_00
Transcript:  All right, let's do it.

Segment 8:
Start time: 25.16
End time: 27.63
Speaker: SPEAKER_00
Transcript:  Who got the truth?

Segment 9:
Start time: 27.63
End time: 28.27
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 10:
Start time: 28.27
End time: 28.91
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 11:
Start time: 28.91
End time: 29.63
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 12:
Start time: 29.63
End time: 31.07
Speaker: SPEAKER_00
Transcript: Who got the truth now?

Segment 13:
Start time: 31.07
End time: 32.72
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 14:
Start time: 32.72
End time: 33.38
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 15:
Start time: 33.38
End time: 33.80
Speaker: SPEAKER_00
Transcript: Is it you?

Segment 16:
Start time: 34.04
End time: 39.88
Speaker: SPEAKER_00
Transcript:  Sit me down, say it straight, another story on the way.

Segment 17:
Start time: 39.88
End time: 40.84
Speaker: SPEAKER_00
Transcript: Who got the truth?

Segment 18:
Start time: 41.42
End time: 46.96
Speaker: SPEAKER_01
Transcript:  Welcome to this episode of Acquired, the podcast about great technology companies and the stories and playbooks behind them.

Segment 19:
Start time: 47.41
End time: 48.00
Speaker: SPEAKER_01
Transcript:  I'm Ben Gilbert.

Segment 20:
Start time: 48.43
End time: 49.00
Speaker: SPEAKER_02
Transcript:  I'm David Rosenthal.

Segment 21:
Start time: 49.69
End time: 50.80
Speaker: SPEAKER_01
Transcript:  And we are your hosts.

Segment 22:
Start time: 51.42
End time: 56.00
Speaker: SPEAKER_01
Transcript:  Listeners, just so we don't bury the lead, this episode was insanely cool for David and I.

Segment 23:
Start time: 56.00
End time: 56.12
Speaker: SPEAKER_01
Transcript:  Yeah.

Segment 24:
Start time: 58.11
End time: 59.88
Speaker: SPEAKER_01
Transcript:  After researching NVIDIA for something like...
```

Sample JSON output:

```json
[{"start": 0.582, "end": 6.619, "text": " I will say, David, I would love to have NVIDIA's full production team every episode.", "speaker": "SPEAKER_01"}, {"start": 7.06, "end": 12.98, "text": " It was nice not having to worry about turning the cameras on and off and making sure that nothing bad happened to myself while we were recording this.", "speaker": "SPEAKER_01"}, {"start": 13.703, "end": 15.189, "text": " Yeah, just the gear.", "speaker": "SPEAKER_02"}, {"start": 15.189, "end": 17.98, "text": "I mean, the drives that came out of the camera.", "speaker": "SPEAKER_02"}, {"start": 18.0, "end": 22.137, "text": " All right, red cameras for the home studio starting next episode.", "speaker": "SPEAKER_01"}, {"start": 23.224, "end": 23.857, "text": " Yeah, great.", "speaker": "SPEAKER_01"}, {"start": 24.0, "end": 25.0, "text": " All right, let's do it.", "speaker": "SPEAKER_00"}, {"start": 25.16, "end": 27.626, "text": " Who got the truth?", "speaker": "SPEAKER_00"}, {"start": 27.626, "end": 28.267, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 28.267, "end": 28.909, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 28.909, "end": 29.63, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 29.63, "end": 31.073, "text": "Who got the truth now?", "speaker": "SPEAKER_00"}, {"start": 31.073, "end": 32.717, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 32.717, "end": 33.379, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 33.379, "end": 33.8, "text": "Is it you?", "speaker": "SPEAKER_00"}, {"start": 34.04, "end": 39.877, "text": " Sit me down, say it straight, another story on the way.", "speaker": "SPEAKER_00"}, {"start": 39.877, "end": 40.84, "text": "Who got the truth?", "speaker": "SPEAKER_00"}, {"start": 41.421, "end": 46.96, "text": " Welcome to this episode of Acquired, the podcast about great technology companies and the stories and playbooks behind them.", "speaker": "SPEAKER_01"}, {"start": 47.408, "end": 48.0, "text": " I'm Ben Gilbert.", "speaker": "SPEAKER_01"}, {"start": 48.429, "end": 49.0, "text": " I'm David Rosenthal.", "speaker": "SPEAKER_02"}, {"start": 49.687, "end": 50.798, "text": " And we are your hosts.", "speaker": "SPEAKER_01"}, {"start": 51.422, "end": 56.0, "text": " Listeners, just so we don't bury the lead, this episode was insanely cool for David and I.", "speaker": "SPEAKER_01"}, {"start": 56.0, "end": 56.122, "text": " Yeah.", "speaker": "SPEAKER_01"}, {"start": 58.107, "end": 59.879, "text": " After researching NVIDIA for something like...", "speaker": "SPEAKER_01"}]
```

## LICENSE

This code is licensed under the MIT License. See LICENSE.txt file for details.
