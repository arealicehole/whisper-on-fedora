# Whisper API Examples

This directory contains example scripts showing how to use the Whisper API.

## Examples

### basic_usage.py
Complete Python examples showing:
- Basic transcription
- Speaker diarization
- Output formatting
- Async processing
- Export formats

### Command Line Examples

```bash
# Basic transcription
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@audio.wav" \
  -o transcript.json

# With speaker detection
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@meeting.wav" \
  -F "diarize=true" \
  -F "num_speakers=3" \
  -o meeting_transcript.json

# Get subtitles
curl -X POST http://localhost:8765/v1/transcribe \
  -F "file=@video.wav" \
  -F "format=srt" \
  -o subtitles.srt
```

### Integration Examples

#### Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribe(filePath, diarize = false) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    if (diarize) {
        form.append('diarize', 'true');
    }
    
    const response = await axios.post(
        'http://localhost:8765/v1/transcribe',
        form,
        { headers: form.getHeaders() }
    );
    
    return response.data;
}

// Usage
transcribe('meeting.wav', true)
    .then(result => {
        result.segments.forEach(seg => {
            console.log(`${seg.speaker}: ${seg.text}`);
        });
    });
```

#### PHP
```php
<?php
$curl = curl_init();

$file = new CURLFile('audio.wav');
$data = [
    'file' => $file,
    'diarize' => 'true',
    'format' => 'json'
];

curl_setopt_array($curl, [
    CURLOPT_URL => 'http://localhost:8765/v1/transcribe',
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $data
]);

$response = curl_exec($curl);
$result = json_decode($response, true);

foreach ($result['segments'] as $segment) {
    echo $segment['speaker'] . ': ' . $segment['text'] . "\n";
}
?>
```

#### Go
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

func transcribeAudio(filename string, diarize bool) (map[string]interface{}, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    body := &bytes.Buffer{}
    writer := multipart.NewWriter(body)
    
    part, err := writer.CreateFormFile("file", filename)
    if err != nil {
        return nil, err
    }
    io.Copy(part, file)
    
    if diarize {
        writer.WriteField("diarize", "true")
    }
    writer.Close()

    req, err := http.NewRequest("POST", "http://localhost:8765/v1/transcribe", body)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", writer.FormDataContentType())

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    return result, nil
}
```

## Testing Different Scenarios

### Meeting Transcription
```python
# For meetings with multiple speakers
result = client.transcribe(
    "board_meeting.wav",
    diarize=True,
    num_speakers=5,  # Helps accuracy if known
    language="en"
)
```

### Podcast Processing
```python
# For podcasts with 2-3 hosts
result = client.transcribe(
    "podcast_episode.mp3",
    diarize=True,
    num_speakers=2
)

# Export for editing
with open("episode_transcript.txt", "w") as f:
    f.write(client.format_transcript(result, "dialogue"))
```

### Video Subtitles
```python
# Generate subtitles for video
srt = client.transcribe(
    "video_audio.wav",
    diarize=False,  # Usually not needed for subtitles
    format="srt"
)

with open("video.srt", "w") as f:
    f.write(srt)
```

### Interview Analysis
```python
# Analyze interview with speaker separation
result = client.transcribe(
    "interview.wav",
    diarize=True,
    num_speakers=2
)

# Separate interviewer and interviewee
for segment in result['segments']:
    if segment['speaker'] == 'SPEAKER_00':
        # Likely the interviewer (usually speaks first)
        print(f"Q: {segment['text']}")
    else:
        print(f"A: {segment['text']}")
```