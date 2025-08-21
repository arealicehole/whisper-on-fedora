# Mermaid Diagrams in Markdown - Implementation Guide

## Overview
Mermaid is a JavaScript-based diagramming tool that renders text definitions to create diagrams dynamically in Markdown files. GitHub natively supports Mermaid rendering.

## Syntax Reference

### Flowchart (Most Common for Architecture)
```mermaid
flowchart TD
    A[Start] --> B{Decision?}
    B -->|Yes| C[Process 1]
    B -->|No| D[Process 2]
    C --> E[End]
    D --> E
```

### Key Syntax Elements
- **Nodes**: `A[Text]` creates a rectangle node
- **Shapes**: 
  - `A[Rectangle]`
  - `B{Diamond/Decision}`
  - `C((Circle))`
  - `D([Stadium shape])`
- **Arrows**: 
  - `-->` solid arrow
  - `-.->` dotted arrow
  - `==>` thick arrow
  - `--Text-->` labeled arrow

### Architecture Diagram Pattern
```mermaid
flowchart LR
    subgraph Client["Client Layer"]
        A[Web App]
        B[Mobile App]
        C[CLI]
    end
    
    subgraph API["API Layer"]
        D[FastAPI Server]
        E[WebSocket Handler]
    end
    
    subgraph Services["Service Layer"]
        F[Whisper Model]
        G[Diarization Pipeline]
    end
    
    A --> D
    B --> D
    C --> D
    D --> F
    D --> G
```

## Implementation in README.md

### Basic Structure
````markdown
## Architecture

```mermaid
flowchart TD
    Start --> Process --> End
```
````

### Complex System Architecture Example
```mermaid
flowchart TD
    Audio[Audio Input<br/>WAV/MP3/M4A] --> API[FastAPI Server<br/>Port 8765]
    API --> Check{Diarization<br/>Requested?}
    Check -->|No| Whisper[Whisper Model<br/>GPU Accelerated]
    Check -->|Yes| Both[Whisper + Pyannote]
    Whisper --> Format[Output Formatter]
    Both --> Format
    Format --> JSON[JSON Response]
    Format --> Text[Plain Text]
    Format --> SRT[SRT Subtitles]
    Format --> VTT[VTT Subtitles]
    
    style Audio fill:#f9f,stroke:#333,stroke-width:2px
    style API fill:#bbf,stroke:#333,stroke-width:2px
    style Whisper fill:#bfb,stroke:#333,stroke-width:2px
    style Both fill:#fbf,stroke:#333,stroke-width:2px
```

## Best Practices

1. **Keep It Simple**: Start with basic shapes and connections
2. **Use Subgraphs**: Group related components
3. **Add Styling**: Use `style` commands for visual hierarchy
4. **Label Connections**: Add text on arrows for clarity
5. **Direction Matters**: Use TD (top-down), LR (left-right), etc.

## GitHub Rendering
- Wrap in triple backticks with `mermaid` language identifier
- GitHub automatically renders in README files
- Maximum size: Keep diagrams under 50 nodes for performance

## Common Patterns for API Documentation

### Request Flow
```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant W as Whisper
    participant D as Diarization
    
    C->>A: POST /v1/transcribe
    A->>W: Process Audio
    W-->>A: Transcription
    A->>D: Add Speakers (optional)
    D-->>A: Speaker Labels
    A-->>C: JSON Response
```

### Decision Tree
```mermaid
flowchart TD
    Start[User Request] --> GPU{GPU Available?}
    GPU -->|Yes| CUDA[Use CUDA Device]
    GPU -->|No| CPU[Use CPU Mode]
    CUDA --> Model{Model Size?}
    CPU --> Model
    Model -->|Tiny| Fast[Fast Processing<br/>Lower Accuracy]
    Model -->|Large| Slow[Slower Processing<br/>High Accuracy]
```

## Fallback for Non-Mermaid Environments

Always provide ASCII art alternative:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  API Server  │────▶│   Output    │
└─────────────┘     └──────────────┘     └─────────────┘
```

## References
- Official Docs: https://mermaid.js.org/
- GitHub Support: https://github.blog/developer-skills/github/include-diagrams-markdown-files-mermaid/
- Live Editor: https://mermaid.live/