# Whisper API Star Wars Inspired Versioning

## Current Version: `binary-sunset-v1`

## Version Naming Scheme

Drawing inspiration from Star Wars droids and tech:

### Major Versions (Breaking Changes)
- `binary-sunset-v1` - Current (dual speaker diarization like Tatooine's twin suns)
- `hyperdrive-v2` - Future major upgrade
- `lightspeed-v3` - Next generation
- `force-echo-v4` - Advanced features

### Minor Versions (Features)
- `v1.1-astromech` - Small fixes and improvements
- `v1.2-protocol` - API enhancements
- `v1.3-motivator` - Performance boosts
- `v1.4-restraining-bolt` - Security updates

### Patch Versions (Fixes)
- `v1.0.1-beep` - Tiny fixes
- `v1.0.2-boop` - More tiny fixes
- `v1.0.3-whistle` - Critical patches

## Why This Naming?

- **binary-sunset**: References the dual nature (transcription + diarization) like Tatooine's twin suns
- **Not literal droid names**: Avoids copyright issues on Akash
- **Tech-themed**: Keeps the sci-fi vibe without being too obvious
- **Semantic versioning compatible**: Still follows major.minor.patch structure

## For Akash Deployment

Use the full tag:
```yaml
image: whisper-blackwell:binary-sunset-v1
```

Or push to registry:
```bash
docker tag whisper-blackwell:binary-sunset-v1 yourusername/whisper-blackwell:binary-sunset-v1
docker push yourusername/whisper-blackwell:binary-sunset-v1
```