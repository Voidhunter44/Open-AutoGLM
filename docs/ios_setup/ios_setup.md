# iOS Environment Setup Guide

This document explains how to configure an iOS device environment for Open-AutoGLM.

## Requirements

- macOS operating system
- Xcode (latest version, download from App Store)
- Apple Developer account (free account is sufficient, no payment required)
- iOS device (iPhone/iPad)
- USB data cable or same WiFi network

## WebDriverAgent Configuration

WebDriverAgent is the core component for iOS automation, which needs to run on the iOS device.

### 1. Clone WebDriverAgent

```bash
git clone https://github.com/facebook/WebDriverAgent.git
cd WebDriverAgent
```

### 2. Install Carthage

Carthage is required to build WebDriverAgent:

```bash
brew install carthage
```

### 3. Install Dependencies and Build

```bash
./Scripts/bootstrap.sh -d
```

### 4. Open in Xcode

```bash
open WebDriverAgent.xcworkspace
```

### 5. Configure Signing

1. Select the `WebDriverAgentRunner` target
2. In the `Signing & Capabilities` tab, enable `Automatically manage signing`
3. Select your team from the dropdown menu
4. Xcode will automatically create the required certificates and profiles

### 6. Build and Install

1. Make sure your iOS device is connected via USB
2. Select your iOS device as the destination (not simulator)
3. Click the `Build` button (Cmd+B) to build the project
4. Click the `Run` button (Cmd+R) to install WebDriverAgent on your device

### 7. Verify Installation

After successful installation, you should be able to access WebDriverAgent at:
```
http://<device-ip>:8100/status
```

You can find your device IP in Settings > WiFi > Connected Network (tap the "i" icon).

## Connecting iOS Device to Open-AutoGLM

### 1. Verify Device Connection

```bash
# List connected iOS devices
idevice_id -l
```

### 2. Test WebDriverAgent

```bash
# Test if WebDriverAgent is accessible
curl http://localhost:8100/status
```

### 3. Run Open-AutoGLM with iOS

```bash
# Use the iOS device type
python main.py --device-type ios --use-ollama "Open Safari and search for news"
```

## Troubleshooting

### Common Issues

1. **"No iOS devices found" error**:
   - Ensure the device is connected via USB
   - Check if the device trusts the computer (look for trust prompt on the device)
   - Verify iTunes/Finder recognizes the device

2. **"WebDriverAgent not running" error**:
   - Make sure WebDriverAgent is installed and running on the device
   - Check that the device and computer are on the same WiFi network
   - Verify the port 8100 is accessible

3. **"Failed to start session" error**:
   - Ensure proper code signing in Xcode
   - Rebuild WebDriverAgent after connecting a new device
   - Check Apple Developer account validity

### Required Tools

Install the following tools for iOS device management:

```bash
# Install libimobiledevice tools
brew install libimobiledevice

# Install ideviceinstaller (optional)
brew install ideviceinstaller

# Verify installation
idevice_id --help
```

## Notes

- iOS automation requires more setup than Android
- Ensure your iOS device remains unlocked during automation
- Some apps may have additional security restrictions
- For best results, use the latest iOS version supported by your device