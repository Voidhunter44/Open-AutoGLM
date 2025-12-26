#!/usr/bin/env python3
"""
Phone Agent CLI - AI-powered phone automation.

Usage:
    python main.py [OPTIONS]

Environment Variables:
    PHONE_AGENT_BASE_URL: Model API base URL (default: http://localhost:8000/v1)
    PHONE_AGENT_MODEL: Model name (default: autoglm-phone-9b)
    PHONE_AGENT_API_KEY: API key for model authentication (default: EMPTY)
    PHONE_AGENT_MAX_STEPS: Maximum steps per task (default: 100)
    PHONE_AGENT_DEVICE_ID: ADB device ID for multi-device setups
"""

import argparse
import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse

from openai import OpenAI

from phone_agent import PhoneAgent
from phone_agent.adb import ADBConnection, list_devices
from phone_agent.agent import AgentConfig
from phone_agent.config.apps import list_supported_apps
from phone_agent.config.config_ollama import OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL
from phone_agent.config.prompts_ollama import SYSTEM_PROMPT as OLLAMA_SYSTEM_PROMPT
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.model import ModelConfig


def check_system_requirements(device_type: DeviceType = DeviceType.ADB) -> bool:
    """
    Check system requirements before running the agent.

    Checks:
    1. ADB/HDC/iOS tools installed
    2. At least one device connected
    3. ADB Keyboard installed on the device (for ADB only)
    4. WebDriverAgent running (for iOS only)

    Args:
        device_type: Type of device tool (ADB, HDC, or IOS).

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking system requirements...")
    print("-" * 50)

    all_passed = True

    # Determine tool name and command
    if device_type == DeviceType.IOS:
        tool_name = "libimobiledevice"
        tool_cmd = "idevice_id"
    else:
        tool_name = "ADB" if device_type == DeviceType.ADB else "HDC"
        tool_cmd = "adb" if device_type == DeviceType.ADB else "hdc"

    # Check 1: Tool installed
    print(f"1. Checking {tool_name} installation...", end=" ")
    if shutil.which(tool_cmd) is None:
        print("❌ FAILED")
        print(f"   Error: {tool_name} is not installed or not in PATH.")

        if device_type == DeviceType.ADB:
            print("   Solution: Install Android SDK Platform Tools:")
            print("     - macOS: brew install android-platform-tools")
            print("     - Linux: sudo apt install android-tools-adb")
            print(
                "     - Windows: Download from https://developer.android.com/studio/releases/platform-tools"
            )
        elif device_type == DeviceType.HDC:
            print("   Solution: Install HarmonyOS SDK Platform Tools")
        else:  # IOS
            print("   Solution: Install libimobiledevice tools:")
            print("     - macOS: brew install libimobiledevice")
            print("     - Linux: sudo apt install libimobiledevice-dev")
        all_passed = False
    else:
        # Double check by running tool version
        try:
            result = subprocess.run(
                [tool_cmd, "version" if tool_cmd != "idevice_id" else "list"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.strip().split("\n")[0]
                print(f"✅ OK ({version_line})")
            else:
                print("❌ FAILED")
                print(f"   Error: {tool_name} command failed to run.")
                all_passed = False
        except FileNotFoundError:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command not found.")
            all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command timed out.")
            all_passed = False

    # If tool is not installed, skip remaining checks
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 2: Device connected
    print("2. Checking connected devices...", end=" ")
    try:
        if device_type == DeviceType.ADB:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            # Filter out header and empty lines, look for 'device' status
            devices = [line for line in lines[1:] if line.strip() and "\tdevice" in line]

            if not devices:
                print("❌ FAILED")
                print("   Error: No Android devices connected.")
                print("   Solution:")
                print("     1. Enable USB debugging on your Android device")
                print("     2. Connect via USB and authorize the connection")
                print("     3. Or connect remotely: python main.py --connect <ip>:<port>")
                all_passed = False
            else:
                device_ids = [d.split("\t")[0] for d in devices]
                print(f"✅ OK ({len(devices)} device(s): {', '.join(device_ids)})")
        elif device_type == DeviceType.HDC:
            result = subprocess.run(
                ["hdc", "list", "targets"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            # Filter out header and empty lines, look for devices
            devices = [line for line in lines if line.strip() and "offline" not in line.lower()]

            if not devices:
                print("❌ FAILED")
                print("   Error: No HarmonyOS devices connected.")
                print("   Solution:")
                print("     1. Enable USB debugging on your HarmonyOS device")
                print("     2. Connect via USB and authorize the connection")
                all_passed = False
            else:
                device_ids = [line.split()[0] for line in devices if line.strip()]
                print(f"✅ OK ({len(devices)} device(s): {', '.join(device_ids)})")
        else:  # DeviceType.IOS
            result = subprocess.run(
                ["idevice_id", "-l"], capture_output=True, text=True, timeout=10
            )
            device_ids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

            if not device_ids or (len(device_ids) == 1 and device_ids[0] == ""):
                print("❌ FAILED")
                print("   Error: No iOS devices connected.")
                print("   Solution:")
                print("     1. Connect iOS device via USB")
                print("     2. Trust the computer on the device if prompted")
                print("     3. Ensure iTunes/Finder recognizes the device")
                all_passed = False
            else:
                print(f"✅ OK ({len(device_ids)} device(s): {', '.join(device_ids[:3])}{'...' if len(device_ids) > 3 else ''})")
    except subprocess.TimeoutExpired:
        print("❌ FAILED")
        print(f"   Error: {tool_cmd} command timed out.")
        all_passed = False
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}")
        all_passed = False

    # If no device connected, skip further checks
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # For ADB devices, check ADB Keyboard
    if device_type == DeviceType.ADB:
        print("3. Checking ADB Keyboard...", end=" ")
        try:
            result = subprocess.run(
                ["adb", "shell", "ime", "list", "-s"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ime_list = result.stdout.strip()

            if "com.android.adbkeyboard/.AdbIME" in ime_list:
                print("✅ OK")
            else:
                print("❌ FAILED")
                print("   Error: ADB Keyboard is not installed on the device.")
                print("   Solution:")
                print("     1. Download ADB Keyboard APK from:")
                print(
                    "        https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk"
                )
                print("     2. Install it on your device: adb install ADBKeyboard.apk")
                print(
                    "     3. Enable it in Settings > System > Languages & Input > Virtual Keyboard"
                )
                all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print("   Error: ADB command timed out.")
            all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False

    # For iOS devices, check WDA
    if device_type == DeviceType.IOS:
        print("3. Checking WebDriverAgent...", end=" ")
        try:
            import requests
            response = requests.get("http://localhost:8100/status", timeout=5)
            if response.status_code == 200:
                print("✅ OK")
            else:
                print("⚠️  WARNING")
                print("   WDA may not be running. Some features may not work properly.")
                print("   Solution: Start WebDriverAgent on your iOS device.")
        except requests.exceptions.ConnectionError:
            print("⚠️  WARNING")
            print("   WDA is not accessible at http://localhost:8100")
            print("   Some features may not work properly.")
            print("   Solution: Start WebDriverAgent on your iOS device.")
        except Exception as e:
            print("⚠️  WARNING")
            print(f"   Error checking WDA: {e}")
            print("   Some features may not work properly.")

    print("-" * 50)

    if all_passed:
        print("✅ All system checks passed!\n")
    else:
        print("❌ System check failed. Please fix the issues above.")

    return all_passed


def validate_ollama_connectivity() -> bool:
    """
    Validate Ollama service connectivity before starting operations.
    
    Returns:
        True if Ollama is accessible, False otherwise.
    """
    print("🔍 Validating Ollama connectivity...")
    print("-" * 50)
    
    try:
        from openai import OpenAI  # Import here to avoid issues if not using Ollama
        
        # Test if Ollama is running by checking available models
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="EMPTY", timeout=30.0)
        
        # Try to list models as a connectivity test
        response = client.models.list()
        
        print(f"✅ Ollama service is accessible")
        print(f"   Base URL: {OLLAMA_BASE_URL}")
        
        # Check if required model is available
        model_available = any(model.id == OLLAMA_DEFAULT_MODEL for model in response.data)
        
        if model_available:
            print(f"✅ Required model '{OLLAMA_DEFAULT_MODEL}' is available")
        else:
            print(f"❌ Required model '{OLLAMA_DEFAULT_MODEL}' not found")
            print(f"   Available models: {[model.id for model in response.data]}")
            print(f"   Solution: Pull the required model with 'ollama pull {OLLAMA_DEFAULT_MODEL}'")
            print("-" * 50)
            return False
            
        print("-" * 50)
        print("✅ Ollama validation passed!\n")
        return True
        
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}")
        print("   Solution:")
        print("     1. Make sure Ollama service is running: 'ollama serve'")
        print("     2. Verify the model is pulled: 'ollama pull qwen3-vl:4b'")
        print("     3. Check if Ollama is accessible at http://localhost:11434")
        print("-" * 50)
        return False


def check_model_api(base_url: str, model_name: str, api_key: str = "EMPTY") -> bool:
    """
    Check if the model API is accessible and the specified model exists.

    Checks:
    1. Network connectivity to the API endpoint
    2. Model exists in the available models list

    Args:
        base_url: The API base URL
        model_name: The model name to check
        api_key: The API key for authentication

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking model API...")
    print("-" * 50)

    all_passed = True

    # Check 1: Network connectivity using chat API
    print(f"1. Checking API connectivity ({base_url})...", end=" ")
    try:
        # Create OpenAI client
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        # Use chat completion to test connectivity (more universally supported than /models)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        # Check if we got a valid response
        if response.choices and len(response.choices) > 0:
            print("✅ OK")
        else:
            print("❌ FAILED")
            print("   Error: Received empty response from API")
            all_passed = False

    except Exception as e:
        print("❌ FAILED")
        error_msg = str(e)

        # Provide more specific error messages
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"   Error: Cannot connect to {base_url}")
            print("   Solution:")
            print("     1. Check if the model server is running")
            print("     2. Verify the base URL is correct")
            print(f"     3. Try: curl {base_url}/chat/completions")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"   Error: Connection to {base_url} timed out")
            print("   Solution:")
            print("     1. Check your network connection")
            print("     2. Verify the server is responding")
        elif (
            "Name or service not known" in error_msg
            or "nodename nor servname" in error_msg
        ):
            print(f"   Error: Cannot resolve hostname")
            print("   Solution:")
            print("     1. Check the URL is correct")
            print("     2. Verify DNS settings")
        else:
            print(f"   Error: {error_msg}")

        all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ Model API checks passed!\n")
    else:
        print("❌ Model API check failed. Please fix the issues above.")

    return all_passed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Agent - AI-powered phone automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python main.py

    # Specify model endpoint
    python main.py --base-url http://localhost:8000/v1

    # Use API key for authentication
    python main.py --apikey sk-xxxxx

    # Run with specific device
    python main.py --device-id emulator-5554

    # Connect to remote device
    python main.py --connect 192.168.1.100:5555

    # List connected devices
    python main.py --list-devices

    # Enable TCP/IP on USB device and get connection info
    python main.py --enable-tcpip

    # List supported apps
    python main.py --list-apps
    
    # Use Ollama for local inference
    python main.py --use-ollama "Open Chrome browser"
        """,
    )

    # Model options
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        help="Model API base URL",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
        help="Model name",
    )

    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key for model authentication",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Maximum steps per task",
    )

    # Device options
    parser.add_argument(
        "--device-type",
        type=str,
        choices=["adb", "hdc", "ios"],
        default=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        help="Type of device connection: adb (Android), hdc (HarmonyOS), or ios (Apple)",
    )

    parser.add_argument(
        "--device-id",
        "-d",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="Device ID (for ADB, HDC, or iOS)",
    )

    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        metavar="ADDRESS",
        help="Connect to remote device (e.g., 192.168.1.100:5555 for ADB)",
    )

    parser.add_argument(
        "--disconnect",
        type=str,
        nargs="?",
        const="all",
        metavar="ADDRESS",
        help="Disconnect from remote device (or 'all' to disconnect all)",
    )

    parser.add_argument(
        "--list-devices", action="store_true", help="List connected devices and exit"
    )

    parser.add_argument(
        "--enable-tcpip",
        type=int,
        nargs="?",
        const=5555,
        metavar="PORT",
        help="Enable TCP/IP debugging on USB device (default port: 5555, ADB only)",
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--list-apps", action="store_true", help="List supported apps and exit"
    )
    
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama for local inference with qwen3-vl:4b model",
    )

    parser.add_argument(
        "--lang",
        type=str,
        choices=["cn", "en"],
        default=os.getenv("PHONE_AGENT_LANG", "en"),
        help="Language for system prompt (cn or en, default: en)",
    )

    parser.add_argument(
        "task",
        nargs="?",
        type=str,
        help="Task to execute (interactive mode if not provided)",
    )

    return parser.parse_args()


def handle_device_commands(args) -> bool:
    """
    Handle device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    # Map device type strings to enums
    device_type_map = {
        "adb": DeviceType.ADB,
        "hdc": DeviceType.HDC,
        "ios": DeviceType.IOS,
    }

    # Set the device type globally
    device_type = device_type_map[args.device_type]
    set_device_type(device_type)

    # Handle --list-devices
    if args.list_devices:
        device_factory = get_device_factory()
        try:
            devices = device_factory.list_devices()
            if not devices:
                print("No devices connected.")
            else:
                print(f"Connected {args.device_type.upper()} devices:")
                print("-" * 60)
                for device in devices:
                    status_icon = "✓" if device.status == "device" else "✗"
                    conn_type = device.connection_type.value
                    model_info = f" ({device.model})" if device.model else ""
                    print(
                        f"  {status_icon} {device.device_id:<30} [{conn_type}]{model_info}"
                    )
        except Exception as e:
            print(f"Error listing devices: {e}")
        return True

    # Handle --connect (ADB only for now)
    if args.connect:
        if device_type != DeviceType.ADB:
            print("--connect is only supported for ADB devices")
            return True

        print(f"Connecting to {args.connect}...")
        success, message = ADBConnection.connect(args.connect)
        print(f"{'✓' if success else '✗'} {message}")
        if success:
            # Set as default device
            args.device_id = args.connect
        return not success  # Continue if connection succeeded

    # Handle --disconnect (ADB only for now)
    if args.disconnect:
        if device_type != DeviceType.ADB:
            print("--disconnect is only supported for ADB devices")
            return True

        if args.disconnect == "all":
            print("Disconnecting all remote devices...")
            success, message = ADBConnection.disconnect()
        else:
            print(f"Disconnecting from {args.disconnect}...")
            success, message = ADBConnection.disconnect(args.disconnect)
        print(f"{'✓' if success else '✗'} {message}")
        return True

    # Handle --enable-tcpip (ADB only)
    if args.enable_tcpip:
        if device_type != DeviceType.ADB:
            print("--enable-tcpip is only supported for ADB devices")
            return True

        port = args.enable_tcpip
        print(f"Enabling TCP/IP debugging on port {port}...")

        success, message = ADBConnection.enable_tcpip(port, args.device_id)
        print(f"{'✓' if success else '✗'} {message}")

        if success:
            # Try to get device IP
            ip = ADBConnection.get_device_ip(args.device_id)
            if ip:
                print(f"\nYou can now connect remotely using:")
                print(f"  python main.py --connect {ip}:{port}")
                print(f"\nOr via ADB directly:")
                print(f"  adb connect {ip}:{port}")
            else:
                print("\nCould not determine device IP. Check device WiFi settings.")
        return True

    return False


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-apps (no system check needed)
    if args.list_apps:
        print("Supported apps:")
        for app in sorted(list_supported_apps()):
            print(f"  - {app}")
        return

    # Handle device commands (these may need partial system checks)
    if handle_device_commands(args):
        return

    # Map device type strings to enums
    device_type_map = {
        "adb": DeviceType.ADB,
        "hdc": DeviceType.HDC,
        "ios": DeviceType.IOS,
    }

    # Set the device type globally
    device_type = device_type_map[args.device_type]
    set_device_type(device_type)

    # Run system requirements check before proceeding
    if not check_system_requirements(device_type):
        sys.exit(1)

    # Handle Ollama flag - override base_url and model if using Ollama
    base_url = args.base_url
    model_name = args.model

    if args.use_ollama:
        base_url = OLLAMA_BASE_URL
        model_name = OLLAMA_DEFAULT_MODEL
        # Validate Ollama connectivity before proceeding
        if not validate_ollama_connectivity():
            sys.exit(1)
    else:
        # Check model API connectivity and model availability for non-Ollama
        if not check_model_api(base_url, model_name, args.apikey):
            sys.exit(1)

    # Create configurations
    model_config = ModelConfig(
        base_url=base_url,
        model_name=model_name,
        api_key=args.apikey,
        lang=args.lang,
    )

    # Determine system prompt based on whether using Ollama
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        device_id=args.device_id,
        verbose=not args.quiet,
        lang=args.lang,
    )

    if args.use_ollama:
        # Set Ollama-specific system prompt
        agent_config.system_prompt = OLLAMA_SYSTEM_PROMPT

    # Create agent based on device type
    if device_type == DeviceType.IOS:
        # Use iOS-specific agent
        from phone_agent.agent_ios import IOSPhoneAgent, IOSAgentConfig

        ios_agent_config = IOSAgentConfig(
            max_steps=args.max_steps,
            device_id=args.device_id,
            verbose=not args.quiet,
            lang=args.lang,
        )

        if args.use_ollama:
            ios_agent_config.system_prompt = OLLAMA_SYSTEM_PROMPT

        agent = IOSPhoneAgent(
            model_config=model_config,
            agent_config=ios_agent_config,
        )
    else:
        # Use standard PhoneAgent for ADB and HDC
        agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )

    # Print header
    print("=" * 50)
    print("Phone Agent - AI-powered phone automation")
    print("=" * 50)
    print(f"Model: {model_config.model_name}")
    print(f"Base URL: {model_config.base_url}")
    print(f"Max Steps: {agent_config.max_steps}")
    print(f"Language: {agent_config.lang}")
    print(f"Device Type: {args.device_type.upper()}")

    # Show device info
    device_factory = get_device_factory()
    devices = device_factory.list_devices()
    if agent_config.device_id:
        print(f"Device: {agent_config.device_id}")
    elif devices:
        print(f"Device: {devices[0].device_id} (auto-detected)")

    print("=" * 50)

    # Run with provided task or enter interactive mode
    if args.task:
        print(f"\nTask: {args.task}\n")
        result = agent.run(args.task)
        print(f"\nResult: {result}")
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.\n")

        while True:
            try:
                task = input("Enter your task: ").strip()

                if task.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not task:
                    continue

                print()
                result = agent.run(task)
                print(f"\nResult: {result}\n")
                agent.reset()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()