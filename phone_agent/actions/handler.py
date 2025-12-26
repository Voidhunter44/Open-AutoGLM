"""Action handlers for PhoneAgent operations."""

import ast
from dataclasses import dataclass
from typing import Any


@dataclass
class ActionResult:
    """Result of an action execution."""

    success: bool
    should_finish: bool = False
    message: str = ""
    extra_info: dict[str, Any] = None
import subprocess
from typing import Any

from phone_agent.adb import ADBConnection
from phone_agent.config import get_messages
from phone_agent.hdc import HDCConnection


class ActionHandler:
    """
    Handler for phone operations (tap, type, swipe, etc.).

    Args:
        device_id: ADB/HDC device ID for multi-device setups.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for manual operation requests.
    """

    def __init__(
        self,
        device_id: str | None = None,
        confirmation_callback=None,
        takeover_callback=None,
    ):
        self.device_id = device_id
        self.confirmation_callback = confirmation_callback
        self.takeover_callback = takeover_callback

        # Detect device type automatically
        self._detect_device_type()

    def _detect_device_type(self) -> None:
        """Detect whether the device is ADB (Android) or HDC (HarmonyOS) based."""
        if self.device_id:
            if ":" in self.device_id and not self.device_id.startswith("emulator"):
                # Remote device - likely ADB for Android
                self.device_type = "adb"
            else:
                # Check if device ID exists in ADB or HDC
                try:
                    result = subprocess.run(
                        ["adb", "devices"], capture_output=True, text=True
                    )
                    if self.device_id in result.stdout:
                        self.device_type = "adb"
                    else:
                        result = subprocess.run(
                            ["hdc", "list", "targets"], capture_output=True, text=True
                        )
                        if self.device_id in result.stdout:
                            self.device_type = "hdc"
                        else:
                            # Default to ADB if not found
                            self.device_type = "adb"
                except:
                    # If commands fail, default to ADB
                    self.device_type = "adb"
        else:
            # Auto-detect using available devices
            try:
                result = subprocess.run(
                    ["adb", "devices"], capture_output=True, text=True
                )
                adb_devices = [
                    line.split()[0]
                    for line in result.stdout.strip().split("\n")[1:]
                    if line.strip() and "\tdevice" in line
                ]

                if adb_devices:
                    self.device_type = "adb"
                    if not self.device_id and adb_devices:
                        self.device_id = adb_devices[0]
                else:
                    result = subprocess.run(
                        ["hdc", "list", "targets"], capture_output=True, text=True
                    )
                    hdc_devices = [
                        line.split()[0]
                        for line in result.stdout.strip().split("\n")
                        if line.strip() and "typ:hw" in line
                    ]
                    
                    if hdc_devices:
                        self.device_type = "hdc"
                        if not self.device_id and hdc_devices:
                            self.device_id = hdc_devices[0]
                    else:
                        # Default to ADB if no devices found
                        self.device_type = "adb"
            except:
                # If detection fails, default to ADB
                self.device_type = "adb"

    def execute(self, action: dict[str, Any], screen_width: int, screen_height: int):
        """
        Execute an action on the phone.

        Args:
            action: Action dictionary with type and parameters.
            screen_width: Current screen width for coordinate normalization.
            screen_height: Current screen height for coordinate normalization.

        Returns:
            Action execution result.
        """
        action_type = action.get("_metadata", "")
        lang = action.get("lang", "cn")
        messages = get_messages(lang)

        if action_type == "takeover":
            message = action.get("message", messages["manual_operation"])
            if self.takeover_callback:
                self.takeover_callback(message)
            else:
                input(f"{message}\n" + messages["press_enter_when_done"])
            return {"success": True, "should_finish": False}

        elif action_type == "confirm":
            message = action.get("message", messages["operation_confirmation"])
            if self.confirmation_callback:
                confirmed = self.confirmation_callback(message)
            else:
                confirmed = input(f"{message} (y/N): ").lower() == "y"
            
            if not confirmed:
                return {"success": False, "should_finish": False}
        
        # Normalize coordinates if needed
        if "element" in action and isinstance(action["element"], list):
            x, y = action["element"]
            if x <= 1.0 and y <= 1.0:
                # Coordinates are normalized, convert to absolute
                action["element"] = [int(x * screen_width), int(y * screen_height)]
        
        if "start" in action and "end" in action:
            start_x, start_y = action["start"]
            end_x, end_y = action["end"]
            if (start_x <= 1.0 and start_y <= 1.0) and (end_x <= 1.0 and end_y <= 1.0):
                # Coordinates are normalized, convert to absolute
                action["start"] = [int(start_x * screen_width), int(start_y * screen_height)]
                action["end"] = [int(end_x * screen_width), int(end_y * screen_height)]

        # Execute action based on type
        action_name = action.get("action", "").lower()

        if self.device_type == "hdc":
            connection = HDCConnection(self.device_id)
        else:
            connection = ADBConnection(self.device_id)

        if action_name == "tap":
            x, y = action["element"]
            connection.tap(x, y)
            return {"success": True, "should_finish": False}

        elif action_name == "long press" or action_name == "long_press":
            x, y = action["element"]
            duration = action.get("duration", 3000)  # Default 3 seconds
            connection.long_press(x, y, duration)
            return {"success": True, "should_finish": False}

        elif action_name == "double tap" or action_name == "double_tap":
            x, y = action["element"]
            connection.double_tap(x, y)
            return {"success": True, "should_finish": False}

        elif action_name == "swipe":
            start_x, start_y = action["start"]
            end_x, end_y = action["end"]
            duration = action.get("duration")
            connection.swipe(start_x, start_y, end_x, end_y, duration)
            return {"success": True, "should_finish": False}

        elif action_name == "type" or action_name == "input_text":
            text = action["text"]
            connection.type_text(text)
            return {"success": True, "should_finish": False}

        elif action_name == "launch" or action_name == "open_app":
            app_name = action["app"]
            success = connection.launch_app(app_name)
            return {"success": success, "should_finish": not success}

        elif action_name == "back":
            connection.back()
            return {"success": True, "should_finish": False}

        elif action_name == "home":
            connection.home()
            return {"success": True, "should_finish": False}

        elif action_name == "clear_text":
            connection.clear_text()
            return {"success": True, "should_finish": False}

        elif action_name == "keyevent":
            key_code = action["key"]
            connection.keyevent(key_code)
            return {"success": True, "should_finish": False}

        elif action_name == "finish":
            return {"success": True, "should_finish": True, "message": action.get("message", "")}

        else:
            # Unknown action type
            return {"success": False, "should_finish": False, "error": f"Unknown action: {action_name}"}

    def keyevent(self, keycode: str) -> None:
        """
        Send a key event to the device.

        Args:
            keycode: Key code to send (e.g., "KEYCODE_POWER", "KEYCODE_VOLUME_UP").
        """
        if self.device_type == "hdc":
            connection = HDCConnection(self.device_id)
        else:
            connection = ADBConnection(self.device_id)

        connection.keyevent(keycode)

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        """Default confirmation callback using console input."""
        response = input(f"Sensitive operation: {message}\nConfirm? (Y/N): ")
        return response.upper() == "Y"

    @staticmethod
    def _default_takeover(message: str) -> None:
        """Default takeover callback using console input."""
        input(f"{message}\nPress Enter after completing manual operation...")


def parse_action(response: str) -> dict[str, Any]:
    """
    Parse action from model response.

    Args:
        response: Raw response string from the model.

    Returns:
        Parsed action dictionary.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    print(f"Parsing action: {response}")
    try:
        response = response.strip()
        
        # Remove XML tags and other artifacts that might be in the response
        response = response.replace("<answer>", "").replace("</answer>", "").strip()
        response = response.replace("<solution>", "").replace("</solution>", "").strip()
        response = response.replace("<result>", "").replace("</result>", "").strip()
        
        # Handle cases where there's trailing text after the action
        # Find the matching parentheses for the action function
        if response.startswith(("do(", "finish(")):
            paren_count = 0
            paren_start = -1
            
            for i, char in enumerate(response):
                if char == '(':
                    if paren_start == -1:
                        paren_start = i
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0 and paren_start != -1:
                        # Found the matching closing parenthesis
                        response = response[:i+1]
                        break
        
        # Clean up any remaining artifacts after extracting the action
        if '\n' in response:
            # Take only the first line that contains the action
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('do(') or line.startswith('finish(')):
                    response = line
                    break

        if response.startswith('do(action="Type"') or response.startswith(
            'do(action="Type_Name"'
        ):
            text = response.split("text=", 1)[1][1:-2]
            action = {"_metadata": "do", "action": "Type", "text": text}
            return action
        elif response.startswith("do"):
            # Use AST parsing instead of eval for safety
            try:
                # Escape special characters (newlines, tabs, etc.) for valid Python syntax
                response = response.replace('\n', '\\n')
                response = response.replace('\r', '\\r')
                response = response.replace('\t', '\\t')

                tree = ast.parse(response, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Expected a function call")

                call = tree.body
                # Extract keyword arguments safely
                action = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value

                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}")

        elif response.startswith("finish"):
            # Extract message from finish action
            try:
                # Find the message part: finish(message="...")
                start_idx = response.find('message="')
                if start_idx != -1:
                    start_idx += len('message="')
                    end_idx = response.rfind('"')
                    if end_idx > start_idx:
                        message = response[start_idx:end_idx]
                        return {"_metadata": "finish", "message": message}
                
                # If we can't extract the message properly, return the cleaned response
                message = response.replace("finish(message=", "").rstrip(')').strip('"')
                return {"_metadata": "finish", "message": message}
            except:
                # Fallback: return the whole response as message
                message = response.replace("finish(", "").replace(")", "")
                return {"_metadata": "finish", "message": message}
        else:
            raise ValueError(f"Failed to parse action: {response}")
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e}")


def do(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'do' actions."""
    kwargs["_metadata"] = "do"
    return kwargs


def finish(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'finish' actions."""
    kwargs["_metadata"] = "finish"
    return kwargs