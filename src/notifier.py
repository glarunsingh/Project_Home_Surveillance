import time
import logging
import cv2
import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    """Send alert photos to Telegram with per-type cooldown to prevent spam."""

    def __init__(self, bot_token, chat_id, cooldown_seconds=30):
        """
        Args:
            bot_token: Telegram bot token from @BotFather.
            chat_id: Telegram chat/group ID to send alerts to.
            cooldown_seconds: Minimum seconds between consecutive alerts of the same type.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown_seconds
        self._last_alert_times = {}  # per-type cooldown: {"unknown_face": timestamp, ...}
        self._base_url = TELEGRAM_API.format(token=bot_token)

    def _is_on_cooldown(self, alert_type="default"):
        """Check if a specific alert type is still within the cooldown period."""
        last_time = self._last_alert_times.get(alert_type, 0)
        elapsed = time.time() - last_time
        if elapsed < self.cooldown:
            remaining = self.cooldown - elapsed
            logger.debug("Alert '%s' on cooldown (%.1fs remaining)", alert_type, remaining)
            return True
        return False

    def send_photo(self, frame, caption="🚨 Security Alert", alert_type="default"):
        """
        Encode a frame as JPEG and send it to Telegram.

        Args:
            frame: OpenCV BGR frame (numpy array).
            caption: Alert message text (max 1024 chars).
            alert_type: Category for per-type cooldown (e.g. "unknown_face", "zone_intrusion").

        Returns:
            True if sent successfully, False otherwise.
        """
        if self._is_on_cooldown(alert_type):
            return False

        # Encode frame to JPEG in memory
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to encode frame as JPEG.")
            return False

        url = f"{self._base_url}/sendPhoto"
        files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
        data = {"chat_id": self.chat_id, "caption": caption[:1024]}

        try:
            resp = requests.post(url, files=files, data=data, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                self._last_alert_times[alert_type] = time.time()
                logger.info("Alert sent to Telegram: %s", caption)
                return True
            else:
                logger.error("Telegram API error %d: %s", resp.status_code, resp.text)
                return False
        except requests.RequestException as e:
            logger.error("Failed to send Telegram alert: %s", e)
            return False

    def send_text(self, message):
        """
        Send a plain text message to Telegram.

        Args:
            message: Text to send (max 4096 chars).

        Returns:
            True if sent successfully, False otherwise.
        """
        url = f"{self._base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message[:4096]}

        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info("Text message sent to Telegram.")
                return True
            else:
                logger.error("Telegram API error %d: %s", resp.status_code, resp.text)
                return False
        except requests.RequestException as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def test_connection(self):
        """
        Verify bot token is valid by calling getMe.

        Returns:
            Bot username string if valid, None if invalid.
        """
        url = f"{self._base_url}/getMe"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                bot_name = resp.json()["result"]["username"]
                logger.info("Telegram bot verified: @%s", bot_name)
                return bot_name
            else:
                logger.error("Invalid bot token. Telegram responded: %s", resp.text)
                return None
        except requests.RequestException as e:
            logger.error("Could not reach Telegram API: %s", e)
            return None


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    import numpy as np

    sys.path.insert(0, ".")
    from src.config import get_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = get_config()
    notifier = TelegramNotifier(
        bot_token=config["telegram_bot_token"],
        chat_id=config["telegram_chat_id"],
        cooldown_seconds=config["alert_cooldown"],
    )

    # Step 1: Verify bot token
    print("Testing Telegram bot connection...")
    bot_name = notifier.test_connection()
    if not bot_name:
        print("ERROR: Bot token is invalid. Check your .env file.")
        sys.exit(1)
    print(f"Bot verified: @{bot_name}")

    # Step 2: Send a test text message
    print("\nSending test text message...")
    if notifier.send_text("🔧 Smart Surveillance System — test message"):
        print("Text message sent! Check your Telegram.")
    else:
        print("Failed to send text message.")
        sys.exit(1)

    # Step 3: Send a test photo (synthetic image)
    print("\nSending test photo...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "TEST ALERT", (150, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    if notifier.send_photo(test_frame, caption="🔧 Test alert — surveillance system"):
        print("Photo sent! Check your Telegram.")
    else:
        print("Failed to send photo.")
        sys.exit(1)

    # Step 4: Verify cooldown works
    print(f"\nTesting cooldown ({config['alert_cooldown']}s)...")
    result = notifier.send_photo(test_frame, caption="Should be blocked by cooldown")
    if not result:
        print("Cooldown working — duplicate alert blocked.")
    else:
        print("WARNING: Cooldown did not work.")

    print("\nAll notifier tests passed!")
