# Legacy Components (Archived)

This folder contains legacy components from the original ESP32 / Ollama prototype:
- `surdas_esp32_cam/`: Legacy ESP32-CAM firmware (replaced by Arduino UNO Q Linux V4L2/USB camera provider in `hal/camera.py`).
- `surdas_esp32_controller/`: Legacy ESP32 controller stub (replaced by Arduino UNO Q STM32 firmware in `sketch/uno_q_firmware/`).
- `test_esp32_connection.py`: Hardcoded HTTP check for 192.168.4.1 (replaced by `tests/test_camera_hal.py`).
- `test_imu_endpoint.py`: Hardcoded HTTP check for 192.168.4.2/imu (replaced by `tests/test_imu_hal.py`).

These files are preserved for historical reference only and are not part of the active Arduino UNO Q runtime.
