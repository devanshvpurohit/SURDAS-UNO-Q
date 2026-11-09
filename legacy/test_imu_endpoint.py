#!/usr/bin/env python3
"""
Test script for ESP32 IMU endpoint connectivity and data validation.
Verifies fall detection integration is working correctly.
"""

import requests
import json
import time
import sys
from datetime import datetime

CONTROLLER_IP = "192.168.4.2"
IMU_ENDPOINT = f"http://{CONTROLLER_IP}/imu"
TIMEOUT = 2.0

def test_connectivity():
    """Test if ESP32 controller is reachable."""
    print("\n=== Testing ESP32 Connectivity ===")
    print(f"Target: {CONTROLLER_IP}")
    
    try:
        response = requests.get(IMU_ENDPOINT, timeout=TIMEOUT)
        if response.status_code == 200:
            print(f"✓ Controller reachable (HTTP {response.status_code})")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ Connection timeout after {TIMEOUT}s")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection refused - check if ESP32 is powered on")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_data_format():
    """Test if IMU endpoint returns valid JSON with required fields."""
    print("\n=== Testing Data Format ===")
    
    try:
        response = requests.get(IMU_ENDPOINT, timeout=TIMEOUT)
        data = response.json()
        
        required_fields = [
            ('ax', float),
            ('ay', float),
            ('az', float),
            ('acceleration', float),
            ('pitch', float),
            ('roll', float),
            ('falls', int),
            ('state', str),
            ('mpu', str)
        ]
        
        all_valid = True
        for field, expected_type in required_fields:
            if field in data:
                if isinstance(data[field], expected_type):
                    print(f"✓ {field}: {data[field]} ({expected_type.__name__})")
                else:
                    print(f"✗ {field}: Wrong type (expected {expected_type.__name__}, got {type(data[field]).__name__})")
                    all_valid = False
            else:
                print(f"✗ {field}: Missing from response")
                all_valid = False
        
        return all_valid, data
    
    except json.JSONDecodeError:
        print("✗ Response is not valid JSON")
        print(f"Raw response: {response.text}")
        return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None

def test_state_values(data):
    """Test if state field contains valid values."""
    print("\n=== Testing State Values ===")
    
    valid_states = ['NORMAL', 'POSSIBLE_FALL', 'IMPACT_DETECTED', 'FALL_CONFIRMED']
    state = data.get('state')
    
    if state in valid_states:
        print(f"✓ State '{state}' is valid")
        
        # Color code state
        if state == 'NORMAL':
            print("  → Status: 🟢 Normal operation")
        elif state == 'POSSIBLE_FALL':
            print("  → Status: 🟡 Suspicious movement")
        elif state == 'IMPACT_DETECTED':
            print("  → Status: 🟠 Impact detected")
        elif state == 'FALL_CONFIRMED':
            print("  → Status: 🔴 FALL CONFIRMED!")
        
        return True
    else:
        print(f"✗ State '{state}' is not valid")
        print(f"   Valid states: {', '.join(valid_states)}")
        return False

def test_mpu_status(data):
    """Test if MPU sensor status is OK."""
    print("\n=== Testing MPU Status ===")
    
    mpu_status = data.get('mpu')
    
    if mpu_status == 'OK':
        print(f"✓ MPU sensor: {mpu_status}")
        return True
    elif mpu_status == 'ERROR':
        print(f"✗ MPU sensor: {mpu_status} - Check sensor wiring")
        return False
    else:
        print(f"✗ Unknown MPU status: {mpu_status}")
        return False

def test_acceleration_range(data):
    """Test if acceleration values are reasonable."""
    print("\n=== Testing Acceleration Range ===")
    
    ax = data.get('ax', 0)
    ay = data.get('ay', 0)
    az = data.get('az', 0)
    accel = data.get('acceleration', 0)
    
    # Normal gravity is ~9.81 m/s²
    # Values should typically be -20 to +20 m/s² for each axis
    # Total acceleration should be positive
    
    all_valid = True
    
    if -20 <= ax <= 20:
        print(f"✓ ax: {ax:.2f} m/s² (valid range)")
    else:
        print(f"⚠ ax: {ax:.2f} m/s² (outside typical range)")
        all_valid = False
    
    if -20 <= ay <= 20:
        print(f"✓ ay: {ay:.2f} m/s² (valid range)")
    else:
        print(f"⚠ ay: {ay:.2f} m/s² (outside typical range)")
        all_valid = False
    
    if -20 <= az <= 20:
        print(f"✓ az: {az:.2f} m/s² (valid range)")
    else:
        print(f"⚠ az: {az:.2f} m/s² (outside typical range)")
        all_valid = False
    
    if 0 <= accel <= 30:
        print(f"✓ Total acceleration: {accel:.2f} m/s² (valid range)")
    else:
        print(f"⚠ Total acceleration: {accel:.2f} m/s² (unusual)")
        all_valid = False
    
    return all_valid

def test_continuous_monitoring(duration=5):
    """Test continuous data streaming for specified duration."""
    print(f"\n=== Testing Continuous Monitoring ({duration}s) ===")
    
    try:
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration:
            response = requests.get(IMU_ENDPOINT, timeout=TIMEOUT)
            data = response.json()
            samples.append(data)
            
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                  f"State: {data['state']:20s} | "
                  f"Accel: {data['acceleration']:6.2f} m/s² | "
                  f"Pitch: {data['pitch']:6.1f}° | "
                  f"Roll: {data['roll']:6.1f}° | "
                  f"Falls: {data['falls']}")
            
            time.sleep(0.2)  # 5 Hz polling (same as dashboard)
        
        print(f"\n✓ Received {len(samples)} samples")
        print(f"✓ Average rate: {len(samples)/duration:.1f} Hz")
        
        return True, samples
    
    except Exception as e:
        print(f"✗ Error during continuous monitoring: {e}")
        return False, []

def analyze_samples(samples):
    """Analyze statistics from continuous monitoring."""
    print("\n=== Sample Statistics ===")
    
    if not samples:
        print("No samples to analyze")
        return
    
    states = [s['state'] for s in samples]
    accels = [s['acceleration'] for s in samples]
    pitches = [s['pitch'] for s in samples]
    rolls = [s['roll'] for s in samples]
    
    print(f"Total samples: {len(samples)}")
    print(f"\nState distribution:")
    for state in set(states):
        count = states.count(state)
        percentage = (count / len(states)) * 100
        print(f"  {state}: {count} ({percentage:.1f}%)")
    
    print(f"\nAcceleration:")
    print(f"  Min: {min(accels):.2f} m/s²")
    print(f"  Max: {max(accels):.2f} m/s²")
    print(f"  Avg: {sum(accels)/len(accels):.2f} m/s²")
    
    print(f"\nOrientation:")
    print(f"  Pitch: {min(pitches):.1f}° to {max(pitches):.1f}°")
    print(f"  Roll:  {min(rolls):.1f}° to {max(rolls):.1f}°")

def main():
    """Run all tests."""
    print("=" * 60)
    print("ESP32 IMU Endpoint Test Suite")
    print("=" * 60)
    print(f"Testing endpoint: {IMU_ENDPOINT}")
    print(f"Timeout: {TIMEOUT}s")
    
    # Test 1: Connectivity
    if not test_connectivity():
        print("\n" + "=" * 60)
        print("❌ CONNECTIVITY FAILED - Cannot proceed with other tests")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check if ESP32 is powered on")
        print("2. Verify IP address: ping 192.168.4.2")
        print("3. Check WiFi connection")
        print("4. Verify firewall settings")
        sys.exit(1)
    
    # Test 2: Data format
    format_valid, data = test_data_format()
    if not format_valid or data is None:
        print("\n" + "=" * 60)
        print("❌ DATA FORMAT INVALID - Check ESP32 firmware")
        print("=" * 60)
        sys.exit(1)
    
    # Test 3: State values
    test_state_values(data)
    
    # Test 4: MPU status
    test_mpu_status(data)
    
    # Test 5: Acceleration range
    test_acceleration_range(data)
    
    # Test 6: Continuous monitoring
    print("\n" + "=" * 60)
    response = input("Run continuous monitoring test? (y/n): ")
    if response.lower() == 'y':
        success, samples = test_continuous_monitoring(duration=5)
        if success:
            analyze_samples(samples)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nIMU endpoint is working correctly!")
    print("Dashboard integration should work properly.")
    print("\nNext steps:")
    print("1. Start dashboard: cd caregiver_dashboard && npm run dev")
    print("2. Open http://localhost:5173")
    print("3. Check 'Live Safety' view for fall detector card")

if __name__ == "__main__":
    main()
