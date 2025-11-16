# ESP32 LED Wiring Quick Reference

## 🔌 Pin Assignment

| Lane | LED Color | ESP32 GPIO | Physical Pin | Direction |
|------|-----------|------------|--------------|-----------|
| **A** | Red       | GPIO13     | D13          | Left Side |
| **A** | Green     | GPIO14     | D14          | Left Side |
| **B** | Red       | GPIO27     | D27          | Left Side |
| **B** | Green     | GPIO26     | D26          | Left Side |
| **C** | Red       | GPIO25     | D25          | Left Side |
| **C** | Green     | GPIO18     | D18          | Right Side |
| **D** | Red       | GPIO19     | D19          | Right Side |
| **D** | Green     | GPIO21     | D21          | Right Side |

## 📐 Connection Pattern

For each LED:

```
ESP32 GPIO Pin → 220Ω Resistor → LED (+) Long Leg
ESP32 GND → LED (-) Short Leg
```

**All LED negatives connect to the same GND pin/rail**

## ⚠️ Important Safety Notes

1. **Always use 220Ω resistors** - Without resistors, LEDs will burn out!
2. **Check LED polarity** - Long leg = Positive (+), Short leg = Negative (-)
3. **ESP32 uses 3.3V logic** - Do not use 5V directly on GPIO pins
4. **Common GND** - All LED negatives must connect to GND
5. **Maximum current** - Each GPIO can source ~12mA, LEDs typically need ~10-20mA

## 🧪 Testing Before Full Setup

Before connecting all 8 LEDs, test with one LED first:

1. Connect one LED: `ESP32 GPIO13 → 220Ω Resistor → LED (+) → LED (-) → GND`
2. Upload code to ESP32
3. Test via browser: `http://[ESP32_IP]/control?lane=1&state=GREEN`
4. LED should light up green
5. Test red: `http://[ESP32_IP]/control?lane=1&state=RED`
6. LED should switch to red

If this works, proceed with all 8 LEDs.

## 🔍 Visual Wiring Guide

```
ESP32 Dev Board
┌────────────────────────────────────┐
│  [D13] ← 220Ω ← LED A Red (+)     │
│  [D14] ← 220Ω ← LED A Green (+)   │
│  [D27] ← 220Ω ← LED B Red (+)     │
│  [D26] ← 220Ω ← LED B Green (+)   │
│  [D25] ← 220Ω ← LED C Red (+)     │
│  [D18] ← 220Ω ← LED C Green (+)   │
│  [D19] ← 220Ω ← LED D Red (+)     │
│  [D21] ← 220Ω ← LED D Green (+)   │
│                                    │
│  [GND] ← All LED (-) legs         │
└────────────────────────────────────┘
```

## ✅ Quick Verification

After wiring, test each lane:

```bash
# Test Lane 1 (A)
http://[ESP32_IP]/control?lane=1&state=GREEN  # Should light GREEN
http://[ESP32_IP]/control?lane=1&state=RED    # Should light RED

# Test Lane 2 (B)
http://[ESP32_IP]/control?lane=2&state=GREEN
http://[ESP32_IP]/control?lane=2&state=RED

# Test Lane 3 (C)
http://[ESP32_IP]/control?lane=3&state=GREEN
http://[ESP32_IP]/control?lane=3&state=RED

# Test Lane 4 (D)
http://[ESP32_IP]/control?lane=4&state=GREEN
http://[ESP32_IP]/control?lane=4&state=RED

# Test all at once
http://[ESP32_IP]/test  # Should cycle through all LEDs
```

## 🔧 Troubleshooting Wiring Issues

**LED not lighting up?**
- Check resistor is connected correctly
- Verify LED polarity (long leg = +)
- Ensure GND connection
- Test with multimeter: LED should have ~2V drop when on

**LED dim or flickering?**
- Check resistor value (should be 220Ω)
- Verify power supply (USB should provide enough power)
- Check wiring connections are secure

**Wrong LED lights up?**
- Verify GPIO pin matches wiring
- Check code pin assignments match physical wiring

**ESP32 not responding?**
- Check Serial Monitor for errors
- Verify WiFi connection
- Check IP address is correct

