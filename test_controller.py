"""Basic checks for the software AVR controller."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

class ControllerTests(unittest.TestCase):
    def test_rejects_out_of_range_voltage(self):
        from avr_controller import IntelligentAVR
        obj = IntelligentAVR.__new__(IntelligentAVR)
        with self.assertRaises(ValueError):
            obj.regulate(300, 1, 25)

if __name__ == "__main__":
    unittest.main()
