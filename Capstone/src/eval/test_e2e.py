
# ---------------------------
# FILE: eval/test_e2e.py
# ---------------------------
"""
Simple evaluation tests - mocks the end-to-end flow.
"""
import unittest
from agent import VAccessOrchestrator
from config import CONFIG

class TestE2E(unittest.TestCase):
    def setUp(self):
        self.orch = VAccessOrchestrator(CONFIG)
        self.session = self.orch.start_session('test_user', 'Hello')

    def test_education(self):
        resp = self.orch.run_education(self.session, 'What is a vaccine?')
        self.assertIsInstance(resp, str)

    def test_schedule_flow(self):
        confirm = self.orch.find_and_schedule(self.session, '94110')
        self.assertIn('confirmed', confirm)

if __name__ == '__main__':
    unittest.main()

