"""
SpectraForensic Automated Test Suite
====================================
Tests backend endpoints, detector algorithms, and sample image analysis.
"""

import unittest
import json
import os
import io
from PIL import Image
import numpy as np

from app import app
from detector import AIDetector


class TestSpectraForensic(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.detector = AIDetector()

    def test_homepage(self):
        """Verify homepage loads properly."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SpectraForensic', response.data)
        self.assertIn(b'AI-Generated Images', response.data)

    def test_sample_detection_endpoint(self):
        """Test GET /api/sample/<sample_name> endpoint."""
        for sample in ['sample_real', 'sample_ai', 'sample_midjourney']:
            response = self.app.get(f'/api/sample/{sample}')
            self.assertEqual(response.status_code, 200, f"Sample {sample} failed to load")
            data = json.loads(response.data.decode('utf-8'))
            self.assertIn('ai_probability', data)
            self.assertIn('real_probability', data)
            self.assertIn('verdict', data)
            self.assertIn('indicators', data)
            self.assertIn('heatmaps', data)
            print(f"Test sample '{sample}': AI Prob = {data['ai_probability']}%, Verdict = '{data['verdict']}'")

    def test_upload_detection_endpoint(self):
        """Test POST /api/detect file upload."""
        # Create an in-memory test image
        img = Image.new('RGB', (200, 200), color=(73, 109, 137))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = self.app.post(
            '/api/detect',
            data={'image': (img_bytes, 'test_synthetic.jpg')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertTrue(0.0 <= data['ai_probability'] <= 100.0)
        print(f"Test synthetic upload: AI Prob = {data['ai_probability']}%, Verdict = '{data['verdict']}'")

    def test_invalid_upload(self):
        """Test invalid / empty upload handling."""
        response = self.app.post('/api/detect', data={})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
