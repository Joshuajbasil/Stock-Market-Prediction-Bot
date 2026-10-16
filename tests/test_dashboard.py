import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from stockbot import dashboard


class DashboardTests(unittest.TestCase):
    def test_page_bundles_assets_and_saved_results(self):
        page = dashboard.render_page()
        self.assertIn('<style>:root', page)
        self.assertNotIn('src="./app.js"', page)
        self.assertNotIn('href="./style.css"', page)
        self.assertIn('id="saved-data"', page)
        self.assertIn('const embedded = JSON.parse', page)
        self.assertIn('location.protocol ===', page)

    def config(self, **updates):
        return {'source': 'demo', 'backend': 'ridge', 'tickers': 'AAPL, MSFT',
                'start': '2015-01-01', 'end': '2025-10-01', 'epochs': 1, **updates}

    def test_rejects_invalid_run_paths(self):
        for name in ('../secret', 'a/b', 'C:\\private', ''):
            with self.assertRaises(ValueError):
                dashboard.run_path(name)

    def test_validates_run_inputs(self):
        for updates in ({'tickers': 'AAPL;whoami'}, {'epochs': 101}, {'source': 'shell'},
                        {'start': '2026-01-01'}, {'tickers': ''}):
            with self.assertRaises(ValueError):
                dashboard.command(self.config(**updates), 'test')

    def test_background_runner_produces_dashboard_data(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dashboard, 'OUTPUTS', Path(temp)):
            with patch.dict(dashboard.JOB, {'status': 'running', 'log': '', 'run': 'test'}, clear=True):
                dashboard.execute(dashboard.command(self.config(), 'test'))
                self.assertEqual(dashboard.JOB['status'], 'complete', dashboard.JOB['log'])
                result = dashboard.detail('test')
                self.assertTrue(result['report']['synthetic'])
                self.assertEqual(set(result['tickers']), {'AAPL', 'MSFT'})
                self.assertEqual(len(result['tickers']['AAPL']['recent']), 12)
                self.assertEqual(dashboard.runs()[0]['id'], 'test')
                self.assertGreater(len(result['tickers']['AAPL']['equity']), 0)


if __name__ == '__main__':
    unittest.main()
