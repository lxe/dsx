import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('rank', Path(__file__).parents[1] / 'rank_from_routes.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Routes(unittest.TestCase):
    def parse(self, text, **kw):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'routes'
            p.write_text(text)
            return m.build_rank([p], layers=2, experts=4, picks=2, **kw)

    def test_counts_order_and_complete_coverage(self):
        rank, stats = self.parse('0 2 3\n1 0 2\n0 2 3\n')
        self.assertEqual(rank[:2], [[0, 2, 2], [0, 3, 2]])
        self.assertEqual(len(rank), 8)
        self.assertEqual(stats['total_picks'], 6)
        self.assertEqual(rank[2:4], [[1, 0, 1], [1, 2, 1]])

    def test_single_token_filter(self):
        rank, stats = self.parse('0 0 1 2 3\n0 2 3\n1 0 1\n', single_token_only=True)
        self.assertEqual(stats['skipped_multi_token_lines'], 1)
        self.assertEqual(stats['total_picks'], 4)

    def test_reject_invalid_input(self):
        for text in ['', '\n', '0 0 0\n', '2 0 1\n', '0 0 4\n', '0 0\n', '0  0 1\n', '0 0 [1, 2]\n', '0 0 1\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_dimensions(self):
        with self.assertRaises(ValueError):
            m.build_rank([], layers=0)

if __name__ == '__main__':
    unittest.main()
