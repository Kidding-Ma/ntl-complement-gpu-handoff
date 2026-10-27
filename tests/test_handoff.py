"""CPU contract tests: no model execution, no network, no real competition data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_submission import LABELS, validate, check_components
from preflight import inspect_inputs
from fetch_assets import command


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.test = self.work / 'test.csv'
        self.sub = self.work / 'submission.parquet'
        pd.DataFrame({'id': ['001', 'a'], 'text': ['hello', 'world']}).to_csv(self.test, index=False)
        pd.DataFrame({'id': ['001', 'a'], 'model': [LABELS[0], LABELS[1]]}).to_parquet(self.sub, index=False)

    def test_valid_submission(self):
        self.assertEqual(validate(self.test, self.sub)['rows'], 2)

    def test_reject_bad_id_and_label(self):
        for ids, labs in [(['001', '001'], LABELS[:2]), (['001', 'b'], LABELS[:2]), (['001', 'a'], ['fake', LABELS[0]])]:
            pd.DataFrame({'id': ids, 'model': labs}).to_parquet(self.sub, index=False)
            with self.assertRaises(ValueError):
                validate(self.test, self.sub)

    def logs(self):
        complement = 'train (2699,4) test (2,2) gpus 2 smoke False\nCOMP STACK DONE 10s\n'
        inference = 'comp stack loaded (10s)\natlas /bundle mixed at 0.16 gpu0:0.0GB\ndeb done gpu0:0.0GB\n'
        for name, prefix, ids in [('q14v5', 'q14v5', range(5)), ('q8v5', 'q8v5', range(5)), ('ll8v5', 'll8v5', range(5)),
                                  ('ll8_ts', 'ts_llama8', (0, 2, 4)), ('q8s7', 'ts8_s7', (0, 2, 4))]:
            for adapter in [prefix + '_f' + str(i) for i in ids] + [prefix]:
                inference += f'l24 {name}: adapter {adapter} done (bf16 redo batches 0)\n'
        for name, adapter in [('n14b', 'lora_qwen3_14b'), ('q25_14', 'lora_qwen25_14b')]:
            inference += f'l24 {name}: adapter {adapter} done (bf16 redo batches 0)\n'
        np.save(self.work / 'comp_logit.npy', np.zeros((2, 18)))
        (self.work / 'comp_meta.json').write_text(json.dumps({'ids': ['001', 'a'], 'labels': LABELS}))
        return complement, inference

    def test_component_acceptance_and_allowed_historical_skip(self):
        c, i = self.logs()
        report = check_components(c + 'SKIP lora24_public: checkpoint not attached\n', i, self.work, self.test)
        self.assertEqual(report['required_adapter_count'], 28)

    def test_reject_partial_and_fake_runs(self):
        c, i = self.logs()
        for bad_c, bad_i in [(c, i + 'ATLAS SKIPPED: error'), (c, i.replace('adapter q8v5_f2 done', 'adapter wrong done')),
                              (c.replace('smoke False', 'smoke True'), i), (c + 'SKIP lora24_claude: missing', i)]:
            with self.assertRaises(ValueError):
                check_components(bad_c, bad_i, self.work, self.test)

    def test_reject_misaligned_complement(self):
        c, i = self.logs()
        (self.work / 'comp_meta.json').write_text(json.dumps({'ids': ['a', '001'], 'labels': LABELS}))
        with self.assertRaises(ValueError):
            check_components(c, i, self.work, self.test)

    def test_preflight_rejects_missing_inputs_and_shards(self):
        d = self.work / 'qwen-3/8b-base/transformers/1'
        d.mkdir(parents=True)
        (d / 'config.json').write_text(json.dumps({'model_type': 'qwen3'}))
        (d / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'missing.safetensors'}}))
        r = inspect_inputs(self.work)
        self.assertFalse(r['ok'])
        self.assertTrue(any('Missing shard' in e for e in r['errors']))

    def test_download_version_arguments(self):
        cmd = command({'kind': 'dataset', 'ref': 'owner/data', 'version': 3}, self.work)
        self.assertIn('owner/data/3', cmd)
        cmd = command({'kind': 'kernel', 'ref': 'owner/notebook', 'version': 1}, self.work)
        self.assertIn('owner/notebook/1', cmd)

    def test_generated_notebook_embeds_exact_modules(self):
        nb = json.loads((ROOT / 'notebooks/solution_strict.ipynb').read_text(encoding='utf-8'))
        ns = {'Path': Path}
        module_cell = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith("Path('gauss.py')"))
        import os
        before = Path.cwd()
        try:
            os.chdir(self.work)
            exec(compile(module_cell, 'module_cell', 'exec'), ns)
        finally:
            os.chdir(before)
        for name in ('gauss.py', 'infer.py', 'comp_stack.py'):
            self.assertEqual((self.work / name).read_bytes(), (ROOT / 'src' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
