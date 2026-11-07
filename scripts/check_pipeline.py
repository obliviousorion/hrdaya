"""Local integration check using generated signals, never a course result."""
import json
from pathlib import Path
import tempfile
import numpy as np
from mlee.baseline import read_record, segment, run

def main():
    with tempfile.TemporaryDirectory(prefix='mlee-check-') as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(42)
        for person in range(1, 4):
            folder = root / f'Person_{person:02}'
            folder.mkdir()
            for record in range(1, 4):
                t = np.arange(10000)/500
                x = rng.normal(0,.004,len(t))
                for peak in np.arange(.8,19.5,.85):
                    x += np.exp(-((t-peak)/(.012+person*.004))**2)
                    x -= .25*np.exp(-((t-peak+.04)/.014)**2)
                    x += (.1*person)*np.exp(-((t-peak-.22)/.06)**2)
                raw = np.round(x*2000).astype('<i2')
                np.column_stack((raw,raw)).astype('<i2').tofile(folder/f'rec_{record}.dat')
                (folder/f'rec_{record}.hea').write_text(
                    f'rec_{record} 2 500 10000\n'
                    f'rec_{record}.dat 16 2000(0)/mV 12 0 0 0 0 ECG I\n'
                    f'rec_{record}.dat 16 2000(0)/mV 12 0 0 0 0 ECG I filtered\n')
        header = root/'Person_01/rec_1.hea'
        x, fs, _, _ = read_record(header)
        beats, peaks = segment(x,fs)
        assert beats.shape[1] == 300 and 15 <= len(beats) <= 30
        assert np.all(np.diff(peaks) >= 150)
        result_file = root/'result.json'
        # Suppress measured output so generated-data numbers cannot be mistaken for ECG results.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            run(root,result_file)
        result = json.loads(result_file.read_text())
        train = {r['record'] for r in result['split_manifest'] if r['split']=='train'}
        test = {r['record'] for r in result['split_manifest'] if r['split']=='test'}
        assert not train & test and len(test)==3 and len(train)==6
        assert 0 <= result['verification_eer'] <= 1
        assert sum(map(sum,result['confusion_matrix'])) == result['test_beats']
        assert result['genuine_trials'] == result['test_beats']
        assert result['impostor_trials'] == result['test_beats']*2
        print('PASS: generated format-16 reader, filtering, beat windows, disjoint split, classifier, EER, JSON. No ECG-ID result.')

if __name__ == '__main__':
    main()
