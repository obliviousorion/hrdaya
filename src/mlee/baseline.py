"""Frozen AC/DCT + SVM baseline. No random beat split; no test-set tuning.

Small WFDB format-16 reader to avoid an uncached dependency. Explicitly rejects
other layouts. Validate against wfdb.rdrecord on real files before final reporting.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import scipy
from scipy.fft import dct
from scipy.signal import butter, sosfiltfilt, find_peaks
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def read_record(header):
    lines = header.read_text().splitlines()
    first = lines[0].split()
    channels, fs, samples = int(first[1]), float(first[2]), int(first[3])
    specs = [line.split() for line in lines[1:1 + channels]]
    if channels != 2 or fs != 500 or any(s[1] != '16' for s in specs):
        raise ValueError(f'Unsupported WFDB layout: {header}')
    if any(s[0] != specs[0][0] for s in specs):
        raise ValueError('Expected multiplexed format-16 channels')
    name = Path(specs[0][0])
    if name.name != str(name):
        raise ValueError('Expected adjacent data file')
    raw = np.fromfile(header.parent / name, dtype='<i2')
    if len(raw) != samples * channels:
        raise ValueError(f'Sample count mismatch: {header}')
    gain_spec = re.fullmatch(r'([\d.]+)(?:\((-?\d+)\))?(?:/.*)?', specs[0][2])
    if not gain_spec:
        raise ValueError('Unsupported ADC gain')
    gain = float(gain_spec[1])
    offset = float(gain_spec[2] or specs[0][4])
    x = (raw.reshape(samples, channels)[:, 0].astype(float) - offset) / gain
    return x, fs, [line for line in lines if line.startswith('#')], first[4:]

def segment(x, fs):
    """Raw channel -> 0.5–40 Hz zero-phase filter -> QRS energy -> 0.6s beats."""
    filtered = sosfiltfilt(butter(4, [0.5, 40], btype='bandpass', fs=fs, output='sos'), x)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = np.convolve(derivative**2, np.ones(int(.10*fs))/int(.10*fs), mode='same')
    candidates, _ = find_peaks(energy, distance=int(.35*fs),
                               prominence=max(np.percentile(energy, 90)*.15, 1e-12))
    peaks = []
    for p in candidates:
        lo, hi = max(0, p-int(.12*fs)), min(len(x), p+int(.12*fs)+1)
        r = lo + int(np.argmax(np.abs(filtered[lo:hi])))
        if (not peaks or r-peaks[-1] >= int(.30*fs)) and int(.20*fs) <= r < len(x)-int(.40*fs):
            peaks.append(r)
    beats = []
    for r in peaks:
        b = filtered[r-int(.20*fs):r+int(.40*fs)]
        if np.std(b) > 1e-8:
            beats.append((b - b.mean()) / b.std())
    return np.asarray(beats), peaks

def features(beats):
    rows = []
    for b in beats:
        ac = np.correlate(b, b, mode='full')[len(b)-1:]
        ac /= ac[0]
        rows.append(dct(ac, norm='ortho')[1:33])
    return np.asarray(rows)

def eer(labels, scores):
    far, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    frr = 1-tpr
    # Interpolate the crossing of the empirical ROC; descriptive test EER only.
    delta = far-frr
    j = int(np.flatnonzero(delta >= 0)[0])
    i = max(0, j-1)
    w = 0 if delta[j] == delta[i] else -delta[i]/(delta[j]-delta[i])
    return float(far[i]+w*(far[j]-far[i]))

def run(data, output):
    paths = sorted(data.glob('**/Person_*/*.hea'))
    if not paths:
        raise ValueError('No ECG-ID headers. Run scripts/download_data.py first.')
    by_person = {}
    for p in paths:
        by_person.setdefault(p.parent.name, []).append(p)
    train, test, ytrain, ytest, manifest, excluded = [], [], [], [], [], []
    for person, records in sorted(by_person.items()):
        records.sort(key=lambda p: int(p.stem.split('_')[-1]))
        accepted = []
        for p in records:
            x, fs, comments, time_fields = read_record(p)
            beats, peaks = segment(x, fs)
            if len(beats) < 3:
                excluded.append({'record':str(p.relative_to(data)), 'reason':'fewer than 3 usable beats'})
                continue
            accepted.append((p, features(beats), comments, time_fields))
        if len(accepted) < 2:
            excluded.append({'person':person,'reason':'fewer than 2 usable records'})
            continue
        for idx, (p, f, comments, time_fields) in enumerate(accepted):
            heldout = idx == len(accepted)-1
            (test if heldout else train).extend(f)
            (ytest if heldout else ytrain).extend([person]*len(f))
            manifest.append({'record':str(p.relative_to(data)), 'split':'test' if heldout else 'train',
                             'person':person,'beats':len(f),'comments':comments,'header_time_fields':time_fields,
                             'header_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                             'data_sha256':hashlib.sha256(p.with_suffix('.dat').read_bytes()).hexdigest()})
    if len(set(ytrain)) < 2:
        raise ValueError('Need at least two eligible identities')
    train, test = np.asarray(train), np.asarray(test)
    ytrain, ytest = np.asarray(ytrain), np.asarray(ytest)
    model = make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf', gamma='scale'))
    model.fit(train, ytrain)
    pred = model.predict(test)
    classes = model.classes_
    # Independent verification baseline: cosine similarity to enrollment templates.
    scaled_train, scaled_test = model[0].transform(train), model[0].transform(test)
    templates = np.stack([scaled_train[ytrain == c].mean(axis=0) for c in classes])
    normalize = lambda a: a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    scores = normalize(scaled_test) @ normalize(templates).T
    genuine = ytest[:, None] == classes[None, :]
    result = {'status':'measured', 'protocol':'last usable record per person held out; NOT proven cross-day',
              'dataset':'ECG-ID 1.0.0', 'model':'AC/DCT(32) + scaled RBF SVM, C=10',
              'verification_model':'cosine enrollment templates on scaled AC/DCT, not SVM probabilities',
              'identities':len(classes),'train_beats':len(train),'test_beats':len(test),
              'identification_accuracy':float(accuracy_score(ytest,pred)),
              'macro_identity_accuracy':float(np.mean([np.mean(pred[ytest==c]==c) for c in classes])),
              'verification_eer':eer(genuine.ravel(),scores.ravel()),
              'eer_note':'Descriptive empirical test EER; deployment threshold needs separate validation.',
              'genuine_trials':int(genuine.sum()),'impostor_trials':int((~genuine).sum()),
              'classes':classes.tolist(),'confusion_matrix':confusion_matrix(ytest,pred,labels=classes).tolist(),
              'split_manifest':manifest,'excluded':excluded,
              'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('classes','confusion_matrix','split_manifest','excluded')},indent=2))

