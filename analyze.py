import numpy as np, librosa, json, sys
y, sr = librosa.load("src/original.wav", sr=22050, mono=True)
dur = len(y)/sr
# tempo / beats
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", trim=False)
tempo = float(np.atleast_1d(tempo)[0])
print(f"duration {dur:.1f}s tempo {tempo:.1f} bpm, {len(beats)} beats")
# harmonic part for chroma
y_h, y_p = librosa.effects.hpss(y)
chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=512, bins_per_octave=36)
# key estimation (Krumhansl)
maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
mnr = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
prof = chroma.mean(axis=1)
names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
scores=[]
for i in range(12):
    scores.append((np.corrcoef(np.roll(maj,i),prof)[0,1], names[i]+"maj"))
    scores.append((np.corrcoef(np.roll(mnr,i),prof)[0,1], names[i]+"min"))
scores.sort(reverse=True)
print("key candidates:", scores[:4])
# chord templates
temps={}
for i in range(12):
    t=np.zeros(12); t[[i,(i+4)%12,(i+7)%12]]=1; temps[names[i]]=t
    t=np.zeros(12); t[[i,(i+3)%12,(i+7)%12]]=1; temps[names[i]+"m"]=t
tn=list(temps.keys()); T=np.stack([temps[k]/np.linalg.norm(temps[k]) for k in tn])
# beat-synchronous chroma
bframes = librosa.time_to_frames(beats, sr=sr, hop_length=512)
bc = librosa.util.sync(chroma, bframes, aggregate=np.median)
# per-beat chord
def best(v):
    v=v/(np.linalg.norm(v)+1e-9); s=T@v; return tn[int(np.argmax(s))], float(s.max())
beat_chords=[best(bc[:,i])[0] for i in range(bc.shape[1])]
# per-bar (4 beats) chord using summed chroma
bars=[]
for b in range(0, bc.shape[1]-1, 4):
    v=bc[:, b:b+4].sum(axis=1); c,s=best(v)
    t0 = beats[b] if b < len(beats) else dur
    bars.append({"bar":len(bars),"t":round(float(t0),2),"chord":c,"conf":round(s,2)})
print("bars:", len(bars))
line=[]
for i,b in enumerate(bars):
    line.append(f"{b['chord']:>3}")
    if (i+1)%8==0: print(f"{bars[i-7]['t']:6.1f}s: "+" ".join(line)); line=[]
if line: print(f"{bars[len(bars)-len(line)]['t']:6.1f}s: "+" ".join(line))
# structure via segmentation on chroma + mfcc
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
feat = np.vstack([librosa.util.normalize(librosa.util.sync(mfcc, bframes)), bc])
try:
    bounds = librosa.segment.agglomerative(feat, k=10)
    print("segment boundaries (s):", [round(float(beats[i]),1) for i in bounds if i < len(beats)])
except Exception as e: print("seg fail", e)
json.dump({"tempo":tempo,"beats":[float(b) for b in beats],"bars":bars,"key":scores[:3]}, open("out/analysis.json","w"), indent=1)
