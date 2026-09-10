import numpy as np, librosa, json
A=json.load(open("out/analysis.json")); beats=np.array(A["beats"]); sr=22050
y,_=librosa.load("src/original.wav", sr=sr, mono=True)
y_h,_=librosa.effects.hpss(y)
chroma=librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=512, bins_per_octave=36)
bf=librosa.time_to_frames(beats, sr=sr, hop_length=512)
bc=librosa.util.sync(chroma, bf, aggregate=np.median)  # 12 x nbeats
names=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
# palette: A minor (harmonic) diatonic-ish chords
pal={"Am":[9,0,4],"Dm":[2,5,9],"E":[4,8,11],"F":[5,9,0],"C":[0,4,7],"G":[7,11,2],"Em":[4,7,11],"Bdim":[11,2,5],"Fm":[5,8,0],"A":[9,1,4],"D":[2,6,9]}
tn=list(pal); T=np.stack([np.eye(12)[v].sum(0) for v in pal.values()]); T=T/np.linalg.norm(T,axis=1,keepdims=True)
def best(v):
    v=v/(np.linalg.norm(v)+1e-9); s=T@v; i=int(np.argmax(s)); return tn[i], float(s[i])
nb=bc.shape[1]
per_beat=[best(bc[:,i])[0] for i in range(nb)]
# downbeat phase: count chord changes at each beat offset mod 4 -> downbeats should have most changes
changes=np.array([i for i in range(1,nb) if per_beat[i]!=per_beat[i-1]])
for ph in range(4): print("phase",ph,"changes on beat:",int(np.sum(changes%4==ph)))
# also onset strength per beat phase
oenv=librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
ob=librosa.util.sync(oenv[None,:], bf, aggregate=np.mean)[0]
for ph in range(4): print("phase",ph,"mean onset strength:",round(float(ob[ph::4].mean()),2))
json.dump({"per_beat":per_beat}, open("out/perbeat.json","w"))
