import pickle, io, math, time, urllib.parse
import streamlit as st
import librosa, librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from collections import Counter
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter

st.set_page_config(page_title="WaveMind", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

# CSS FOR UI

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;600&display=swap');
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
header[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
#MainMenu{display:none!important}
footer{display:none!important}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#050510!important;color:#e0e0ff!important;font-family:'Inter',sans-serif!important}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#050510!important;color:#e0e0ff!important;font-family:'Inter',sans-serif!important}
.block-container{padding:0 2.5rem 2rem!important;max-width:100%!important}
h1,h2,h3{font-family:'Orbitron',monospace!important}
h1{color:#00E5FF!important;text-shadow:0 0 30px #00E5FF55}
[data-testid="stTabs"] button{color:#444466!important;font-weight:600!important;border-bottom:2px solid transparent!important;transition:all .3s!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00E5FF!important;border-bottom:2px solid #00E5FF!important}
div[data-testid="stMetric"]{background:#0a0a2e!important;border:1px solid #00E5FF22!important;border-radius:14px!important;padding:1rem 1.2rem!important;transition:all .3s!important}
div[data-testid="stMetric"]:hover{border-color:#00E5FF55!important;transform:translateY(-2px)}
div[data-testid="stMetric"] label{color:#00E5FF!important;font-size:.72rem!important;letter-spacing:2px!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#fff!important;font-size:1.5rem!important;font-weight:700!important;font-family:'Orbitron',monospace!important}
[data-testid="stFileUploader"]{background:#0a0a2e!important;border:1px dashed #BF00FF44!important;border-radius:16px!important;padding:.8rem!important}
.stButton>button{background:transparent!important;border:1px solid #00E5FF44!important;color:#00E5FF!important;border-radius:10px!important;font-weight:600!important;transition:all .3s!important}
.stButton>button:hover{border-color:#00E5FF!important;box-shadow:0 0 20px #00E5FF33!important}
[data-testid="stDownloadButton"]>button{background:transparent!important;border:1px solid #00E5FF44!important;color:#00E5FF!important;border-radius:10px!important}
[data-testid="stExpander"]{background:#0a0a2e!important;border:1px solid #ffffff0d!important;border-radius:14px!important;overflow:hidden!important}
audio{filter:invert(1) hue-rotate(180deg);width:100%}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#00E5FF,#BF00FF)!important;border-radius:99px!important}
[data-testid="stTextInput"] input{background:#0a0a2e!important;border:1px solid #00E5FF22!important;color:#e0e0ff!important;border-radius:10px!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:linear-gradient(#00E5FF44,#BF00FF44);border-radius:99px}

@keyframes float{
    0%,100%{transform:translateY(0)}
    50%{transform:translateY(-6px)}
}

@keyframes ringGlow{
    0%,100%{filter:drop-shadow(0 0 6px #00E5FF)}
    50%{filter:drop-shadow(0 0 18px #00E5FF)}
}

@keyframes noteFloat{
    0%{transform:translateY(0) rotate(0deg);opacity:.85}
    100%{transform:translateY(-160px) rotate(18deg);opacity:0}
}

@keyframes pulseGlow{
    0%,100%{opacity:.5;transform:scale(1)}
    50%{opacity:1;transform:scale(1.08)}
}

@keyframes shimmer{
    0%{background-position:-200% 0}
    100%{background-position:200% 0}
}

@keyframes drift{
    0%{transform:translateX(0) translateY(0)}
    33%{transform:translateX(12px) translateY(-10px)}
    66%{transform:translateX(-8px) translateY(6px)}
    100%{transform:translateX(0) translateY(0)}
}

.note{
    position:absolute;
    bottom:10px;
    animation:noteFloat 3.6s ease-in-out infinite;
    pointer-events:none;
}

.bg-orb{
    position:absolute;
    border-radius:50%;
    filter:blur(60px);
    opacity:.18;
    animation:drift 14s ease-in-out infinite;
    pointer-events:none;
}

.shimmer-line{
    background:linear-gradient(90deg,transparent,#00E5FF66,#BF00FF66,#00FF9955,transparent);
    background-size:200% 100%;
    animation:shimmer 4s linear infinite;
}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="position:relative;text-align:center;padding:2.6rem 0 1.2rem;overflow:hidden">

  <div class="bg-orb" style="width:220px;height:220px;background:#00E5FF;top:-60px;left:5%"></div>
  <div class="bg-orb" style="width:180px;height:180px;background:#BF00FF;top:-40px;right:8%;animation-delay:3s"></div>
  <div class="bg-orb" style="width:140px;height:140px;background:#00FF99;bottom:-50px;left:40%;animation-delay:6s"></div>

  <div class="note" style="left:8%;font-size:3.6rem;color:#00E5FFdd;animation-delay:0s">♪</div>
  <div class="note" style="left:20%;font-size:2.8rem;color:#BF00FFdd;animation-delay:.6s">♫</div>
  <div class="note" style="right:16%;font-size:3.2rem;color:#00FF99dd;animation-delay:1.2s">♩</div>
  <div class="note" style="right:28%;font-size:2.6rem;color:#FF00AAdd;animation-delay:1.8s">♬</div>
  <div class="note" style="left:35%;font-size:2.4rem;color:#00E5FFcc;animation-delay:2.4s">♪</div>
  <div class="note" style="right:42%;font-size:2.5rem;color:#BF00FFcc;animation-delay:3s">♫</div>

  <h1 style="font-size:3rem;letter-spacing:8px;margin-bottom:.1rem;background:linear-gradient(135deg,#00E5FF,#BF00FF,#00FF99);-webkit-background-clip:text;-webkit-text-fill-color:transparent">🌊 WAVEMIND</h1>

  <p style="color:#BF00FF;font-size:.82rem;letter-spacing:5px;margin:0">H E A R &nbsp; B E Y O N D &nbsp; N O I S E</p>

  <p style="color:#ffffff22;font-size:.7rem;letter-spacing:3px;margin-top:.4rem">EE200 · Audio Fingerprinting</p>

</div>

<div class="shimmer-line" style="height:2px;margin:0 0 1.5rem;box-shadow:0 0 14px #00E5FF55"></div>
""", unsafe_allow_html=True)

# Database

@st.cache_resource(show_spinner=False)
def load_db():
    with open("database.pkl","rb") as f:
        db=pickle.load(f)

    return db,sorted({s for v in db.values() for s,_ in v})

database, songs =load_db()
DB_SR = 48000

c1,c2,c3,c4=st.columns(4)

c1.metric("🎵 Songs",len(songs))
c2.metric("🔑 Hashes",f"{len(database):,}")
c3.metric("📡 SR",f"{DB_SR:,} Hz")
c4.metric("⚡ Status","Online")

st.markdown('<div style="height:1px;background:#ffffff08;margin:.5rem 0 1rem"></div>',unsafe_allow_html=True)

# Fingerprinting pipeline

def gen_hashes(y, sr, n_peaks=300, fan=10):
    """
    Step 1: Build a 'constellation map' of loud frequency peaks over time,
    then pair nearby peaks together to create hashes.

    Why pairs (not single peaks)?
    A single peak isn't unique enough -- many songs share the same note.
    A PAIR of peaks is far more unique, and is what makes matching fast and accurate.
    """

    # Spectrogram: how loud each frequency is at each point in time
    _,__,Sxx = spectrogram(y, fs=sr, nperseg=4096, noverlap=2048)
    S = 10*np.log10(Sxx+1e-10)   # convert to decibels (log scale, like human hearing)

    # Keep only local maxima that are clearly louder than average -> these
    # are the "peaks" that make up the constellation map
    fi,ti = np.where((S==maximum_filter(S,size=30))&(S>S.mean()+7))

    if not len(fi):
        return [],np.array([]),np.array([])   # silence / no usable peaks

    # If too many peaks were found, keep only the loudest n_peaks
    vals=S[fi,ti]
    if len(vals)>n_peaks:
        idx=np.argsort(vals)[-n_peaks:]
        fi,ti=fi[idx],ti[idx]

    # Sort peaks by time so we can pair each peak with the ones just after it
    o=np.argsort(ti)
    ti,fi=ti[o],fi[o]

    hashes=[]
    for i in range(len(ti)):
        for j in range(1,fan+1):
            if i+j<len(ti):
                dt=int(ti[i+j])-int(ti[i])
                if 2<=dt<=200:   # ignore pairs that are too close or too far apart
                    hashes.append(((int(fi[i]),int(fi[i+j]),dt),int(ti[i])))

    return hashes,fi,ti


def _match_segment(y, sr):
    """
    Hashing one short audio segment, then look each hash up in the
    pre-built database. Every database hit is a "vote" for (song, offset).

    offset = (time the hash appears in the original song) - (time it
    appears in our query clip). If the clip really is from that song,
    most votes will land on the SAME offset -> that's the alignment signal.
    """

    if sr!=DB_SR:
        y=librosa.resample(y,orig_sr=sr,target_sr=DB_SR)
        sr=DB_SR

    query_hashes,_,_=gen_hashes(y,sr)

    votes=Counter([(s,td-tq) for hq,tq in query_hashes if hq in database for s,td in database[hq]])

    return votes, len(query_hashes)


@st.cache_data(show_spinner=False, max_entries=30)
def identify(audio_bytes, sr):
    
    ## Why multiple windows? A single 10s window might land on a quiet or noisy part of the song. Sampling a few spots and combining votes
    ##makes the match much more robust.
    

    y=np.frombuffer(audio_bytes,dtype=np.float32)
    t0=time.time()

    dur=len(y)/DB_SR if sr==DB_SR else len(y)/sr
    seg=min(10*sr, len(y))   # 10-second analysis window

    # Pick up to 3 evenly-spaced starting points across the clip
    starts=[0]
    if dur>15:
        starts.append(len(y)//3)
    if dur>25:
        starts.append(2*len(y)//3)

    combined=Counter()
    total_query_hashes=0

    for s0 in starts:
        votes,nq=_match_segment(y[s0:s0+seg],sr)
        combined+=votes
        total_query_hashes+=nq

    runtime=round(time.time()-t0,3)

    if not combined:
        return "No Match",[],runtime,0,total_query_hashes,combined

    # The (song, offset) pair with the most votes wins.
    # A strong, consistent offset = the same part of the song lining up
    # across multiple hashes = a confident match.
    best_song, best_votes = combined.most_common(1)[0]

    return best_song[0], list(combined.keys()), runtime, best_votes, total_query_hashes, combined


def confidence_score(raw,nq,cnt):
    if not cnt or not raw:
        return 0
    top2=cnt.most_common(2)
    sec=top2[1][1] if len(top2)>1 else 0
    sel=min(1.0,(raw/max(sec,1)-1)/9) if sec else 1.0
    cov=min(1.0,raw/max(nq*.005,1))
    return round(min(99,max(45,45+(.7*sel+.3*cov)*54)),1)

def confidence_color(p):
    if p>=85:
        return "#178D32"
    if p>=65:
        return "#FFD700"
    return "#FF8C00"

@st.cache_data(show_spinner=False)
def album_art(name):
    try:
        r=requests.get(f"https://itunes.apple.com/search?term={urllib.parse.quote(name)}&media=music&limit=1",timeout=4)
        d=r.json()
        if d["resultCount"]>0:
            return d["results"][0].get("artworkUrl100","").replace("100x100bb","400x400bb")
    except:
        pass
    return None

@st.cache_data(show_spinner=False)
def hash_counts():
    hc={}
    for h in database:
        for s,_ in database[h]:
            hc[s]=hc.get(s,0)+1
    return hc

#  All PLOTS

BG,CYAN,PURPLE,GREEN,PINK="#070714","#00E5FF","#BF00FF","#00FF99","#FF00AA"

def _fig(w=10,h=4):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8F8FC1",labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a3a")
    ax.grid(True,color="#1a1a3a",lw=.5,ls='--',alpha=.5)
    return fig,ax

def plot_wave(y,sr):
    fig,ax=_fig(11,2.2)
    t=np.linspace(0,len(y)/sr,len(y))
    s=max(1,len(y)//5000)
    ax.fill_between(t[::s],y[::s],alpha=.25,color=CYAN)
    ax.plot(t[::s],y[::s],color=CYAN,lw=.6)
    ax.set_title("Waveform",color=CYAN,fontsize=11,pad=8)
    ax.set_xlim(0,t[-1])
    for sp in ax.spines.values():
        sp.set_edgecolor(CYAN)
    return fig

def plot_spec(y,sr):
    D=librosa.amplitude_to_db(np.abs(librosa.stft(y,n_fft=2048)),ref=np.max)
    fig,ax=_fig(11,4.5)
    img=librosa.display.specshow(D,sr=sr,x_axis='time',y_axis='hz',ax=ax,cmap='inferno')
    cb=fig.colorbar(img,ax=ax,format='%+2.0f dB')
    plt.setp(cb.ax.yaxis.get_ticklabels(),color="#aac3cc")
    ax.set_title("Spectrogram",color=CYAN,fontsize=11,pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(CYAN)
    return fig

def plot_constellation(fi,ti):
    fig,ax=_fig(11,4.5)
    if len(fi):
        cm=mcolors.LinearSegmentedColormap.from_list("c",[CYAN,PURPLE,PINK])
        ax.scatter(ti,fi,s=6,c=fi,cmap=cm,norm=plt.Normalize(fi.min(),fi.max()),alpha=.85,linewidths=0)
    ax.set_title("Constellation Map",color=PURPLE,fontsize=11,pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(PURPLE)
    return fig

def plot_offset_histogram(offsets,pred):
    fig,ax=_fig(11,4)
    ov=[o for s,o in offsets if s==pred]
    if ov:
        counts,_,patches=ax.hist(ov,bins=100,color=CYAN,alpha=.15,edgecolor='none')
        cm=mcolors.LinearSegmentedColormap.from_list("h",[PURPLE,CYAN,GREEN])
        for p,v in zip(patches,counts):
            p.set_facecolor(cm(v/(counts.max()+1e-9)))
            p.set_alpha(.8)
        ax.axvline(max(set(ov),key=ov.count),color=GREEN,lw=1.5,ls='--')
    ax.set_title("Offset Histogram",color=GREEN,fontsize=11,pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GREEN)
    return fig

def conf_ring(pct,color):
    r=60
    c=2*math.pi*r
    d=c*(pct/100)

    return f"""<div style="display:flex;align-items:center;gap:2.5rem;padding:1.4rem 0">
  <div style="position:relative;width:160px;height:160px;flex-shrink:0;animation:ringGlow 2s ease-in-out infinite">
    <svg width="160" height="160" viewBox="0 0 160 160">
      <circle cx="80" cy="80" r="{r}" fill="none" stroke="#ffffff0a" stroke-width="11"/>
      <circle cx="80" cy="80" r="{r}" fill="none" stroke="{color}" stroke-width="11" stroke-dasharray="{d:.1f} {c-d:.1f}" stroke-dashoffset="{c/4:.1f}" stroke-linecap="round" style="filter:drop-shadow(0 0 10px {color}) drop-shadow(0 0 22px {color}88)"/>
    </svg>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center">
      <div style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;color:#fff;text-shadow:0 0 20px {color}">{pct:.0f}%</div>
      <div style="font-size:.65rem;letter-spacing:3px;color:{color};font-weight:600;margin-top:2px">CONFIDENCE</div>
    </div>
  </div>
  <div>
    <div style="font-size:1rem;font-weight:600;color:#fff;margin-bottom:.5rem">Song identified successfully</div>
    <div style="font-size:.85rem;color:#c0c0d8;line-height:1.8">Hash alignment confirmed at a<br>consistent offset in the database.</div>
  </div>
</div>"""

def runner_up_card(rank, song, score, best_score):
    name=song.replace(".mp3","").replace(".wav","").strip()
    art=album_art(name)
    pct=round(score/max(best_score,1)*100)
    bar=f'<div style="height:3px;background:linear-gradient(90deg,#00E5FF,#BF00FF);width:{pct}%;border-radius:99px;margin-top:6px"></div>'
    img=f'<img src="{art}" style="width:48px;height:48px;border-radius:8px;object-fit:cover;flex-shrink:0">' if art else '<div style="width:48px;height:48px;background:#0d0d35;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0">🎵</div>'
    return f"""<div style="display:flex;align-items:center;gap:.9rem;background:#0a0a2e;border:1px solid #ffffff0d;border-radius:12px;padding:.75rem 1rem;margin-bottom:.5rem">
      <div style="font-family:'Orbitron',monospace;font-size:.8rem;color:#444466;font-weight:700;width:20px">#{rank}</div>
      {img}
      <div style="flex:1;min-width:0">
        <div style="font-size:.85rem;font-weight:600;color:#c0c0d8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
        <div style="font-size:.72rem;color:#555577;font-family:monospace">{score} hashes aligned{bar}</div>
      </div>
    </div>"""

# TABS

tab1,tab2,tab3=st.tabs(["  🎵  Library  ","  🔍  Identify  ","  📂  Batch  "])

# LIBRARY

with tab1:
    st.markdown("### 🎵 Song Library")

    q=st.text_input("Search songs...",placeholder="Type a song name")

    hc=hash_counts()

    df=pd.DataFrame({"Song":songs,"Hashes":[hc.get(s,0) for s in songs]})

    if q:
        df=df[df["Song"].str.contains(q,case=False)]

    c1,c2,c3=st.columns(3)

    c1.metric("Songs Indexed",len(songs))
    c2.metric("Unique Hashes",f"{len(database):,}")
    c3.metric("Avg Hashes/Song",f"{int(df['Hashes'].mean()):,}" if not df.empty else "0")

    st.markdown(f'<p style="color:#4a5070;font-size:.7rem;letter-spacing:3px;text-transform:uppercase;margin:1rem 0 .7rem">DATABASE — {len(df)} tracks</p>',unsafe_allow_html=True)

    GRID_COLS = 7

    for i,row in enumerate([df["Song"].tolist()[j:j+GRID_COLS] for j in range(0,len(df),GRID_COLS)]):
        cols=st.columns(GRID_COLS)

        for col,song in zip(cols,row):
            with col:
                name=song.replace(".mp3","").replace(".wav","").strip()
                art=album_art(name)

                if art:
                    st.image(art,use_container_width=True)
                else:
                    st.markdown('<div style="width:100%;aspect-ratio:1;background:#0a0a2e;border:1px solid #00E5FF15;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.5rem">🎵</div>',unsafe_allow_html=True)

                st.markdown(f'<div style="font-size:.68rem;font-weight:600;color:#e0e0ff;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{name}">{name}</div><div style="font-size:.58rem;color:#4a5070;font-family:monospace">{hc.get(song,0):,} hashes</div>',unsafe_allow_html=True)

#IDENTIFY 

with tab2:
    st.markdown("### 🔍 Identify a Song")

    uf=st.file_uploader("Drop an audio clip — MP3 or WAV",type=["mp3","wav"],key="single")

    if uf:
        st.audio(uf)

        with st.spinner("⚡ Fingerprinting across segments..."):
            y,sr=librosa.load(io.BytesIO(uf.read()),sr=None,mono=True)
            pred,offsets,rt,raw,nq,cnt=identify(y.tobytes(),sr)

        if pred=="No Match":
            st.error("❌ No match found in database.")

        else:
            name=pred.replace(".mp3","").replace(".wav","").strip()
            art=album_art(name)
            pct=confidence_score(raw,nq,cnt)
            color=confidence_color(pct)

            col_art,col_info=st.columns([1,2.2])

            with col_art:
                if art:
                    st.image(art,width=320)
                else:
                    st.markdown('<div style="width:320px;height:320px;background:#0a0a2e;border:1px solid #00E5FF33;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:5rem">🎵</div>',unsafe_allow_html=True)

            with col_info:
                st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a0a2e,#0d0820);border:1px solid {color}44;border-radius:18px;padding:1.5rem;box-shadow:0 0 30px {color}0f">
  <p style="color:{color};font-size:.72rem;letter-spacing:2px;font-family:monospace;margin:0;font-weight:600">✦ MATCH FOUND ✦</p>
  <p style="color:#fff;font-family:'Orbitron',monospace;font-size:2.1rem;font-weight:700;margin:.2rem 0 .8rem;letter-spacing:-.5px">{name}</p>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:.5rem">
    <span style="background:#ffffff06;border:1px solid {color}22;border-radius:8px;padding:.35rem .7rem;font-family:monospace">
      <span style="color:#b0b0cc;font-size:.7rem;font-weight:500">RUNTIME</span><br>
      <span style="color:#fff;font-weight:700;font-size:.95rem">{rt}s</span></span>
    <span style="background:#ffffff06;border:1px solid {color}22;border-radius:8px;padding:.35rem .7rem;font-family:monospace">
      <span style="color:#b0b0cc;font-size:.7rem;font-weight:500">ALIGNED</span><br>
      <span style="color:#fff;font-weight:700;font-size:.95rem">{raw} hashes</span></span>
    <span style="background:#ffffff06;border:1px solid {color}22;border-radius:8px;padding:.35rem .7rem;font-family:monospace">
      <span style="color:#b0b0cc;font-size:.7rem;font-weight:500">SEGMENTS</span><br>
      <span style="color:#fff;font-weight:700;font-size:.95rem">{min(3,max(1,int(len(y)/sr)//10))}</span></span>
  </div>
  {conf_ring(pct,color)}
</div>""", unsafe_allow_html=True)

            if raw<5:
                st.markdown(f"""<div style="background:#2a1a0a;border:1px solid #FF8C0044;border-radius:12px;padding:.8rem 1.1rem;margin-top:.8rem;display:flex;align-items:center;gap:.7rem">
  <span style="font-size:1.2rem">⚠️</span>
  <span style="font-size:.8rem;color:#e0c0a0;line-height:1.5">Only <b>{raw} hash{'es' if raw!=1 else ''}</b> aligned — this match is weak evidence. Try a longer or cleaner clip for a more reliable result.</span>
</div>""", unsafe_allow_html=True)

            # ── Top 3 runner-ups ──────────────────────────────────────────────

            by_song=Counter()
            for (s,_off),sc in cnt.items():
                by_song[s]+=sc

            runners=[(s,sc) for s,sc in by_song.most_common(4) if s!=pred][:3]

            if runners:
                st.markdown('<p style="color:#4a5070;font-size:.7rem;letter-spacing:3px;text-transform:uppercase;margin:1.2rem 0 .5rem">OTHER CANDIDATES</p>',unsafe_allow_html=True)

                for rank,(song,score) in enumerate(runners,2):
                    st.markdown(runner_up_card(rank,song,score,raw),unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:#ffffff08;margin:1rem 0"></div>',unsafe_allow_html=True)

        with st.spinner("Rendering analysis..."):
            _,fi,ti=gen_hashes(y[:min(len(y),10*sr)],sr)

        if len(fi):
            with st.expander("📈 Waveform",expanded=True):
                st.pyplot(plot_wave(y,sr),use_container_width=True)

            cl,cr=st.columns(2)

            with cl:
                with st.expander("🌈 Spectrogram",expanded=True):
                    st.pyplot(plot_spec(y[:min(len(y),30*sr)],sr),use_container_width=True)

            with cr:
                with st.expander("✦ Constellation Map",expanded=True):
                    st.pyplot(plot_constellation(fi,ti),use_container_width=True)

            if pred!="No Match":
                with st.expander(" Offset Histogram",expanded=True):
                    st.pyplot(plot_offset_histogram(offsets,pred),use_container_width=True)

        else:
            st.warning("Not enough peaks detected.")

#BATCH

with tab3:
    st.markdown("### 📂 Batch Identification")

    ufs=st.file_uploader("Upload multiple clips",type=["mp3","wav"],accept_multiple_files=True,key="batch")

    if ufs:
        bar=st.progress(0)
        status=st.empty()
        results=[]

        for i,f in enumerate(ufs):
            status.info(f"Processing **{f.name}** ({i+1}/{len(ufs)})")

            y,sr=librosa.load(io.BytesIO(f.read()),sr=None,mono=True)
            pr,_,rt,raw,nq,cnt=identify(y.tobytes(),sr)

            pct=confidence_score(raw,nq,cnt) if pr!="No Match" else 0

            results.append([f.name.rsplit(".",1)[0], pr.replace(".mp3","").replace(".wav",""), f"{pct:.0f}%", rt])

            bar.progress((i+1)/len(ufs))

        status.success("✅ Done!")

        df_r=pd.DataFrame(results,columns=["filename","prediction","confidence","runtime_s"])

        st.dataframe(df_r,use_container_width=True)

        st.download_button("⬇ Download results.csv",df_r[["filename","prediction"]].to_csv(index=False),"results.csv","text/csv",use_container_width=True)

st.markdown("""<div class="shimmer-line" style="height:2px;margin:2rem 0 1rem;box-shadow:0 0 14px #00E5FF44"></div>
<div style="text-align:center;padding-bottom:1rem"><p style="font-family:'Orbitron',monospace;color:#00E5FF33;letter-spacing:4px;font-size:.72rem">🌊 WAVEMIND &nbsp;·&nbsp; EE200</p></div>""",unsafe_allow_html=True)
