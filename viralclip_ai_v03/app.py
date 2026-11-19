import os, re, uuid, subprocess, wave, struct, base64, json, shutil
from pathlib import Path
from typing import List
import httpx
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("VIRALCLIP_DATA_DIR", str(ROOT / "data")))
UPLOADS, OUTPUTS, FRAMES = DATA/"uploads", DATA/"outputs", DATA/"frames"
for p in (UPLOADS, OUTPUTS, FRAMES): p.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
JOBS = {}
app = FastAPI(title="ViralClip AI", version="0.3.0")
app.mount("/static", StaticFiles(directory=ROOT/"static"), name="static")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode: raise RuntimeError((p.stderr or p.stdout)[-3000:])
    return p.stdout.strip()

def probe_duration(path):
    return float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)]))

def fmt(sec):
    sec=max(0,int(sec)); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def transcribe(path):
    from faster_whisper import WhisperModel
    model=WhisperModel(os.getenv("WHISPER_MODEL","base"),
        device=os.getenv("WHISPER_DEVICE","cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE","int8"))
    segments, info=model.transcribe(str(path),beam_size=3,vad_filter=True,word_timestamps=False)
    rows=[]
    for s in segments:
        t=(s.text or "").strip()
        if t: rows.append({"start":float(s.start),"end":float(s.end),"text":t})
    return rows, getattr(info,"language",None)

def audio_energy_peaks(wav_path, duration, bucket=2.0):
    """RMS energy by short window. Produces relative peaks, not emotion labels."""
    vals=[]
    try:
        with wave.open(str(wav_path),"rb") as w:
            channels=w.getnchannels(); width=w.getsampwidth(); rate=w.getframerate()
            frames_per=max(1,int(rate*bucket))
            pos=0
            while True:
                raw=w.readframes(frames_per)
                if not raw: break
                if width==2:
                    count=len(raw)//2
                    nums=struct.unpack("<"+"h"*count,raw[:count*2]) if count else []
                    rms=(sum(x*x for x in nums)/max(1,len(nums)))**0.5
                else:
                    rms=0
                vals.append({"time":pos*bucket,"rms":rms})
                pos+=1
    except Exception:
        return []
    if not vals: return []
    levels=sorted(x["rms"] for x in vals)
    threshold=levels[int(.85*(len(levels)-1))]
    peaks=[{"time":x["time"],"strength":round(x["rms"]/max(1,threshold),2)}
           for x in vals if x["rms"]>=threshold and x["rms"]>0]
    # Keep spaced peaks
    chosen=[]
    for p in sorted(peaks,key=lambda x:x["strength"],reverse=True):
        if all(abs(p["time"]-q["time"])>8 for q in chosen): chosen.append(p)
        if len(chosen)>=40: break
    return sorted(chosen,key=lambda x:x["time"])

def extract_frames(video, times, job_id):
    folder=FRAMES/job_id; folder.mkdir(parents=True,exist_ok=True)
    frames=[]
    for i,t in enumerate(times[:60]):
        out=folder/f"frame_{i:03d}.jpg"
        try:
            run(["ffmpeg","-y","-ss",str(max(0,t)),"-i",str(video),"-frames:v","1","-vf","scale=640:-1","-q:v","5",str(out)])
            if out.exists() and out.stat().st_size:
                frames.append({"time":t,"path":out})
        except Exception: pass
    return frames

def vision_rank(frames):
    """Optional OpenAI-compatible vision endpoint. Disabled unless configured."""
    endpoint=os.getenv("VISION_API_BASE_URL","").rstrip("/")
    key=os.getenv("VISION_API_KEY","")
    model=os.getenv("VISION_MODEL","")
    if not (endpoint and key and model and frames): return []
    all_results=[]
    # Keep batches small to control latency/cost.
    for offset in range(0,len(frames),8):
        batch=frames[offset:offset+8]
        content=[{"type":"text","text":
          "Review these sampled frames from a gaming livestream. Each image has a timestamp. "
          "Identify visually notable moments such as crashes, chases, near misses, surprising events, "
          "strong facial reactions, or clear visual punchlines. Return ONLY JSON: "
          '{"moments":[{"timestamp_seconds":123.4,"reason":"short explanation","visual_interest":1-5}]} '
          "Do not claim a moment is viral. Images follow with timestamp labels."}]
        for fr in batch:
            data=base64.b64encode(fr["path"].read_bytes()).decode()
            content.append({"type":"text","text":f"Timestamp: {fr['time']:.1f} seconds"})
            content.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{data}"}})
        try:
            resp=httpx.post(endpoint+"/chat/completions",
                headers={"Authorization":f"Bearer {key}"},
                json={"model":model,"temperature":0.1,"messages":[{"role":"user","content":content}]},
                timeout=120)
            resp.raise_for_status()
            txt=resp.json()["choices"][0]["message"]["content"]
            match=re.search(r"\{.*\}",txt,re.S)
            if match: all_results.extend(json.loads(match.group())["moments"])
        except Exception as e:
            print("Vision provider warning:",str(e))
    return all_results

def rank_highlights(transcript, peaks, visual, duration, min_len, max_len, limit):
    exciting=re.compile(r"\b(oh|wow|what|no way|bro|insane|crazy|wait|help|run|"
       r"laugh|lmao|haha|omg|let's go|are you serious|oh my god|police|chase|"
       r"crash|killed|win|lost|fail|scared|unbelievable|finally)\b",re.I)
    candidates=[]
    for s in transcript:
        text=s["text"]; center=(s["start"]+s["end"])/2
        score=1+min(3,len(exciting.findall(text)))+min(1.5,text.count("!")*.35)
        score+=.4 if len(text)>35 else 0
        candidates.append({"center":center,"score":score,"transcript":text,"signals":["speech context"]})
    for p in peaks:
        candidates.append({"center":p["time"],"score":2+min(3,p["strength"]-1),
                           "transcript":"","signals":["audio energy peak"]})
    for v in visual:
        candidates.append({"center":float(v.get("timestamp_seconds",0)),
                           "score":2+float(v.get("visual_interest",1))*.8,
                           "transcript":v.get("reason",""),"signals":["visual event"]})
    # Merge candidates close together so one incident becomes one clip.
    candidates.sort(key=lambda x:x["center"])
    merged=[]
    for c in candidates:
        if c["center"]<0 or c["center"]>duration: continue
        if merged and c["center"]-merged[-1]["center"]<12:
            old=merged[-1]
            old["score"]=max(old["score"],c["score"])+.25
            old["signals"]=list(set(old["signals"]+c["signals"]))
            if c["transcript"] and len(c["transcript"])>len(old["transcript"]): old["transcript"]=c["transcript"]
            old["center"]=(old["center"]+c["center"])/2
        else: merged.append(c.copy())
    selected=[]
    for c in sorted(merged,key=lambda x:x["score"],reverse=True):
        if all(abs(c["center"]-x["center"])>max(18,min_len*.75) for x in selected):
            selected.append(c)
        if len(selected)>=limit: break
    result=[]
    for i,c in enumerate(sorted(selected,key=lambda x:x["center"]),1):
        start=max(0,c["center"]-min_len*.4); end=min(duration,start+max_len)
        if end-start<min_len: start=max(0,end-min_len)
        result.append({"id":i,"start":start,"end":end,"start_time":fmt(start),
          "end_time":fmt(end),"duration":round(end-start),"score":round(c["score"],1),
          "score_label":"Strong candidate" if c["score"]>=4 else "Review candidate",
          "title":(c["transcript"] or "Potential visual/audio highlight")[:120],
          "signals":c["signals"],"reason":c["transcript"] or "Audio or visual activity detected"})
    return result

def render_clip(source,item,job_dir,index,kind):
    start=item["start"]; length=item["end"]-start; out=job_dir/f"clip_{index:02d}_{kind}.mp4"
    vf=("scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        if kind=="landscape" else "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")
    run(["ffmpeg","-y","-ss",str(start),"-i",str(source),"-t",str(length),"-vf",vf,
         "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",
         "-movflags","+faststart",str(out)])
    return f"/download/{job_dir.name}/{out.name}"

def process(job_id, source, min_len, max_len, limit, formats):
    jobdir=OUTPUTS/job_id; jobdir.mkdir(parents=True,exist_ok=True)
    try:
        JOBS[job_id]["status"]="extracting audio"
        duration=probe_duration(source)
        wav=jobdir/"audio.wav"
        run(["ffmpeg","-y","-i",str(source),"-vn","-ac","1","-ar","16000","-sample_fmt","s16",str(wav)])
        JOBS[job_id]["status"]="transcribing speech"
        transcript,language=transcribe(wav)
        JOBS[job_id]["status"]="analysing audio peaks"
        peaks=audio_energy_peaks(wav,duration)
        # Sample frames around audio peaks and at regular intervals for visual context.
        times=set([p["time"] for p in peaks])
        step=max(20,duration/45)
        times.update([x for x in range(0,int(duration),int(step))])
        frames=extract_frames(source,sorted(times),job_id)
        JOBS[job_id]["status"]="analysing sampled video frames"
        visual=vision_rank(frames)
        highlights=rank_highlights(transcript,peaks,visual,duration,min_len,max_len,limit)
        JOBS[job_id].update({"duration":duration,"language":language,"transcript_segments":len(transcript),
          "audio_peaks":len(peaks),"visual_observations":len(visual),"highlights":highlights,
          "status":"rendering clips","message":"Signals combined. Rendering selected clips."})
        for h in highlights:
            h["downloads"]={}
            for kind in formats: h["downloads"][kind]=render_clip(source,h,jobdir,h["id"],kind)
        JOBS[job_id]["status"]="complete"
        JOBS[job_id]["message"]="Analysis and clip rendering complete."
        wav.unlink(missing_ok=True)
    except Exception as e:
        JOBS[job_id]["status"]="error"; JOBS[job_id]["message"]=str(e)

@app.get("/")
def home(): return FileResponse(ROOT/"static"/"index.html")

@app.post("/api/analyze")
async def analyze(background_tasks:BackgroundTasks,file:UploadFile=File(...),
  clip_min:int=Form(20),clip_max:int=Form(60),max_clips:int=Form(15),formats:str=Form("both")):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED: raise HTTPException(400,"Upload MP4, MOV, MKV, WEBM or M4V.")
    job=uuid.uuid4().hex[:12]; dest=UPLOADS/f"{job}{ext}"; size=0
    with dest.open("wb") as f:
        while True:
            chunk=await file.read(1024*1024)
            if not chunk: break
            size+=len(chunk)
            if size>MAX_BYTES:
                f.close(); dest.unlink(missing_ok=True); raise HTTPException(413,"Upload size limit exceeded.")
            f.write(chunk)
    kinds=["vertical","landscape"] if formats=="both" else [formats]
    JOBS[job]={"job_id":job,"status":"queued","filename":file.filename,"highlights":[],"message":"Queued"}
    background_tasks.add_task(process,job,dest,max(10,clip_min),max(clip_min,clip_max),min(40,max(1,max_clips)),kinds)
    return JOBS[job]

@app.get("/api/jobs/{job}")
def status(job):
    if job not in JOBS: raise HTTPException(404,"Job not found")
    return JOBS[job]

@app.get("/download/{job}/{filename}")
def download(job,filename):
    folder=(OUTPUTS/job).resolve(); target=(folder/filename).resolve()
    if folder not in target.parents or not target.is_file(): raise HTTPException(404,"Clip not found")
    return FileResponse(target,media_type="video/mp4",filename=target.name)
