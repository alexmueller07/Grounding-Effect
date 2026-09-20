import json, os, sys, hashlib, urllib.request, time
D="/data/alexmueller/sprint/pope/data"
IMG=os.path.join(D,"val2014_images"); os.makedirs(IMG, exist_ok=True)
need=set()
for s in ["random","popular","adversarial"]:
    for l in open(f"{D}/coco_pope_{s}.json"):
        need.add(json.loads(l)["image"])
need=sorted(need)
print("need", len(need), flush=True)
ann=json.load(open("/data/datasets/coco/annotations_2014/instances_val2014.json"))
url={im["file_name"]: im["coco_url"] for im in ann["images"]}
dims={im["file_name"]: (im["width"], im["height"]) for im in ann["images"]}
missing_meta=[f for f in need if f not in url]
if missing_meta:
    print("FATAL no coco_url for", len(missing_meta), missing_meta[:5]); sys.exit(9)
ok=0; failed=[]
for i,f in enumerate(need):
    dst=os.path.join(IMG,f)
    if os.path.exists(dst) and os.path.getsize(dst)>1000: ok+=1; continue
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url[f], dst); ok+=1; break
        except Exception as e:
            if attempt==3: failed.append((f,repr(e)))
            else: time.sleep(2*(attempt+1))
    if (i+1)%100==0: print("  ...",i+1, flush=True)
print("downloaded/present", ok, "failed", len(failed))
if failed: print(failed[:10]); sys.exit(9)
# verify openability + dims against annotation
from PIL import Image
bad=[]
man={}
for f in need:
    p=os.path.join(IMG,f)
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w,h=im.size; mode=im.mode
        if (w,h)!=dims[f]: bad.append((f,"dim",(w,h),dims[f]))
        man[f]={"w":w,"h":h,"mode":mode,"bytes":os.path.getsize(p),
                "sha256":hashlib.sha256(open(p,"rb").read()).hexdigest()}
    except Exception as e:
        bad.append((f,"open",repr(e)))
if bad:
    print("FATAL bad images", len(bad), bad[:5]); sys.exit(9)
json.dump(man, open(f"{D}/image_manifest.json","w"), indent=0, sort_keys=True)
agg=hashlib.sha256("".join(man[f]["sha256"] for f in need).encode()).hexdigest()
print("ALL_IMAGES_OK", len(need), "aggregate_sha256", agg)
