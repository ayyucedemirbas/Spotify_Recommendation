from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import requests
    import librosa
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import plotly.express as px
    import yt_dlp
except ImportError as _err:
    sys.exit(
        f"Missing dependency: {_err}"
    )

CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID",     "your_cli_id")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "your_cli_sec")
REDIRECT_URI = "http://127.0.0.1:9999/callback" #change it if necessary, to empty the port, use: lsof -ti :9999 | xargs kill -9
SCOPE = "user-library-read"
SCOPE         = "user-library-read"

WORK_DIR    = Path("spotify_explorer")
PREVIEW_DIR = WORK_DIR / "previews"
FEAT_DIR    = WORK_DIR / "features"
OUT_DIR     = WORK_DIR / "output"

MAX_TRACKS    = 200
AUDIO_SR      = 22_050
CLIP_SECS     = 30.0
N_MFCC        = 13
K_CLUSTERS    = 6
N_RECS        = 20

def make_client() -> spotipy.Spotify:
    if "YOUR_CLIENT_ID" in (CLIENT_ID, CLIENT_SECRET):
        sys.exit(
            "Spotify credentials not set."
        )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=str(WORK_DIR / ".auth_cache"),
        open_browser=True,
    ))
    return sp

def fetch_liked_songs(sp: spotipy.Spotify) -> list[dict]:
    tracks: list[dict] = []
    page = sp.current_user_saved_tracks(limit=50)

    while page:
        for item in page["items"]:
            t = item.get("track")
            if not t or not t.get("id"):
                continue
            tracks.append({
                "id":          t["id"],
                "name":        t["name"],
                "artist":      ", ".join(a["name"] for a in t["artists"]),
                "artist_ids":  [a["id"] for a in t["artists"]],
                "album":       t["album"]["name"],
                "popularity":  t.get("popularity", 0),
                "uri":         t["uri"],
                "preview_url": t.get("preview_url"),
            })
            if len(tracks) >= MAX_TRACKS:
                return tracks

        page = sp.next(page) if page.get("next") else None

    return tracks

def _download_via_ytdlp(track: dict, dest: Path) -> Path | None:
    mp3_path = dest / f"{track['id']}.mp3"
    if mp3_path.exists():
        return mp3_path

    query = f"ytsearch1:{track['artist']} {track['name']} audio"

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "download_ranges": lambda info, _: [{"start_time": 0, "end_time": 32}],
        "force_keyframes_at_cuts": True,
        "outtmpl": str(dest / f"{track['id']}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        if mp3_path.exists():
            return mp3_path
        for f in sorted(dest.glob(f"{track['id']}.*")):
            return f
    except Exception as exc:
        print(f"      {track['name'][:40]}: {exc}")
    return None

def download_previews(tracks: list[dict]) -> dict[str, Path]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    n = len(tracks)

    for i, t in enumerate(tracks, 1):
        url = t.get("preview_url")
        if url:
            dest = PREVIEW_DIR / f"{t['id']}.mp3"
            if not dest.exists():
                try:
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                except Exception as exc:
                    print(f"     {t['name'][:40]}: {exc}")
                    dest = None
            if dest and dest.exists():
                paths[t["id"]] = dest

        else:
            p = _download_via_ytdlp(t, PREVIEW_DIR)
            if p:
                paths[t["id"]] = p

        if i % 10 == 0 or i == n:
            print(f"    {i}/{n}  ({len(paths)} with audio)")

        time.sleep(0.5) 

    return paths

def extract_features(path: Path) -> np.ndarray | None:
    try:
        y, sr = librosa.load(str(path), sr=AUDIO_SR, duration=CLIP_SECS, mono=True)
        if len(y) < sr * 3:
            return None

        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        rolloff  = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

        zcr      = float(librosa.feature.zero_crossing_rate(y).mean())
        rms      = float(librosa.feature.rms(y=y).mean())
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            chroma.mean(axis=1),
            chroma.std(axis=1),
            contrast.mean(axis=1),
            [centroid, rolloff, zcr, rms, float(tempo)],
        ]).astype(np.float32)

    except Exception as exc:
        print(f"   Feature error — {path.name}: {exc}")
        return None

def build_feature_matrix(
    tracks: list[dict],
    preview_paths: dict[str, Path],
) -> tuple[list[dict], np.ndarray]:
    cache_npz  = FEAT_DIR / "features.npz"
    cache_meta = FEAT_DIR / "meta.json"
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    target_ids = {t["id"] for t in tracks if t["id"] in preview_paths}

    if cache_npz.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if set(meta["ids"]) == target_ids:
            X = np.load(str(cache_npz))["X"]
            id2t  = {t["id"]: t for t in tracks}
            valid = [id2t[i] for i in meta["ids"] if i in id2t]
            print(f"   Loaded {len(valid)} cached feature vectors")
            return valid, X

    print(f"  Extracting features for {len(target_ids)} previews…")
    valid: list[dict] = []
    rows: list[np.ndarray] = []
    ids_out: list[str] = []
    done = 0

    for t in tracks:
        if t["id"] not in preview_paths:
            continue
        feat = extract_features(preview_paths[t["id"]])
        done += 1
        if done % 25 == 0:
            print(f"    {done}/{len(target_ids)} done")
        if feat is None:
            continue
        valid.append(t)
        rows.append(feat)
        ids_out.append(t["id"])

    print(f"    {done}/{len(target_ids)} done")

    if not rows:
        return [], np.empty((0, 0))

    X = np.stack(rows)
    np.savez_compressed(str(cache_npz), X=X)
    cache_meta.write_text(json.dumps({"ids": ids_out}))
    print(f"   Feature matrix: {X.shape[0]} tracks × {X.shape[1]} dims")
    return valid, X

def build_embedding(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = StandardScaler().fit_transform(X)

    n_pca = min(40, X_scaled.shape[1], X_scaled.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)

    n_nbr = min(15, X_pca.shape[0] - 1)
    X_2d  = UMAP(
        n_components=2,
        n_neighbors=n_nbr,
        min_dist=0.10,
        metric="cosine",
        random_state=42,
    ).fit_transform(X_pca)

    return X_pca, X_2d

def cluster_tracks(X_pca: np.ndarray, k: int) -> np.ndarray:
    k = max(2, min(k, X_pca.shape[0] // 2))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X_pca)

def recommend_songs(
    sp: spotipy.Spotify,
    tracks: list[dict],
    X_pca: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    liked_ids = {t["id"] for t in tracks}
    n_clusters = int(labels.max()) + 1

    seeds: list[dict] = []
    for cl in range(n_clusters):
        idx     = np.where(labels == cl)[0]
        centroid = X_pca[idx].mean(axis=0, keepdims=True)
        best    = idx[np.argmin(np.linalg.norm(X_pca[idx] - centroid, axis=1))]
        seeds.append(tracks[best])

    print("   seeds (one per cluster):")
    for s in seeds[:5]:
        print(f"       {s['name'][:40]:<40}  —  {s['artist']}")

    candidates: list[dict] = []
    seen_artist_names: set[str] = set()

    print(f"\n    Fetching recommendations via Search API fallback…")
    for seed in seeds:
        primary_artist = seed["artist"].split(",")[0].strip()
        if primary_artist in seen_artist_names:
            continue
        seen_artist_names.add(primary_artist)
        
        try:
            query = f"artist:{primary_artist}"
            results = sp.search(q=query, type="track", limit=15)
            
            for t in results["tracks"]["items"]:
                if t["id"] not in liked_ids:
                    candidates.append({
                        "id":         t["id"],
                        "name":       t["name"],
                        "artist":     ", ".join(a["name"] for a in t["artists"]),
                        "popularity": t.get("popularity", 0),
                        "uri":        t["uri"],
                        "source":     "search_fallback",
                    })
        except Exception as err:
            print(f"    Search error for {primary_artist}: {err}")

    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    unique.sort(key=lambda x: -x["popularity"])
    return unique[:N_RECS]

_DARK = "#0f0f0f"

def plot_static(
    tracks: list[dict],
    X_2d: np.ndarray,
    labels: np.ndarray,
    out: Path,
) -> None:
    palette = cm.get_cmap("tab10", int(labels.max()) + 1)
    colors  = [palette(int(l)) for l in labels]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor(_DARK)
    fig.patch.set_facecolor(_DARK)

    ax.scatter(X_2d[:, 0], X_2d[:, 1],
               c=colors, s=55, alpha=0.85, linewidths=0, zorder=3)

    rng    = np.random.default_rng(0)
    sample = rng.choice(len(tracks), min(40, len(tracks)), replace=False)
    for i in sample:
        ax.annotate(
            tracks[i]["name"][:22],
            (X_2d[i, 0], X_2d[i, 1]),
            color="white", fontsize=5.5, alpha=0.65,
            xytext=(4, 3), textcoords="offset points",
        )

    ax.scatter(*X_2d.mean(axis=0), marker="*", s=300,
               color="gold", zorder=5, label="Overall centroid")

    ax.set_title("Your Spotify Favorites — Audio Embedding (UMAP)",
                 color="white", fontsize=15, pad=10)
    ax.set_xlabel("UMAP-1", color="#888")
    ax.set_ylabel("UMAP-2", color="#888")
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)
    print(f"   Static plot     ->  {out}")

def plot_interactive(
    tracks: list[dict],
    X_2d: np.ndarray,
    labels: np.ndarray,
    out: Path,
) -> None:
    df = pd.DataFrame({
        "x":       X_2d[:, 0],
        "y":       X_2d[:, 1],
        "song":    [t["name"]                      for t in tracks],
        "artist":  [t["artist"]                    for t in tracks],
        "album":   [t["album"]                     for t in tracks],
        "pop":     [t["popularity"]                for t in tracks],
        "cluster": [f"Cluster {int(l) + 1}"       for l in labels],
        "uri":     [t["uri"]                       for t in tracks],
    })

    fig = px.scatter(
        df, x="x", y="y",
        color="cluster",
        hover_data={
            "song":    True,
            "artist":  True,
            "album":   True,
            "pop":     True,
            "cluster": False,
            "x":       False,
            "y":       False,
        },
        title="Spotify Audio Embedding (UMAP)",
        template="plotly_dark",
        opacity=0.87,
    )
    fig.update_traces(marker_size=9)
    fig.update_layout(title_font_size=16, legend_title_text="Favorites cluster")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f" Interactive plot  ->  {out}")

def main() -> None:
    sp = make_client()
    me = sp.me()
    print(f"  Logged in as  {me.get('display_name', 'unknown')}")

    print(f"Fetching up to {MAX_TRACKS} liked songs…")
    tracks = fetch_liked_songs(sp)
    if not tracks:
        sys.exit("  No liked songs found on your account.")
    print(f"    {len(tracks)} tracks fetched")

    print(f"Downloading audio previews…")
    preview_paths = download_previews(tracks)
    n_no_preview  = len(tracks) - len([t for t in tracks if t.get("preview_url")])
    if n_no_preview:
        print(f"    {n_no_preview} tracks had no preview URL (Spotify region / market)")
    if not preview_paths:
        sys.exit("  No audio previews available.  "
                 "Try a different set of liked songs or Spotify market.")
    print(f"    {len(preview_paths)} previews ready")

    print(f"Extracting audio features (MFCCs, chroma, spectral, tempo)…")
    valid_tracks, X = build_feature_matrix(tracks, preview_paths)
    if len(valid_tracks) < 10:
        sys.exit(
            f"  Only {len(valid_tracks)} usable tracks — need at least 10.\n"
            "    Make sure ffmpeg is installed for MP3 decoding."
        )

    print(f"Building the embedding")
    X_pca, X_2d = build_embedding(X)
    labels       = cluster_tracks(X_pca, K_CLUSTERS)
    n_clusters   = int(labels.max()) + 1
    print(f"    {len(valid_tracks)} tracks  ->  {X.shape[1]} dims  "
          f"  2D UMAP  |  {n_clusters}  clusters")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_emb = pd.DataFrame({
        "id":        [t["id"]         for t in valid_tracks],
        "name":      [t["name"]       for t in valid_tracks],
        "artist":    [t["artist"]     for t in valid_tracks],
        "album":     [t["album"]      for t in valid_tracks],
        "popularity":[t["popularity"] for t in valid_tracks],
        "uri":       [t["uri"]        for t in valid_tracks],
        "cluster":   (labels + 1).tolist(),
        "umap_x":    X_2d[:, 0].tolist(),
        "umap_y":    X_2d[:, 1].tolist(),
    })
    df_emb.to_csv(OUT_DIR / "embedding.csv", index=False)

    cluster_rows = []
    print("  Cluster summary:")
    for cl in range(n_clusters):
        idx      = np.where(labels == cl)[0]
        cl_arts  = [valid_tracks[i]["artist"].split(",")[0].strip() for i in idx]
        top3     = [a for a, _ in Counter(cl_arts).most_common(3)]
        avg_pop  = float(np.mean([valid_tracks[i]["popularity"] for i in idx]))
        cluster_rows.append({
            "cluster":       cl + 1,
            "n_tracks":      len(idx),
            "top_artists":   " / ".join(top3),
            "avg_popularity": round(avg_pop, 1),
        })
        print(f"    Cluster {cl+1}  ({len(idx):>3} tracks, pop≈{avg_pop:.0f})  "
              f"  {' / '.join(top3)}")

    pd.DataFrame(cluster_rows).to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    print(f"Finding {N_RECS} recommendations…")
    recs = recommend_songs(sp, valid_tracks, X_pca, labels)

    if recs:
        pd.DataFrame(recs).to_csv(OUT_DIR / "recommendations.csv", index=False)
        print(f"Recommended tracks ({len(recs)}):")
        for i, r in enumerate(recs, 1):
            print(f"    {i:>2}.  {r['name'][:38]:<38}  —  {r['artist']}")
    else:
        print("    No recommendations returned — try adjusting seed count.")

    print(f"Saving visualizations…")
    plot_static(valid_tracks, X_2d, labels, OUT_DIR / "plot_static.png")
    plot_interactive(valid_tracks, X_2d, labels, OUT_DIR / "plot_interactive.html")


if __name__ == "__main__":
    main()
