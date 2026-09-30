from pathlib import Path


def map_glosses_to_video_clips(avatar_payload: dict, video_dir="avatar/videos") -> dict:
    """
    Runtime adapter for the current prototype:
    BIM gloss sequence -> locally stored pre-generated avatar video clips.

    Viggle is used during content preparation, not required at runtime.
    """
    if not avatar_payload.get("avatar_ready", False):
        return {
            "success": False,
            "status": avatar_payload.get("status", "NOT_READY"),
            "clips": [],
            "missing_glosses": []
        }

    base = Path(video_dir)
    clips, missing = [], []

    for gloss in avatar_payload.get("bim_gloss_sequence", []):
        clip = base / f"{gloss}.mp4"
        if clip.exists():
            clips.append(str(clip))
        else:
            missing.append(gloss)

    return {
        "success": len(missing) == 0,
        "status": "READY_FOR_PLAYBACK" if not missing else "MISSING_VIDEO_CLIP",
        "clips": clips,
        "missing_glosses": missing
    }
