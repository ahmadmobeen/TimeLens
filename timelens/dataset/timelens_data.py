# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import json
import os
import re


def parse_query(query):
    "Clean and normalize a text query by removing unnecessary whitespace and trailing periods"
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


ALLOW_MISSING_VIDEOS_ENV = "TIMELENS_ALLOW_MISSING_VIDEOS"


def check_videos_present(video_paths, dataset_name, video_root):
    """Fail loudly when annotated videos are not on disk. Returns the set of missing paths.

    Why this exists. Every loader in this file used to carry a COMMENTED-OUT existence check, so a
    loader pointed at annotations whose videos were absent yielded unloadable paths and reported
    nothing. That is exactly how NS-P2 issue #81 came to record "corpus is on disk" for
    TimeLens-100K when only its 19,466-record jsonl was present: the claim was never contradicted
    by anything the loader did. A run would then die at the first decord read -- or worse, survive
    on a partial epoch and produce a clean loss curve and a meaningless verdict.

    Behaviour. Missing files raise FileNotFoundError with the COUNT and the first ten paths, after
    the whole manifest is checked -- a count is diagnostic where a first-missing-path is not
    ("2 of 19,466 missing" and "19,466 of 19,466 missing" are different problems). Setting
    TIMELENS_ALLOW_MISSING_VIDEOS=1 downgrades that to dropping the missing records, and it still
    prints what it dropped. There is deliberately no mode in which missing videos pass unremarked.

    `video_paths` is checked once per unique video, not once per query, so the stat cost is one per
    file even on corpora with ~5 queries each.
    """
    unique = sorted(set(video_paths))
    missing = {p for p in unique if not os.path.exists(p)}
    if not missing:
        return missing

    shown = sorted(missing)[:10]
    detail = "\n  ".join(shown)
    if len(missing) > len(shown):
        detail += "\n  ... and %d more" % (len(missing) - len(shown))
    msg = (
        "%s: %d of %d annotated videos are missing under VIDEO_ROOT=%r.\n"
        "The annotations are present but the video corpus is not (or not fully) on disk, so these "
        "records would yield unloadable paths.\n  %s"
        % (dataset_name, len(missing), len(unique), video_root, detail)
    )
    if os.environ.get(ALLOW_MISSING_VIDEOS_ENV) == "1":
        print(
            "WARNING: %s\nWARNING: %s=1 -- dropping those %d videos from this split."
            % (msg, ALLOW_MISSING_VIDEOS_ENV, len(missing)),
            flush=True,
        )
        return missing
    raise FileNotFoundError(
        msg
        + "\nSet %s=1 to drop the missing records instead of failing; it prints what it drops."
        % ALLOW_MISSING_VIDEOS_ENV
    )


class ActivitynetTimeLensDataset:
    # VIDEO_ROOT is an ABSOLUTE path on the shared mount, following Ego4DNLQDataset and
    # TimeLens100KDataset. ActivityNet-TimeLens re-annotates the SAME ActivityNet videos the
    # omniembed split already uses, so there is nothing to download: the relative
    # data/TimeLens-Bench/videos/activitynet was never populated, and pointing at the existing
    # corpus resolves all of it. Verified by id: annotation keys are v_XXXXXXXXXXX and the files
    # are {id}.mp4 under this root.
    ANNO_PATH_TEST = "data/TimeLens-Bench/activitynet-timelens.json"
    VIDEO_ROOT = "/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos"
    DATASET_SOURCE = "ActivityNet-TimeLens"

    @classmethod
    def load_annos(cls, split="test"):
        assert split == "test", f"Invalid split: {split}"

        with open(cls.ANNO_PATH_TEST, "r") as f:
            raw_annos = json.load(f)

        # Checked before any anno is emitted. This was a commented-out per-item check, so a wrong
        # VIDEO_ROOT yielded zero clips in silence: shards launched, wrote 0 records, exited 0.
        missing = check_videos_present(
            [os.path.join(cls.VIDEO_ROOT, vid + ".mp4") for vid in raw_annos],
            cls.__name__,
            cls.VIDEO_ROOT,
        )

        annos = []
        for vid, raw_anno in raw_annos.items():
            video_path = os.path.join(cls.VIDEO_ROOT, vid + ".mp4")
            if video_path in missing:
                continue
            for span, query in zip(raw_anno["spans"], raw_anno["queries"]):
                anno = dict(
                    source=cls.DATASET_SOURCE,
                    data_type="grounding",
                    video_path=video_path,
                    duration=raw_anno["duration"],
                    query=parse_query(query),
                    span=[span],
                )

                annos.append(anno)

        return annos


class QVHighlightsTimeLensDataset(ActivitynetTimeLensDataset):
    ANNO_PATH_TEST = "data/TimeLens-Bench/qvhighlights-timelens.json"
    VIDEO_ROOT = "data/TimeLens-Bench/videos/qvhighlights"
    DATASET_SOURCE = "QVHighlights-TimeLens"


class CharadesTimeLensDataset(ActivitynetTimeLensDataset):
    ANNO_PATH_TEST = "data/TimeLens-Bench/charades-timelens.json"
    VIDEO_ROOT = "data/TimeLens-Bench/videos/charades"
    DATASET_SOURCE = "Charades-TimeLens"


class Ego4DNLQDataset(ActivitynetTimeLensDataset):
    """Ego4D v2 NLQ val, converted to bench format (issue #96).

    Built by experiments/nsp1-zoom/convert_ego4d_nlq.py: 415 clips, 4467 pairs, of which
    1145 are sub-2s and 386 sub-1s. This is the only split in the project with a sub-1s
    stratum worth the name -- native Charades-STA has no sub-2s moments at all and TACoS
    test has 7 sub-1s out of 4001.

    VIDEO_ROOT is an ABSOLUTE path because the clips are 15134 files on a shared mount and
    are deliberately not copied into data/. Everything else follows the parent exactly: the
    converter writes the same {vid: {duration, spans, queries}} schema.

    NOTE FOR CROP RUNS. Clips are ~480s, so at C.TOTAL_TOKENS the full-video pass samples
    384 frames = 0.8 effective fps, one frame per 1.25s. Moments shorter than that interval
    can fall between sampled frames entirely, which is precisely what the crop is expected
    to fix and why the effective fps has to be reported next to any recall number here.
    """

    ANNO_PATH_TEST = "data/TimeLens-Bench/ego4d-nlq-val.json"
    VIDEO_ROOT = "/gpfs/public/datasets/ego4d_data/v2/clips"
    DATASET_SOURCE = "Ego4D-NLQ"


class TimeLens100KDataset:
    """TimeLens-100K training corpus: 19,466 videos / 96,586 query-span pairs over five subsets
    (cosmo_cap, internvid_vtime, didemo, queryd, hirest), 574.9 h.

    VIDEO_ROOT is an ABSOLUTE path on the shared mount, following Ego4DNLQDataset and
    ActivityNetOmniEmbedDataset: the corpus is 136 GiB of mp4 and is deliberately not copied into
    data/. Fetched from the authors' HuggingFace release by
    experiments/nsp2-foveation/g2_fetch_timelens100k.sh, whose extraction root is this directory;
    the tars' internal paths are <subset>/<id>.mp4, matching the jsonl's video_path exactly, so no
    path rewriting happens anywhere.
    """

    ANNO_PATH_TRAIN = "data/TimeLens-100K/timelens-100k.jsonl"
    VIDEO_ROOT = "/gpfs/public/datasets/TimeLens-100K/videos"

    @classmethod
    def load_annos(cls, split="train"):
        assert split == "train", f"Invalid split: {split}"
        raw_annos = []
        with open(cls.ANNO_PATH_TRAIN, "r", encoding="utf-8") as f:
            for line in f:
                raw_annos.append(json.loads(line))

        # Checked before any anno is emitted, so a missing corpus cannot reach a training loop as
        # unloadable paths. See check_videos_present for why this is not a commented-out line.
        missing = check_videos_present(
            [os.path.join(cls.VIDEO_ROOT, raw["video_path"]) for raw in raw_annos],
            cls.__name__,
            cls.VIDEO_ROOT,
        )

        annos = []
        for raw_anno in raw_annos:
            video_path = os.path.join(cls.VIDEO_ROOT, raw_anno["video_path"])
            if video_path in missing:
                continue
            for event in raw_anno["events"]:
                query = parse_query(event["query"])
                span = event["span"]
                anno = dict(
                    source=raw_anno["source"],
                    data_type="grounding",
                    video_path=video_path,
                    duration=raw_anno["duration"],
                    query=query,
                    span=span,
                )
                annos.append(anno)

        return annos


class ActivityNetOmniEmbedDataset:
    """NS-P1 float-GT off-grid control (committee C2): omniembed ActivityNet-Captions val1 — GT is
    ~12.6% integer (genuinely sub-second), 13.5k videos on disk, duration 2-746s (real c1 variance).
    Deliberately NOT TimeLens-Bench's activitynet re-annotation (would re-impose a 1s grid and isn't on
    disk). Score the RAW answers via benchmarks.read_timelens_eval(from_answers=True) to avoid UNIT rounding."""
    ANNO_PATH_TEST = "/gpfs/public/datasets/omniembed/activitynet_captions/activitynet_captions_val1.json"
    VIDEO_ROOT = "/gpfs/public/datasets/omniembed/activitynet_captions/videos/Activity_Videos"
    DATASET_SOURCE = "ActivityNet-OmniEmbed"
    UNIT = 0.1  # we score from raw answers anyway; keep fine so stored preds aren't 1s-flattened

    @classmethod
    def load_annos(cls, split="test"):
        assert split == "test", f"Invalid split: {split}"
        with open(cls.ANNO_PATH_TEST, "r") as f:
            raw = json.load(f)  # LIST of records: {video, video_id, duration, timestamps:[[s,e]], sentences:[...]}
        annos = []
        n_dropped = 0  # on-disk coverage is 100% (verified); guard kept since videos use mixed containers (.mp4/.mkv/.webm)
        for rec in raw:
            vid = rec.get("video") or (str(rec.get("video_id", "")) + ".mp4")  # rec["video"] carries the real extension
            video_path = os.path.join(cls.VIDEO_ROOT, vid)
            if not os.path.exists(video_path):
                n_dropped += len(rec.get("timestamps", []))
                continue
            duration = rec["duration"]
            for span, sent in zip(rec.get("timestamps", []), rec.get("sentences", [])):
                annos.append(dict(
                    source=cls.DATASET_SOURCE, data_type="grounding",
                    video_path=video_path, duration=duration,
                    query=parse_query(sent), span=[span],
                ))
        print(f"[{cls.DATASET_SOURCE}] {len(annos)} scoreable moments; dropped {n_dropped} (missing video)")
        # NS-P1 first-read knob: keep <= N moments per GT-length bucket (deterministic, file order) so a
        # subset run has decisive S0/S1/S2 support without the full overnight pass. Unset => full set.
        cap = os.environ.get("TIMELENS_ANET_PER_BUCKET")
        if cap:
            cap = int(cap)
            def _bucket(span):
                L = float(span[1]) - float(span[0])
                for lab, lo, hi in (("S0", 0, 2), ("S1", 2, 5), ("S2", 5, 10), ("M", 10, 30), ("L", 30, 1e18)):
                    if lo <= L < hi:
                        return lab
                return None
            from collections import defaultdict
            kept, counts = [], defaultdict(int)
            for a in annos:
                b = _bucket(a["span"][0])
                if b and counts[b] < cap:
                    counts[b] += 1
                    kept.append(a)
            annos = kept
            print(f"[{cls.DATASET_SOURCE}] stratified subset <={cap}/bucket: {dict(counts)} -> {len(annos)} moments")
        return annos


DATASET_DICT = {
    "activitynet-timelens": ActivitynetTimeLensDataset,
    "activitynet-omniembed": ActivityNetOmniEmbedDataset,
    "qvhighlights-timelens": QVHighlightsTimeLensDataset,
    "charades-timelens": CharadesTimeLensDataset,
    "ego4d-nlq": Ego4DNLQDataset,
    "timelens-100k": TimeLens100KDataset,
}

if __name__ == "__main__":
    # Example usage
    DATASET_NAME = "timelens-100k"
    annos = DATASET_DICT[DATASET_NAME].load_annos()
    print(f"Loaded {len(annos)} annotations from {DATASET_NAME}")
