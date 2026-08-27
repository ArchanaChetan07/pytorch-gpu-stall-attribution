# Injecting stalls honestly

Every class in the taxonomy has an injector, because a class that cannot be
injected cannot be validated. This page is about the one injector where the
portable version is not the honest version, and what to do about it.

## The I/O injector is the compromise

`DataloaderIO` performs the real read and then sleeps for a calibrated interval.
It reproduces the *timing* of a slower storage tier and none of its *behaviour*:
no queueing, no readahead interaction, no page-cache pressure, no change in the
distribution's tail shape.

That matters here specifically, because the sampler's I/O signals are exactly the
things the compromise distorts:

| signal | real slow device | sleep fallback |
|---|---|---|
| process state `D` | yes, workers park in uninterruptible sleep | **no** — `S`, interruptible |
| `read_bytes` climbing | yes | yes (the real read still happens) |
| `/proc/pressure/io` | yes | **no** |

So a sweep run entirely on the fallback will calibrate the I/O-versus-CPU
threshold against `read_bytes` alone and never exercise the state signal. Say so
if that is what you did.

## The honest instruments

Both need privileges most rented containers withhold. Check before you rent.

### `dm-delay` — inject latency

Deterministic per-device read latency, independent of write latency. The closest
thing to "the same storage, further away".

```bash
# 0 = start sector, delay in ms applied to reads only
SECTORS=$(blockdev --getsz /dev/nvme0n1)
echo "0 $SECTORS delay /dev/nvme0n1 0 8" | dmsetup create slowdisk
mkfs.ext4 /dev/mapper/slowdisk && mount /dev/mapper/slowdisk /mnt/slow
```

Sweep by recreating the target with a different delay:

```bash
dmsetup suspend slowdisk
echo "0 $SECTORS delay /dev/nvme0n1 0 20" | dmsetup reload slowdisk
dmsetup resume slowdisk
```

### cgroup v2 `io.max` — inject bandwidth pressure

Caps throughput rather than adding latency. Use it when the question is "what if
the tier were slower", not "what if it were further away".

```bash
MAJMIN=$(lsblk -ndo MAJ:MIN /dev/nvme0n1)
mkdir -p /sys/fs/cgroup/whystall
echo "$MAJMIN rbps=52428800" > /sys/fs/cgroup/whystall/io.max   # 50 MB/s
echo $$ > /sys/fs/cgroup/whystall/cgroup.procs
python examples/train_reference.py --injector control --cache-state cold
```

Put the *training process* in the cgroup, not just the workers: they inherit it,
and adding them individually races with respawn.

## Page cache: the thing that quietly invalidates the whole run

An I/O measurement on a box where the dataset is already in RAM measures nothing,
and nothing will warn you. State cold or warm for every run — `HarnessConfig`
takes `cache_state` and writes it into the manifest.

To actually get cold, in descending order of reliability:

```bash
sync && echo 3 > /proc/sys/vm/drop_caches      # needs privilege
```

If you cannot:

- size the corpus past **twice** system RAM, so it cannot be resident; or
- open with `O_DIRECT`, bypassing the cache entirely; or
- run cold-first, once, and discard every later repetition.

Say which. `cache_state="unknown"` in a manifest is more useful than a
confident wrong value.

## Validating the fallback against the real thing

If you get privileged access even once, run the same magnitude sweep both ways
and publish the comparison: recall per class under `dm-delay` versus under the
sleep fallback, at matched magnitudes.

That single cross-check is what makes every unprivileged run afterwards
defensible. Without it the fallback is an assumption; with it, it is a
characterised instrument with a stated error.

The comparison is a section in the results, not an appendix — it is the kind of
methodological care that is the entire reason anyone would cite this.

## Combination injections

Real pipelines stall for two reasons at once. Compose two injectors at the
harness level rather than writing one that does two jobs, so the labels stay
attributable:

```python
ds = DataloaderIO(3000).wrap_dataset(DataloaderCPU(3000).wrap_dataset(dataset))
```

Expect this to be where the classifier looks worst, and publish it anyway. A
tool that admits it cannot separate simultaneous decode and I/O stalls is more
trustworthy than one that silently picks whichever signal is marginally ahead —
which is why `refine_window` returns `dataloader_mixed` rather than guessing.
