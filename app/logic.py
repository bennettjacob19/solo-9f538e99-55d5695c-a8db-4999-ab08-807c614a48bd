"""Business logic: pool swap, leasing, simulated clock and withdrawal data.

Concurrency model (SQLite WAL, single writer):
  * Reads never block: the gateway process opens the database read-only.
  * Every state-changing request runs in one BEGIN IMMEDIATE transaction.
  * A prepared clock advance is staging data; leasing stays available while
    it is pending. COMMIT either publishes clock + free ranges + withdrawal
    together, or rolls everything back, so both processes always observe one
    consistent committed state.
"""
import ipaddress

from .common import (LEASE_SECONDS, MAX_SLOTS, slot_prefix, validate_pool)
from .db import connect


class DrillError(Exception):
    """Validation / state conflict reported to the operator (HTTP 4xx)."""


class ListChanged(DrillError):
    """The expiry set changed between prepare and commit: redo required."""


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def get_state(conn):
    meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
    pools = conn.execute("SELECT * FROM pools ORDER BY id").fetchall()
    pending = None
    if meta["pending_id"]:
        pending = conn.execute(
            "SELECT * FROM clock_advance WHERE id = ?",
            (meta["pending_id"],)).fetchone()
    return {"meta": meta, "pools": pools, "pending": pending}


def active_pool_id(conn, meta=None):
    meta = meta or conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
    return meta["new_pool_id"] or conn.execute(
        "SELECT id FROM pools WHERE is_new = 0").fetchone()["id"]


def free_summary(conn, pool_id):
    rows = conn.execute(
        "SELECT start_slot, end_slot FROM free_intervals"
        " WHERE pool_id = ? ORDER BY start_slot", (pool_id,)).fetchall()
    free = sum(r["end_slot"] - r["start_slot"] for r in rows)
    total = conn.execute(
        "SELECT slot_count FROM pools WHERE id = ?", (pool_id,)).fetchone()["slot_count"]
    return {"total": total, "free": free, "fragments": len(rows)}


# --------------------------------------------------------------------------
# Free-space management: persisted half-open [start, end) slot ranges
# --------------------------------------------------------------------------

def _take_first_slot(conn, pool_id):
    """First-fit allocation. Cost is one indexed lookup against free ranges;
    the number of leases in an occupied run never enters the scan."""
    row = conn.execute(
        "SELECT rowid, start_slot, end_slot FROM free_intervals"
        " WHERE pool_id = ? ORDER BY start_slot LIMIT 1", (pool_id,)).fetchone()
    if row is None:
        return None
    slot = row["start_slot"]
    if row["end_slot"] - slot == 1:
        conn.execute(
            "DELETE FROM free_intervals WHERE rowid = ?", (row["rowid"],))
    else:
        conn.execute(
            "UPDATE free_intervals SET start_slot = ? WHERE rowid = ?",
            (slot + 1, row["rowid"]))
    return slot


def _release_slot(conn, pool_id, slot):
    """Return a slot and coalesce every adjacent free range.

    Cost is O(k) where k is the number of fragments absorbed, never the
    number of leases in an occupied run. Within one clock-advance batch
    slots may be released in any order (e.g. 0,1,3), so we first collect the
    full connected component of ranges adjacent to [slot, slot+1), delete
    it, then insert one maximal [start, end) range.
    """
    start, end = slot, slot + 1
    # Absorb stored ranges that touch or overlap the growing interval:
    # range [a,b) joins exactly when a <= end AND b >= start. The SQL filter
    # narrows the candidates; the Python guard rejects ranges that pass the
    # loose bounds but are separated by a gap (e.g. [0,2) while start=3).
    while True:
        rows = conn.execute(
            "SELECT rowid, start_slot, end_slot FROM free_intervals"
            " WHERE pool_id = ? AND end_slot >= ? AND start_slot <= ?"
            " ORDER BY start_slot",
            (pool_id, start, end)).fetchall()
        neighbor = next((r for r in rows
                         if r["start_slot"] <= end and r["end_slot"] >= start),
                        None)
        if neighbor is None:
            break
        start = min(start, neighbor["start_slot"])
        end = max(end, neighbor["end_slot"])
        conn.execute("DELETE FROM free_intervals WHERE rowid = ?",
                     (neighbor["rowid"],))
    conn.execute(
        "INSERT INTO free_intervals(pool_id, start_slot, end_slot)"
        " VALUES (?, ?, ?)", (pool_id, start, end))


def _defragment(conn, pool_id):
    """Merge all mutually adjacent free ranges of one pool into maximal
    ranges. Cost is O(f) in the number of free fragments, independent of the
    number of leases. Called once after a clock-advance batch so releases in
    arbitrary order still collapse to the same final interval set."""
    rows = conn.execute(
        "SELECT rowid, start_slot, end_slot FROM free_intervals"
        " WHERE pool_id = ? ORDER BY start_slot", (pool_id,)).fetchall()
    if len(rows) <= 1:
        return
    merged_start, merged_end = rows[0]["start_slot"], rows[0]["end_slot"]
    stale = [rows[0]["rowid"]]
    for row in rows[1:]:
        if row["start_slot"] == merged_end:
            merged_end = row["end_slot"]
            stale.append(row["rowid"])
        else:
            if stale:
                conn.execute(
                    "UPDATE free_intervals SET start_slot = ?, end_slot = ?"
                    " WHERE rowid = ?",
                    (merged_start, merged_end, stale[0]))
                for rid in stale[1:]:
                    conn.execute(
                        "DELETE FROM free_intervals WHERE rowid = ?", (rid,))
            merged_start, merged_end = row["start_slot"], row["end_slot"]
            stale = [row["rowid"]]
    conn.execute(
        "UPDATE free_intervals SET start_slot = ?, end_slot = ? WHERE rowid = ?",
        (merged_start, merged_end, stale[0]))
    for rid in stale[1:]:
        conn.execute("DELETE FROM free_intervals WHERE rowid = ?", (rid,))


def _release_range(conn, pool_id, start, end):
    """Return a whole contiguous slot range [start, end) and coalesce it with
    every adjacent stored free range. One insert/delete per absorbed fragment
    plus one final insert: the cost depends on free fragments, never on how
    many leases the released range contained."""
    lo, hi = start, end
    while True:
        rows = conn.execute(
            "SELECT rowid, start_slot, end_slot FROM free_intervals"
            " WHERE pool_id = ? AND end_slot >= ? AND start_slot <= ?"
            " ORDER BY start_slot",
            (pool_id, lo, hi)).fetchall()
        neighbor = next((r for r in rows
                         if r["start_slot"] <= hi and r["end_slot"] >= lo),
                        None)
        if neighbor is None:
            break
        lo = min(lo, neighbor["start_slot"])
        hi = max(hi, neighbor["end_slot"])
        conn.execute("DELETE FROM free_intervals WHERE rowid = ?",
                     (neighbor["rowid"],))
    conn.execute(
        "INSERT INTO free_intervals(pool_id, start_slot, end_slot)"
        " VALUES (?, ?, ?)", (pool_id, lo, hi))


def _free_capacity(conn, pool_id):
    row = conn.execute(
        "SELECT COALESCE(SUM(end_slot - start_slot), 0) AS n"
        " FROM free_intervals WHERE pool_id = ?", (pool_id,)).fetchone()
    return row["n"]


# --------------------------------------------------------------------------
# New-pool segment leasing: one lease_segments row per contiguous slot run
# --------------------------------------------------------------------------

def _allocate_segments(conn, pool, count, now):
    """First-fit batch allocation from segment storage.

    Walks the persisted free ranges (ORDER BY start_slot) and carves
    contiguous chunks; every chunk costs exactly one free-range mutation and
    one lease_segments insert, regardless of how many slots it spans. The
    per-lease receipts are derived in memory (id = first_lease_id + offset),
    so write volume grows only with the number of contiguous slot runs the
    batch touches. Lease ids come from the global cursor meta.lease_seq and
    are therefore always larger than any legacy lease id.
    """
    pool_id = pool["id"]
    if count > _free_capacity(conn, pool_id):
        raise DrillError("池 %s 可领前缀不足（池满），本次请求未写入"
                         % pool["cidr"])
    network = ipaddress.IPv6Network(pool["cidr"])
    deleg_len = pool["deleg_len"]
    remaining = count
    segments = []   # (start, end, first_lease_id)
    receipts = []
    while remaining > 0:
        row = conn.execute(
            "SELECT rowid, start_slot, end_slot FROM free_intervals"
            " WHERE pool_id = ? ORDER BY start_slot LIMIT 1",
            (pool_id,)).fetchone()
        if row is None:
            raise DrillError("池 %s 已无可分配前缀（池满），本次请求未写入"
                             % pool["cidr"])
        take = min(remaining, row["end_slot"] - row["start_slot"])
        seg_start = row["start_slot"]
        seg_end = seg_start + take
        if seg_end == row["end_slot"]:
            conn.execute(
                "DELETE FROM free_intervals WHERE rowid = ?", (row["rowid"],))
        else:
            conn.execute(
                "UPDATE free_intervals SET start_slot = ? WHERE rowid = ?",
                (seg_end, row["rowid"]))
        first_id = conn.execute(
            "SELECT lease_seq FROM meta WHERE id = 1").fetchone()["lease_seq"]
        first_id += 1
        conn.execute(
            "INSERT INTO lease_segments(pool_id, start_slot, end_slot,"
            " first_lease_id, starts_at, expires_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, 'active')",
            (pool_id, seg_start, seg_end, first_id, now,
             now + LEASE_SECONDS))
        conn.execute("UPDATE meta SET lease_seq = ? WHERE id = 1",
                     (first_id + take - 1,))
        segments.append((seg_start, seg_end, first_id))
        for offset in range(take):
            slot = seg_start + offset
            receipts.append({
                "id": first_id + offset,
                "slot": slot,
                "prefix": slot_prefix(network, deleg_len, slot),
                "starts_at": now,
                "expires_at": now + LEASE_SECONDS,
            })
        remaining -= take
    return segments, receipts


def _segment_lease_ids(seg_row):
    """Expand the consecutive lease-id range of one segment, in memory only."""
    width = seg_row["end_slot"] - seg_row["start_slot"]
    return list(range(seg_row["first_lease_id"],
                      seg_row["first_lease_id"] + width))


def _due_segments(conn, target):
    return conn.execute(
        "SELECT * FROM lease_segments WHERE status = 'active'"
        " AND expires_at <= ? ORDER BY id", (target,)).fetchall()


# --------------------------------------------------------------------------
# Pool swap
# --------------------------------------------------------------------------

def change_pool(conn, cidr_text, deleg_len):
    """Validate and bring the new pool online. Invalid input raises before
    any write, so the previous state is preserved exactly."""
    try:
        network, deleg_len, slot_count = validate_pool(cidr_text, int(deleg_len))
    except (TypeError, ValueError) as exc:
        raise DrillError("新池配置无效，已保留原状：%s" % exc)
    cidr = str(network)

    conn.execute("BEGIN IMMEDIATE")
    try:
        meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        if meta["new_pool_id"] is not None:
            raise DrillError("本演练为单次换池，新池已上线，不能再次换池")
        if meta["pending_id"] is not None:
            raise DrillError("时钟推进处于暂停待提交状态，请先提交或退回后再换池")
        dup = conn.execute("SELECT 1 FROM pools WHERE cidr = ?", (cidr,)).fetchone()
        if dup:
            raise DrillError("该 CIDR 已存在，配置无效，已保留原状")
        now = meta["now_seconds"]
        cur = conn.execute(
            "INSERT INTO pools(cidr, prefix_len, deleg_len, slot_count, is_new)"
            " VALUES (?, ?, ?, ?, 1)",
            (cidr, network.prefixlen, deleg_len, slot_count))
        new_id = cur.lastrowid
        # One range covering the whole pool: no candidate prefixes expanded.
        conn.execute(
            "INSERT INTO free_intervals(pool_id, start_slot, end_slot)"
            " VALUES (?, 0, ?)", (new_id, slot_count))
        # Arm the global lease-id cursor above every existing lease id so
        # segment-derived ids never collide with legacy ids. New-pool leases
        # always receive fresh ids even when they reuse a released prefix.
        max_lease_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM leases").fetchone()["m"]
        conn.execute(
            "UPDATE meta SET new_pool_id = ?, lease_seq = ? WHERE id = 1",
            (new_id, max_lease_id))
        conn.execute(
            "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
            " detail, at_seconds) VALUES (NULL, 'pool-swap', ?, ?, ?, ?)",
            (now, now, "新池 %s 上线，委派 /%d，共 %d 个前缀；旧租约按原期限到期"
             % (cidr, deleg_len, slot_count), now))
        _republish_withdrawal(conn)
        conn.execute("COMMIT")
        return new_id, cidr, deleg_len, slot_count
    except Exception:
        conn.execute("ROLLBACK")
        raise


# --------------------------------------------------------------------------
# Leasing
# --------------------------------------------------------------------------

def allocate(conn, count=1):
    """Issue 60-minute lease(s) from the active pool, skipping unexpired
    prefixes via the free ranges. Available even while an advance is paused.

    The built-in old pool keeps one row per lease (legacy ids unchanged).
    The new pool stores one lease_segments row per contiguous slot run, so a
    batch of thousands of adjacent prefixes performs only O(runs) writes;
    per-lease receipts are derived in memory and still returned one by one.
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise DrillError("领址数量必须是整数")
    if count < 1 or count > MAX_SLOTS:
        raise DrillError("领址数量需在 1..%d 之间" % MAX_SLOTS)

    conn.execute("BEGIN IMMEDIATE")
    try:
        meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        pool_id = active_pool_id(conn, meta)
        pool = conn.execute(
            "SELECT * FROM pools WHERE id = ?", (pool_id,)).fetchone()
        now = meta["now_seconds"]
        if pool["is_new"]:
            return _allocate_new_pool(conn, meta, pool, count, now)
        return _allocate_legacy(conn, meta, pool, count, now)
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _allocate_new_pool(conn, meta, pool, count, now):
    """Segment-backed allocation path. Disk writes per batch scale with the
    number of contiguous slot runs touched; receipts stay per-lease."""
    segments, issued = _allocate_segments(conn, pool, count, now)
    seg_text = ",".join(
        "槽[%d,%d)→租约id[%d,%d)"
        % (s, e, fid, fid + e - s - 1) for s, e, fid in segments)
    conn.execute(
        "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
        " detail, at_seconds) VALUES (NULL, 'lease', ?, ?, ?, ?)",
        (now, now, "从 %s 发放 %d 个 60 分钟租约，共 %d 个连续段（%s）"
         % (pool["cidr"], len(issued), len(segments), seg_text), now))
    conn.execute("COMMIT")
    return pool["cidr"], issued


def _allocate_legacy(conn, meta, pool, count, now):
    """Original per-lease allocation path for the built-in old pool."""
    network = ipaddress.IPv6Network(pool["cidr"])
    issued = []
    for _ in range(count):
        slot = _take_first_slot(conn, pool["id"])
        if slot is None:
            raise DrillError("池 %s 已无可分配前缀（池满），本次请求未写入"
                             % pool["cidr"])
        pfx = slot_prefix(network, pool["deleg_len"], slot)
        cur = conn.execute(
            "INSERT INTO leases(pool_id, slot, prefix, starts_at,"
            " expires_at, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (pool["id"], slot, pfx, now, now + LEASE_SECONDS))
        issued.append({"id": cur.lastrowid, "prefix": pfx, "slot": slot,
                       "starts_at": now, "expires_at": now + LEASE_SECONDS})
    conn.execute(
        "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
        " detail, at_seconds) VALUES (NULL, 'lease', ?, ?, ?, ?)",
        (now, now, "从 %s 发放 %d 个 60 分钟租约（租约 id %s）"
         % (pool["cidr"], len(issued),
            ",".join(str(x["id"]) for x in issued)), now))
    conn.execute("COMMIT")
    return pool["cidr"], issued


# --------------------------------------------------------------------------
# Simulated clock: prepare (pause) / commit / abort
# --------------------------------------------------------------------------

def _expiry_set(conn, target):
    rows = conn.execute(
        "SELECT id FROM leases WHERE status = 'active' AND expires_at <= ?"
        " ORDER BY id", (target,)).fetchall()
    return [r["id"] for r in rows]


def _snapshot_overview(conn, advance_id):
    """Human/API view of a stored snapshot: legacy lease ids plus one entry
    per new-pool segment (with its derived lease-id span)."""
    rows = conn.execute(
        "SELECT kind, ref_id FROM advance_snapshot WHERE advance_id = ?"
        " ORDER BY kind, ref_id", (advance_id,)).fetchall()
    legacy_ids, segment_entries = [], []
    for r in rows:
        if r["kind"] == "legacy":
            legacy_ids.append(r["ref_id"])
        else:
            seg = conn.execute(
                "SELECT * FROM lease_segments WHERE id = ?",
                (r["ref_id"],)).fetchone()
            ids = _segment_lease_ids(seg)
            segment_entries.append({
                "segment_id": seg["id"],
                "slot_start": seg["start_slot"], "slot_end": seg["end_slot"],
                "lease_id_start": ids[0], "lease_id_end": ids[-1],
                "lease_ids": ids,
            })
    return legacy_ids, segment_entries


def prepare_advance(conn, target):
    try:
        target = int(target)
    except (TypeError, ValueError):
        raise DrillError("目标时刻必须是整数秒")
    conn.execute("BEGIN IMMEDIATE")
    try:
        meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        if meta["pending_id"] is not None:
            raise DrillError("已有暂停中的时钟推进，请先提交或退回")
        now = meta["now_seconds"]
        if target <= now:
            raise DrillError("目标时刻必须晚于当前模拟时钟 %d" % now)
        ids = _expiry_set(conn, target)
        due_segments = _due_segments(conn, target)
        max_legacy = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM leases").fetchone()["m"]
        max_seg_cursor = conn.execute(
            "SELECT COALESCE(MAX(lease_seq), 0) AS m FROM meta"
            " WHERE id = 1").fetchone()["m"]
        cur = conn.execute(
            "INSERT INTO clock_advance(target_seconds, snapshot_ids,"
            " snapshot_max_id, state) VALUES (?, ?, ?, 'pending')",
            # snapshot_ids keeps a legacy-only summary for history/UI; the
            # authoritative due set lives in advance_snapshot.
            (target, ",".join(map(str, ids)),
             max(max_legacy, max_seg_cursor)))
        advance_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO advance_snapshot(advance_id, kind, ref_id)"
            " VALUES (?, 'legacy', ?)",
            [(advance_id, lease_id) for lease_id in ids])
        conn.executemany(
            "INSERT INTO advance_snapshot(advance_id, kind, ref_id)"
            " VALUES (?, 'segment', ?)",
            [(advance_id, seg["id"]) for seg in due_segments])
        conn.execute(
            "UPDATE meta SET pending_id = ? WHERE id = 1", (advance_id,))
        seg_lease_total = sum(
            s["end_slot"] - s["start_slot"] for s in due_segments)
        conn.execute(
            "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
            " detail, at_seconds) VALUES (?, 'prepare', ?, ?, ?, ?)",
            (advance_id, now, target,
             "暂停待提交，本次预计到期：旧池租约 %d 条（%s）；新池连续段 %d 段"
             " 共 %d 条租约（段 %s）"
             % (len(ids), ",".join(map(str, ids)) or "无",
                len(due_segments), seg_lease_total,
                ",".join(str(s["id"]) for s in due_segments) or "无"), now))
        conn.execute("COMMIT")
        # API still reports every due lease id, expanded from segments in
        # memory; the on-disk snapshot only stores segment references.
        segment_ids = [lid for s in due_segments for lid in _segment_lease_ids(s)]
        all_ids = sorted(ids + segment_ids)
        return advance_id, target, all_ids
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _abort_pending(conn, advance_id, reason, now):
    conn.execute(
        "UPDATE clock_advance SET state = 'aborted' WHERE id = ?", (advance_id,))
    conn.execute("UPDATE meta SET pending_id = NULL WHERE id = 1")
    conn.execute(
        "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
        " detail, at_seconds) VALUES (?, 'abort', ?, ?, ?, ?)",
        (advance_id, now, now, reason, now))


def commit_advance(conn, fault_inject=False):
    """Atomically expire this batch, advance the clock and republish the
    withdrawal conclusion. New leases taken during the pause that expire in
    the same batch are included. A changed expiry list aborts with the
    original clock and all issued leases intact.

    Legacy leases are still settled one row at a time; new-pool leases are
    settled per segment (one UPDATE + one range release per segment), so the
    commit's write volume grows with the number of contiguous slot runs
    involved, not the number of leases.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        if not meta["pending_id"]:
            raise DrillError("没有暂停中的时钟推进可提交")
        adv = conn.execute(
            "SELECT * FROM clock_advance WHERE id = ?",
            (meta["pending_id"],)).fetchone()
        target = adv["target_seconds"]
        snap_legacy, snap_segments = _snapshot_overview(conn, adv["id"])
        current_legacy = _expiry_set(conn, target)
        current_segments = _due_segments(conn, target)

        # List-change detection against the stored snapshot references.
        snap_legacy_set = set(snap_legacy)
        cur_legacy_set = set(current_legacy)
        missing_legacy = sorted(snap_legacy_set - cur_legacy_set)
        snap_seg_ids = {s["segment_id"] for s in snap_segments}
        cur_seg_ids = {s["id"] for s in current_segments}
        missing_segments = sorted(snap_seg_ids - cur_seg_ids)
        if missing_legacy or missing_segments:
            _abort_pending(
                conn, adv["id"],
                "到期名单变化（缺失旧池租约 %s、新池段 %s），退回并保留原时钟与"
                "已领租约，请重做"
                % (",".join(map(str, missing_legacy)) or "无",
                   ",".join(map(str, missing_segments)) or "无"),
                meta["now_seconds"])
            conn.execute("COMMIT")
            raise ListChanged(
                "到期名单变化，已退回，请重做（原时钟 %d 与已领租约均保留）"
                % meta["now_seconds"])

        added_legacy = sorted(cur_legacy_set - snap_legacy_set)
        added_segments = [s for s in current_segments
                          if s["id"] not in snap_seg_ids]
        touched_pools = set()

        # Legacy path: per-lease expiry and per-slot release (unchanged).
        for lease_id in current_legacy:
            row = conn.execute(
                "SELECT pool_id, slot FROM leases WHERE id = ?",
                (lease_id,)).fetchone()
            conn.execute(
                "UPDATE leases SET status = 'expired' WHERE id = ?",
                (lease_id,))
            _release_slot(conn, row["pool_id"], row["slot"])
            touched_pools.add(row["pool_id"])
        for pool_id in touched_pools:
            _defragment(conn, pool_id)

        # New-pool path: one UPDATE and one range release per segment.
        settled_seg_lease_ids = []
        for seg in current_segments:
            conn.execute(
                "UPDATE lease_segments SET status = 'expired' WHERE id = ?",
                (seg["id"],))
            _release_range(conn, seg["pool_id"],
                           seg["start_slot"], seg["end_slot"])
            settled_seg_lease_ids.extend(_segment_lease_ids(seg))
        if current_segments:
            _defragment(conn, current_segments[0]["pool_id"])

        settled = sorted(current_legacy + settled_seg_lease_ids)
        added = sorted(added_legacy
                       + [lid for s in added_segments
                          for lid in _segment_lease_ids(s)])

        conn.execute(
            "UPDATE meta SET now_seconds = ?, pending_id = NULL WHERE id = 1",
            (target,))
        conn.execute(
            "UPDATE clock_advance SET state = 'committed' WHERE id = ?",
            (adv["id"],))
        seg_detail = ",".join(
            "段%d槽[%d,%d)→租约id[%d,%d]"
            % (s["id"], s["start_slot"], s["end_slot"],
               _segment_lease_ids(s)[0], _segment_lease_ids(s)[-1])
            for s in current_segments) or "无"
        conn.execute(
            "INSERT INTO clock_log(advance_id, event, from_seconds, to_seconds,"
            " detail, at_seconds) VALUES (?, 'commit', ?, ?, ?, ?)",
            (adv["id"], meta["now_seconds"], target,
             "本次到期 %d 条：旧池 %d 条（%s）；新池 %d 段 %d 条（%s）%s"
             % (len(settled), len(current_legacy),
                ",".join(map(str, current_legacy)) or "无",
                len(current_segments), len(settled_seg_lease_ids),
                seg_detail,
                ("；其中暂停期间新领并计入：%s" % ",".join(map(str, added)))
                if added else ""), target))
        _republish_withdrawal(conn)
        if fault_inject:
            # Simulate losing the process immediately before COMMIT. SQLite
            # discards the open transaction on close; recovery reopens from
            # the last committed state.
            conn.close()
            raise RuntimeError("FAULT_INJECTED: process lost before COMMIT")
        conn.execute("COMMIT")
        return {"target": target, "settled": settled,
                "settled_segments": [
                    {"segment_id": s["id"],
                     "lease_ids": _segment_lease_ids(s)}
                    for s in current_segments],
                "added": added}
    except DrillError:
        raise
    except RuntimeError:
        raise  # fault injection: connection already closed
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def abort_advance(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
        if not meta["pending_id"]:
            raise DrillError("没有暂停中的时钟推进可退回")
        _abort_pending(conn, meta["pending_id"],
                       "值班员手动退回，时钟与已领租约保持不变",
                       meta["now_seconds"])
        conn.execute("COMMIT")
        return meta["pending_id"]
    except Exception:
        conn.execute("ROLLBACK")
        raise


# --------------------------------------------------------------------------
# Withdrawal publication (written here, read by the independent gateway)
# --------------------------------------------------------------------------

def _republish_withdrawal(conn):
    old = conn.execute("SELECT * FROM pools WHERE is_new = 0").fetchone()
    meta = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
    new_net = None
    new_id = None
    if meta["new_pool_id"]:
        new_row = conn.execute(
            "SELECT * FROM pools WHERE id = ?",
            (meta["new_pool_id"],)).fetchone()
        new_id = new_row["id"]
        new_net = ipaddress.IPv6Network(new_row["cidr"])

    uncovered = []
    rows = conn.execute(
        "SELECT * FROM leases WHERE pool_id = ? AND status = 'active'"
        " ORDER BY expires_at, id", (old["id"],)).fetchall()
    for row in rows:
        if new_net is None or not _is_covered(row["prefix"], new_net):
            uncovered.append(row)
    earliest = uncovered[0]["expires_at"] if uncovered else None
    conn.execute(
        "UPDATE withdrawal_publish SET pool_id = ?, cidr = ?,"
        " earliest_seconds = ?, published_at = ? WHERE id = 1",
        (new_id, str(new_net) if new_net else None, earliest,
         meta["now_seconds"]))


def _is_covered(prefix_text, new_network):
    from .common import is_covered
    return is_covered(prefix_text, new_network)


def read_published(conn):
    """Gateway-side read: the published earliest safe withdrawal time."""
    pub = conn.execute(
        "SELECT * FROM withdrawal_publish WHERE id = 1").fetchone()
    return pub


def read_uncovered_old_leases(conn):
    """Gateway-side read: valid old leases not fully covered by the new
    aggregate. Pure SELECT against committed data only."""
    pub = read_published(conn)
    old = conn.execute("SELECT * FROM pools WHERE is_new = 0").fetchone()
    rows = conn.execute(
        "SELECT * FROM leases WHERE pool_id = ? AND status = 'active'"
        " ORDER BY expires_at, id", (old["id"],)).fetchall()
    new_net = ipaddress.IPv6Network(pub["cidr"]) if pub["cidr"] else None
    out = []
    for row in rows:
        if new_net is None or not _is_covered(row["prefix"], new_net):
            out.append(dict(row))
    return out


def active_lease_count(conn, pool_id, is_new):
    """Active lease count of a pool. New-pool segments store the count as
    summed slot widths, so no per-lease enumeration is needed."""
    if is_new:
        row = conn.execute(
            "SELECT COALESCE(SUM(end_slot - start_slot), 0) AS c"
            " FROM lease_segments WHERE pool_id = ? AND status = 'active'",
            (pool_id,)).fetchone()
        return row["c"]
    return conn.execute(
        "SELECT COUNT(*) AS c FROM leases"
        " WHERE pool_id = ? AND status = 'active'",
        (pool_id,)).fetchone()["c"]


def recent_leases(conn, limit=30):
    """Latest `limit` individual leases for the dashboard.

    Legacy rows come straight from leases. New-pool leases are *derived* from
    segment rows in memory (no per-lease storage): segments are scanned from
    the highest first_lease_id and expanded from the tail until the limit is
    reached, so the result stays bounded by 30 regardless of pool size.
    """
    legacy = conn.execute(
        "SELECT l.id, l.pool_id, p.cidr AS pool_cidr, l.slot, l.prefix,"
        " l.starts_at, l.expires_at, l.status FROM leases l"
        " JOIN pools p ON p.id = l.pool_id ORDER BY l.id DESC"
        " LIMIT ?", (limit,)).fetchall()
    out = [dict(r) for r in legacy]
    if len(out) >= limit:
        return out[:limit]
    segs = conn.execute(
        "SELECT s.*, p.cidr AS pool_cidr, p.deleg_len FROM lease_segments s"
        " JOIN pools p ON p.id = s.pool_id"
        " ORDER BY s.first_lease_id DESC").fetchall()
    for seg in segs:
        network = ipaddress.IPv6Network(seg["pool_cidr"])
        width = seg["end_slot"] - seg["start_slot"]
        # Expand from the highest lease id backwards within this segment.
        for offset in range(width - 1, -1, -1):
            slot = seg["start_slot"] + offset
            out.append({
                "id": seg["first_lease_id"] + offset,
                "pool_id": seg["pool_id"],
                "pool_cidr": seg["pool_cidr"],
                "slot": slot,
                "prefix": slot_prefix(network, seg["deleg_len"], slot),
                "starts_at": seg["starts_at"],
                "expires_at": seg["expires_at"],
                "status": seg["status"],
            })
            if len(out) >= limit:
                out.sort(key=lambda r: r["id"], reverse=True)
                return out[:limit]
    out.sort(key=lambda r: r["id"], reverse=True)
    return out[:limit]


def recent_logs(conn, limit=30):
    return conn.execute(
        "SELECT * FROM clock_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
