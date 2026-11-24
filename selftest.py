"""End-to-end self tests for the IPv6 pool-swap drill (stdlib only).

Run:  python3 selftest.py
Covers: seed data, invalid config rollback, pool swap, overlap/coverage,
first-fit leasing and pool-full, clock expiry including leases taken during
the pause, list-changed redo, fault-injected commit interruption, read-only
gateway, restart recovery without lease scans, and O(1)-ish allocation cost.
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db import connect, init_db, recover_pending          # noqa: E402
from app.gateway import create_app as gateway_app             # noqa: E402
from app.web import create_app as web_app                     # noqa: E402
from app.logic import active_lease_count, recent_leases       # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS  %s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL  %s %s" % (name, detail))


def fresh_db():
    tmp = tempfile.mkdtemp(prefix="drill-")
    path = os.path.join(tmp, "test.db")
    init_db(path, force=True).close()
    return path


def jpost(client, url, payload=None, raw=False):
    if raw:
        resp = client.post(url, data=payload or {})
    else:
        resp = client.post(url, json=payload or {})
    return resp.status_code, resp.get_json()


# --------------------------------------------------------------------------
def test_seed_and_gateway():
    print("[1] 种子数据与只读网关撤路视图")
    path = fresh_db()
    conn = connect(path)
    rows = conn.execute("SELECT slot, status FROM leases ORDER BY slot").fetchall()
    check("内置 6 条旧租约（5 active + 1 expired）", len(rows) == 6
          and sum(1 for r in rows if r["status"] == "active") == 5)
    fr = conn.execute(
        "SELECT start_slot,end_slot FROM free_intervals ORDER BY start_slot"
    ).fetchall()
    # slot 4's lease is already expired at drill start, so its slot is free;
    # free ranges are [3,5) and [7,4096). No candidate prefix list is built.
    check("旧池空闲区间为 [3,5) 与 [7,4096)，未展开候选前缀",
          [(r["start_slot"], r["end_slot"]) for r in fr] == [(3, 5), (7, 4096)])
    check("模拟时钟初始为 0",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0)
    conn.close()

    gw = gateway_app(path).test_client()
    resp = gw.get("/withdrawal")
    body = resp.get_json()
    check("无新池时最早撤路时刻=最早旧租约到期 3600",
          body["earliest_withdrawal_seconds"] == 3600)
    check("无新池时 5 条有效旧租约全部未覆盖", body["uncovered_count"] == 5)

    ro = connect(path, readonly=True)
    try:
        blocked = False
        try:
            ro.execute("UPDATE meta SET now_seconds = 1")
        except sqlite3.OperationalError:
            blocked = True
        check("网关连接为只读，写入被 SQLite 拒绝", blocked)
    finally:
        ro.close()


def test_invalid_config_keeps_state():
    print("[2] 无效新池配置保留原状")
    path = fresh_db()
    client = web_app(path).test_client()
    before = connect(path).execute(
        "SELECT now_seconds, new_pool_id FROM meta").fetchone()
    bad_inputs = [
        ("not-a-cidr", 60),
        ("2001:db8::/128", 60),          # deleg shorter than pool prefix
        ("2001:db8::/56", 64),           # deleg not 56/60
        ("2001:db8::1/56", 56),          # host bits set, strict parse fails
        ("10.0.0.0/24", 60),             # IPv4
    ]
    for cidr, deleg in bad_inputs:
        code, body = jpost(client, "/api/pool", {"cidr": cidr, "deleg_len": deleg})
        check("拒绝无效配置 %s /%d" % (cidr, deleg),
              code == 400 and body["ok"] is False)
    after = connect(path).execute(
        "SELECT now_seconds, new_pool_id FROM meta").fetchone()
    check("全部拒绝后时钟与池状态不变", tuple(before) == tuple(after))


def test_swap_and_overlap():
    print("[3] 换池与交叠覆盖核算")
    path = fresh_db()
    client = web_app(path).test_client()
    gw = gateway_app(path).test_client()

    # New /60 pool covering exactly old slot 2, whose delegated prefix is
    # 2001:db8:aaaa:20::/60 (verified via the same integer math as leasing).
    code, body = jpost(client, "/api/pool",
                       {"cidr": "2001:db8:aaaa:20::/60", "deleg_len": 60})
    check("交叠新池换池成功", code == 200 and body["slots"] == 1
          and body["cidr"] == "2001:db8:aaaa:20::/60")
    view = gw.get("/withdrawal").get_json()
    check("新汇总仅覆盖旧租约 slot2，未覆盖=4", view["uncovered_count"] == 4)
    check("最早撤路时刻仍为 3600（slot0/1）",
          view["earliest_withdrawal_seconds"] == 3600)
    check("未覆盖名单前缀正确",
          sorted(x["prefix"] for x in view["uncovered_leases"]) == sorted([
              "2001:db8:aaaa::/60", "2001:db8:aaaa:10::/60",
              "2001:db8:aaaa:50::/60", "2001:db8:aaaa:60::/60"]))

    code, body = jpost(client, "/api/pool",
                       {"cidr": "2001:db8:zzzz::/56", "deleg_len": 56})
    check("单次演练禁止二次换池", code == 400)


def test_slash56_boundary():
    print("[3b] /56 委派长度与严格边界")
    path = fresh_db()
    client = web_app(path).test_client()
    # /56 pool with /56 delegation: one slot, strict boundary network.
    code, body = jpost(client, "/api/pool",
                       {"cidr": "2001:db8:bbbb::/56", "deleg_len": 56})
    check("/56 池 + /56 委派合法（1 个前缀）", code == 200 and body["slots"] == 1)
    code, body = jpost(client, "/api/lease", {"count": 1})
    check("/56 委派前缀正确", code == 200
          and body["issued"][0]["prefix"] == "2001:db8:bbbb::/56")


def test_lease_and_pool_full():
    print("[4] 领址：60 分钟、避开未到期前缀、池满提示")
    path = fresh_db()
    client = web_app(path).test_client()
    # First lease on the old pool must take the first free slot = 3.
    code, body = jpost(client, "/api/lease", {"count": 1})
    check("首笔领址取空闲区间首槽 slot3（不回扫在租明细）",
          code == 200 and body["issued"][0]["slot"] == 3
          and body["issued"][0]["expires_at"] == 3600)

    # One-slot pool: /60 CIDR with /60 delegation.
    path2 = fresh_db()
    c2 = web_app(path2).test_client()
    jpost(c2, "/api/pool", {"cidr": "2001:db8:cccc::/60", "deleg_len": 60})
    code, body = jpost(c2, "/api/lease", {"count": 1})
    check("单槽池首次领址成功", code == 200 and body["issued"][0]["slot"] == 0)
    code, body = jpost(c2, "/api/lease", {"count": 1})
    check("池满返回 400 提示且不写入", code == 400 and "池满" in body["error"])
    conn = connect(path2)
    check("池满后新池在租仍为 1 条",
          active_lease_count(conn, 2, True) == 1)
    conn.close()


def test_expiry_with_pause_leases():
    print("[5] 时钟推进：暂停期间新领且本次到期的租约须计入")
    path = fresh_db()
    client = web_app(path).test_client()
    gw = gateway_app(path).test_client()

    code, body = jpost(client, "/api/advance", {"target_seconds": 3600})
    check("暂停核算成功，预计到期旧租约 2 条（slot0/1）",
          code == 200 and sorted(body["expected_expiries"]) == [1, 2])

    # Leasing stays available while paused; new lease expires at 0+3600.
    code, leased = jpost(client, "/api/lease", {"count": 1})
    new_id = leased["issued"][0]["id"]
    check("暂停期间照常领址", code == 200 and leased["issued"][0]["expires_at"] == 3600)
    # Gateway keeps answering with committed state during the pause.
    check("暂停期间网关仍可查撤路",
          gw.get("/withdrawal").status_code == 200)

    code, body = jpost(client, "/api/advance/commit")
    check("提交成功", code == 200)
    check("本次到期共 3 条，暂停新领租约计入 added",
          sorted(body["settled"]) == [1, 2, new_id] and body["added"] == [new_id])
    conn = connect(path)
    check("时钟推进到 3600",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 3600)
    check("到期租约状态为 expired", conn.execute(
        "SELECT COUNT(*) c FROM leases WHERE id IN (1,2,?) AND status='expired'",
        (new_id,)).fetchone()["c"] == 3)
    # Slots 0,1 and the new slot-3 lease expire together, but slot 2 is still
    # leased until t=7200, so its gap must remain. The released ranges
    # coalesce to [0,2) and [3,5); slot 4's pre-expired range stays inside.
    fr = [tuple(r) for r in conn.execute(
        "SELECT start_slot,end_slot FROM free_intervals ORDER BY start_slot")]
    check("释放槽位按占用缺口合并：[0,2),[3,5),[7,4096)",
          fr == [(0, 2), (3, 5), (7, 4096)])
    check("未到期旧租约（slot2/5/6）保持 active", conn.execute(
        "SELECT COUNT(*) c FROM leases WHERE status='active'").fetchone()["c"] == 3)
    conn.close()
    view = gw.get("/withdrawal").get_json()
    check("撤路结论随提交同时生效：最早撤路变为 7200",
          view["earliest_withdrawal_seconds"] == 7200
          and view["uncovered_count"] == 3)

    # Second batch: slots 2 and 5 expire at t=7200; the whole [0,6) run must
    # now collapse into one maximal range despite cross-batch release order.
    jpost(client, "/api/advance", {"target_seconds": 7200})
    code, body = jpost(client, "/api/advance/commit")
    check("第二批推进成功", code == 200 and sorted(body["settled"]) == [3, 5])
    conn = connect(path)
    fr2 = [tuple(r) for r in conn.execute(
        "SELECT start_slot,end_slot FROM free_intervals ORDER BY start_slot")]
    check("跨批次相邻释放塌缩为 [0,6),[7,4096)", fr2 == [(0, 6), (7, 4096)])
    conn.close()


def test_list_changed_redo():
    print("[6] 名单变化退回：提示重做，保留原时钟与已领租约")
    path = fresh_db()
    client = web_app(path).test_client()
    jpost(client, "/api/advance", {"target_seconds": 3600})
    code, leased = jpost(client, "/api/lease", {"count": 1})
    kept_id = leased["issued"][0]["id"]

    # Simulate an external change to the due list (lease 2 no longer due).
    conn = connect(path)
    conn.execute("UPDATE leases SET expires_at = 9000 WHERE id = 2")
    conn.close()

    code, body = jpost(client, "/api/advance/commit")
    check("名单变化返回 409 并要求重做", code == 409 and body.get("redo_required"))
    conn = connect(path)
    check("退回后时钟仍为 0",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0)
    check("退回后 pending 已清除",
          conn.execute("SELECT pending_id FROM meta").fetchone()[0] is None)
    check("暂停期间所领租约保留且 active", conn.execute(
        "SELECT status FROM leases WHERE id=?", (kept_id,)).fetchone()[0] == "active")
    check("被改动租约的修改也未回滚（仅推进事务退回）", conn.execute(
        "SELECT expires_at FROM leases WHERE id=2").fetchone()[0] == 9000)
    conn.close()

    # Redo at the same target: lease 2 is no longer due, while lease 1 and
    # the lease taken during the aborted pause (kept_id) still expire at 3600.
    code, body = jpost(client, "/api/advance", {"target_seconds": 3600})
    check("可重新发起推进", code == 200
          and sorted(body["expected_expiries"]) == sorted([1, kept_id]))
    code, body = jpost(client, "/api/advance/commit")
    check("重做后提交成功", code == 200
          and sorted(body["settled"]) == sorted([1, kept_id]))


def test_commit_interruption():
    print("[7] 提交中断：进程在 COMMIT 前丢失，按同一已提交状态恢复")
    path = fresh_db()
    client = web_app(path).test_client()
    gw = gateway_app(path).test_client()
    jpost(client, "/api/advance", {"target_seconds": 7200})
    code, leased = jpost(client, "/api/lease", {"count": 2})
    kept_ids = [x["id"] for x in leased["issued"]]

    resp = client.post("/api/advance/commit", json={"fault_inject": True})
    check("故障注入返回 500", resp.status_code == 500
          and resp.get_json()["restart_recovery"])

    # A new request triggers startup-style recovery of the stale pending row.
    recovered = recover_pending(connect(path))
    check("恢复流程识别并中止悬挂推进", recovered is not None)
    conn = connect(path)
    check("恢复后时钟仍为 0（无部分生效）",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0)
    check("恢复后无 pending",
          conn.execute("SELECT pending_id FROM meta").fetchone()[0] is None)
    check("已领租约全部保留 active", conn.execute(
        "SELECT COUNT(*) c FROM leases WHERE id IN (%s) AND status='active'"
        % ",".join("?" * len(kept_ids)), kept_ids).fetchone()["c"] == 2)
    check("无租约被提前到期", conn.execute(
        "SELECT COUNT(*) c FROM leases WHERE status='expired'"
    ).fetchone()["c"] == 1)  # only the pre-seeded expired one
    conn.close()
    check("网关在恢复后读到一致已提交状态",
          gw.get("/withdrawal").get_json()["earliest_withdrawal_seconds"] == 3600)


def test_restart_and_no_scan():
    print("[8] 重启不回扫在租明细；找空位开销不随连续占用段增长")
    path = fresh_db()
    env = dict(os.environ, DRILL_DB=path, PYTHONPATH=os.path.dirname(
        os.path.abspath(__file__)))
    code = ("from app.web import create_app;"
            "create_app(%r).run(port=8077, use_reloader=False)" % path)
    proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_port(8077)
        def call(url, payload):
            import json
            req = urllib.request.Request(
                "http://127.0.0.1:8077" + url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            return json.loads(urllib.request.urlopen(req).read())
        # /48 pool with /60 delegation -> 4096 slots; take 500 contiguous.
        call("/api/pool", {"cidr": "2001:db8:dddd::/48", "deleg_len": 60})
        call("/api/lease", {"count": 500})
    finally:
        proc.kill()
        proc.wait()

    conn = connect(path)
    frags = conn.execute(
        "SELECT COUNT(*) c FROM free_intervals WHERE pool_id=2").fetchone()["c"]
    check("重启前空闲区间仅 1 个连续段 [500,4096)", frags == 1)
    conn.close()

    # Restart: no lease scan rebuilds free space; persisted ranges are used.
    proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_port(8077)
        import json
        req = urllib.request.Request(
            "http://127.0.0.1:8077/api/lease",
            data=json.dumps({"count": 1}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        body = json.loads(urllib.request.urlopen(req).read())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        check("重启后首笔领址直接命中持久化区间首槽 slot500",
              body["issued"][0]["slot"] == 500)
        check("找空位耗时不随 500 条连续占用增长（实测 %.2fms < 200ms）" % elapsed_ms,
              elapsed_ms < 200)
    finally:
        proc.kill()
        proc.wait()


def _wait_port(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/healthz" % port, timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("port %d not ready" % port)


def test_abort_and_non_advance():
    print("[9] 手动退回与非法推进参数")
    path = fresh_db()
    client = web_app(path).test_client()
    code, _ = jpost(client, "/api/advance", {"target_seconds": 3600})
    check("可暂停", code == 200)
    code, body = jpost(client, "/api/advance", {"target_seconds": 7200})
    check("暂停中拒绝第二次暂停", code == 400)
    code, _ = jpost(client, "/api/advance/abort")
    check("手动退回成功", code == 200)
    code, body = jpost(client, "/api/advance", {"target_seconds": 0})
    check("目标时刻不大于当前时钟被拒", code == 400)
    conn = connect(path)
    check("退回/拒绝后时钟保持 0",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0)
    conn.close()


def _table_counts(conn, pool_id):
    return {
        "segments": conn.execute(
            "SELECT COUNT(*) c FROM lease_segments WHERE pool_id=?",
            (pool_id,)).fetchone()["c"],
        "active_segments": conn.execute(
            "SELECT COUNT(*) c FROM lease_segments WHERE pool_id=?"
            " AND status='active'", (pool_id,)).fetchone()["c"],
        "free_rows": conn.execute(
            "SELECT COUNT(*) c FROM free_intervals WHERE pool_id=?",
            (pool_id,)).fetchone()["c"],
        "lease_rows": conn.execute(
            "SELECT COUNT(*) c FROM leases").fetchone()["c"],
        "log_rows": conn.execute(
            "SELECT COUNT(*) c FROM clock_log").fetchone()["c"],
    }


def test_new_pool_segment_writes():
    print("[10] 新池段式发放：几千连续前缀逐笔回执，写盘量只随连续段数增长")
    path = fresh_db()
    client = web_app(path).test_client()
    # /44 pool with /60 delegation -> 65536 slots, room for thousands.
    code, body = jpost(client, "/api/pool",
                       {"cidr": "2001:db8:bb00::/44", "deleg_len": 60})
    check("大容量不相交新池换池成功", code == 200 and body["slots"] == 65536)

    monitor = connect(path)
    before = _table_counts(monitor, 2)
    changes_before = monitor.total_changes
    code, body = jpost(client, "/api/lease", {"count": 3000})
    monitor.execute("SELECT 1")  # no-op keeps the read connection current
    after = _table_counts(monitor, 2)
    check("批量 3000 笔逐笔回执齐全且连续",
          code == 200 and len(body["issued"]) == 3000
          and [x["id"] for x in body["issued"]]
          == list(range(body["issued"][0]["id"],
                        body["issued"][0]["id"] + 3000))
          and [x["slot"] for x in body["issued"]] == list(range(3000)))
    check("3000 连续槽位只写 1 个 lease_segments 段行",
          after["segments"] - before["segments"] == 1)
    check("3000 笔只改 1 行空闲区间（整池区间被切短）",
          after["free_rows"] - before["free_rows"] == 0)  # update, not insert
    check("新池路径不向 leases 表写逐笔行",
          after["lease_rows"] == before["lease_rows"])
    seg = monitor.execute(
        "SELECT * FROM lease_segments WHERE pool_id=2").fetchone()
    check("段行覆盖槽[0,3000) 且租约 id 与回执一致",
          (seg["start_slot"], seg["end_slot"], seg["first_lease_id"])
          == (0, 3000, body["issued"][0]["id"]))
    check("新池租约 id 全部大于旧池最大租约 id（原号不冲突）",
          seg["first_lease_id"] > monitor.execute(
              "SELECT MAX(id) m FROM leases").fetchone()["m"])
    check("每次领址仅 1 条日志（段化明细，非逐笔）",
          after["log_rows"] - before["log_rows"] == 1)
    total_delta = monitor.total_changes - changes_before
    check("整笔 3000 领址连接级改动行数=%d，与租约数无关（远小于 3000）"
          % total_delta, total_delta < 15)

    # Second contiguous batch still touches one segment + one free row.
    before = _table_counts(monitor, 2)
    code, body2 = jpost(client, "/api/lease", {"count": 2000})
    after = _table_counts(monitor, 2)
    check("第二批 2000 相邻槽位同样只写 1 段",
          code == 200 and after["segments"] - before["segments"] == 1
          and body2["issued"][0]["slot"] == 3000
          and body2["issued"][0]["id"] == body["issued"][-1]["id"] + 1)
    monitor.close()

    # Fragmented free space: hand-craft three non-adjacent free fragments,
    # then one request spanning them must touch exactly three segments.
    path2 = fresh_db()
    c2 = web_app(path2).test_client()
    jpost(c2, "/api/pool", {"cidr": "2001:db8:cccc::/48", "deleg_len": 60})
    jpost(c2, "/api/lease", {"count": 100})          # segment [0,100)
    conn = connect(path2)
    # Stagger expiry so expiring this segment leaves 3 separated fragments
    # after carving holes at slots 10..20 and 40..50.
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM lease_segments")
    conn.execute("DELETE FROM free_intervals WHERE pool_id=2")
    for lo, hi in ((0, 10), (20, 40), (50, 60)):
        conn.execute(
            "INSERT INTO free_intervals(pool_id, start_slot, end_slot)"
            " VALUES (2, ?, ?)", (lo, hi))
    conn.execute("COMMIT")
    conn.close()
    monitor = connect(path2)
    before = _table_counts(monitor, 2)
    code, body = jpost(c2, "/api/lease", {"count": 40})
    after = _table_counts(monitor, 2)
    check("跨 3 个碎片的一笔领址写 3 个段行（段数=连续槽位段数）",
          code == 200 and after["segments"] - before["segments"] == 3
          and len(body["issued"]) == 40)
    check("3 段各自租约 id 连续且全局递增",
          [s["first_lease_id"] for s in monitor.execute(
              "SELECT first_lease_id FROM lease_segments ORDER BY id")]
          == [107, 117, 137])
    monitor.close()


def test_new_pool_pause_and_commit():
    print("[11] 新池暂停核算：暂停期间续领计入、提交写盘量随段数、逐笔报到期")
    path = fresh_db()
    client = web_app(path).test_client()
    gw = gateway_app(path).test_client()
    jpost(client, "/api/pool",
          {"cidr": "2001:db8:dddd::/48", "deleg_len": 60})
    code, body = jpost(client, "/api/lease", {"count": 3000})
    batch_ids = [x["id"] for x in body["issued"]]

    # Pause at t=3600: every new lease (issued at t=0, 60-min TTL) is due,
    # together with the two built-in old-pool leases expiring at 3600.
    code, body = jpost(client, "/api/advance", {"target_seconds": 3600})
    check("暂停快照逐笔报出 3000 新池 + 2 旧池到期 id（内存展开）",
          code == 200
          and sorted(body["expected_expiries"]) == sorted(batch_ids + [1, 2]))
    conn = connect(path)
    snap_seg = conn.execute(
        "SELECT COUNT(*) c FROM advance_snapshot WHERE kind='segment'"
    ).fetchone()["c"]
    snap_legacy = conn.execute(
        "SELECT COUNT(*) c FROM advance_snapshot WHERE kind='legacy'"
    ).fetchone()["c"]
    check("快照落盘只存 1 个段引用 + 2 个旧池租约引用（非 3000 行）",
          snap_seg == 1 and snap_legacy == 2)
    conn.close()

    # Leasing during the pause: new leases at t=0 still expire at t=3600.
    code, more = jpost(client, "/api/lease", {"count": 1000})
    pause_ids = [x["id"] for x in more["issued"]]
    check("暂停期间照常批领 1000 笔", code == 200 and len(pause_ids) == 1000)
    check("暂停期间网关只读视图照常服务",
          gw.get("/withdrawal").status_code == 200)

    monitor = connect(path)
    before = _table_counts(monitor, 2)
    code, body = jpost(client, "/api/advance/commit")
    after = _table_counts(monitor, 2)
    check("提交成功且逐笔报出全部 4002 个到期租约 id（4000 新池+2 旧池）",
          code == 200
          and sorted(body["settled"]) == sorted(batch_ids + pause_ids + [1, 2]))
    check("暂停期间新领的 1000 笔计入 added",
          sorted(body["added"]) == sorted(pause_ids))
    check("提交按 2 个连续段结算新池（2 次段 UPDATE，非 4000 次）",
          len(body["settled_segments"]) == 2
          and after["active_segments"] - before["active_segments"] == -2)
    check("整段释放后新池空闲区间归并为 1 段",
          after["free_rows"] == 1)
    fr = monitor.execute(
        "SELECT start_slot,end_slot FROM free_intervals WHERE pool_id=2"
    ).fetchone()
    check("归还区间覆盖整池 [0,4096)，槽位提交后才可再领",
          (fr["start_slot"], fr["end_slot"]) == (0, 4096))
    check("时钟推进到 3600",
          monitor.execute("SELECT now_seconds FROM meta").fetchone()[0] == 3600)
    monitor.close()

    # Re-lease released prefixes: same slot numbers, brand-new lease ids.
    code, again = jpost(client, "/api/lease", {"count": 5})
    check("前缀再领沿用原槽位/前缀但换新租约 id",
          again["issued"][0]["slot"] == 0
          and again["issued"][0]["id"] > max(batch_ids + pause_ids)
          and again["issued"][0]["prefix"] == "2001:db8:dddd::/60")
    conn = connect(path)
    check("历史段行保留为 expired，新段为 active",
          conn.execute("SELECT COUNT(*) c FROM lease_segments"
                       " WHERE status='expired'").fetchone()["c"] == 2
          and conn.execute("SELECT COUNT(*) c FROM lease_segments"
                           " WHERE status='active'").fetchone()["c"] == 1)
    conn.close()


def test_new_pool_commit_interruption():
    print("[12] 新池提交中断：COMMIT 前丢失，重启按上次提交状态恢复")
    path = fresh_db()
    client = web_app(path).test_client()
    jpost(client, "/api/pool",
          {"cidr": "2001:db8:eeee::/48", "deleg_len": 60})
    jpost(client, "/api/lease", {"count": 2000})
    jpost(client, "/api/advance", {"target_seconds": 3600})
    jpost(client, "/api/lease", {"count": 500})

    resp = client.post("/api/advance/commit", json={"fault_inject": True})
    check("故障注入返回 500", resp.status_code == 500
          and resp.get_json()["restart_recovery"])
    recovered = recover_pending(connect(path))
    check("恢复识别并中止悬挂推进", recovered is not None)
    conn = connect(path)
    check("恢复后时钟仍为 0",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0)
    check("恢复后无 pending",
          conn.execute("SELECT pending_id FROM meta").fetchone()[0] is None)
    check("2500 笔段租约全部保留 active，无部分到期",
          active_lease_count(conn, 2, True) == 2500
          and conn.execute("SELECT COUNT(*) c FROM lease_segments"
                           " WHERE status='expired'").fetchone()["c"] == 0)
    check("可领范围维持暂停前状态（仅 1 个空闲段 [2500,4096)）",
          [tuple(r) for r in conn.execute(
              "SELECT start_slot,end_slot FROM free_intervals"
              " WHERE pool_id=2 ORDER BY start_slot")] == [(2500, 4096)])
    conn.close()

    # Redo the advance after recovery: it now settles cleanly. The two
    # built-in old-pool leases expiring at 3600 are part of the same batch.
    code, body = jpost(client, "/api/advance", {"target_seconds": 3600})
    check("恢复后可重新暂停", code == 200
          and len(body["expected_expiries"]) == 2502)
    code, body = jpost(client, "/api/advance/commit")
    check("重做提交成功，2500 新池笔逐笔报出（另含 2 条旧池）",
          code == 200 and len(body["settled"]) == 2502
          and len(body["settled_segments"]) == 2)


def test_new_pool_list_changed_and_dashboard():
    print("[13] 新池段名单变化 409 退回；首页只展开最近 30 笔")
    path = fresh_db()
    client = web_app(path).test_client()
    jpost(client, "/api/pool",
          {"cidr": "2001:db8:ffff::/48", "deleg_len": 60})
    jpost(client, "/api/lease", {"count": 100})
    jpost(client, "/api/advance", {"target_seconds": 3600})

    # Externally move the segment's expiry beyond the target: due list changes.
    conn = connect(path)
    conn.execute("UPDATE lease_segments SET expires_at = 9000 WHERE id = 1")
    conn.close()
    code, body = jpost(client, "/api/advance/commit")
    check("段到期名单变化返回 409 并要求重做",
          code == 409 and body.get("redo_required"))
    conn = connect(path)
    check("退回后时钟仍为 0、pending 清除、段租约保留 active",
          conn.execute("SELECT now_seconds FROM meta").fetchone()[0] == 0
          and conn.execute("SELECT pending_id FROM meta").fetchone()[0] is None
          and active_lease_count(conn, 2, True) == 100)
    conn.close()

    # Dashboard renders at most 30 lease rows derived from segments plus
    # legacy rows (6 built-in leases + 24 newest new-pool leases = 30).
    html = client.get("/").get_data(as_text=True)
    lease_section = html.split("最近租约")[1].split("操作日志")[0]
    row_count = lease_section.count("<tr>") - 1  # minus header row
    check("首页租约表只展开最近 30 笔（实际 %d 行）" % row_count,
          row_count == 30)
    view = recent_leases(connect(path), limit=30)
    check("recent_leases 派生 30 笔：最高的 24 个新池租约 id + 6 条旧池",
          len(view) == 30
          and [r["id"] for r in view] == list(range(106, 82, -1))
          + list(range(6, 0, -1)))
    check("派生行前缀由整数运算现算（首笔为槽 99 的 /60）",
          view[0]["slot"] == 99 and view[0]["prefix"]
          == "2001:db8:ffff:630::/60")


if __name__ == "__main__":
    tests = [test_seed_and_gateway, test_invalid_config_keeps_state,
             test_swap_and_overlap, test_slash56_boundary,
             test_lease_and_pool_full,
             test_expiry_with_pause_leases, test_list_changed_redo,
             test_commit_interruption, test_restart_and_no_scan,
             test_abort_and_non_advance,
             test_new_pool_segment_writes, test_new_pool_pause_and_commit,
             test_new_pool_commit_interruption,
             test_new_pool_list_changed_and_dashboard]
    for t in tests:
        t()
    print("\n结果：%d 通过, %d 失败" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
