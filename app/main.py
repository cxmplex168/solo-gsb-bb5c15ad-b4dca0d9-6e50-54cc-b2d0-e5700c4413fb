"""SRT 字幕时间校准工具后端。

核心逻辑：根据用户给出的若干「原字幕时刻 -> 视频实际时刻」锚点，
按原时刻排序后在相邻锚点间做线性插值，两端按最近一段的斜率外推，
统一换算每条字幕的起止时间，并检测负时间 / 起止倒置 / 与下一条重叠。
"""
import math
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI(title="SRT 字幕时间校准 / 片段剪辑 / 双语合成 / 阅读速度修复")

STATIC_DIR = Path(__file__).parent / "static"

TIME_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*$")
ARROW_RE = re.compile(r"-->")


# ---------------------------------------------------------------- 时间工具

def parse_time(s: str) -> int:
    """把 'HH:MM:SS,mmm' 解析为毫秒整数，失败抛 ValueError。"""
    m = TIME_RE.match(s or "")
    if not m:
        raise ValueError(f"无法解析时间: {s!r}（应为 HH:MM:SS,mmm）")
    h, mnt, sec, ms = (int(g) for g in m.groups())
    ms = int(str(m.group(4)).ljust(3, "0"))  # 允许 1~3 位毫秒
    return ((h * 60 + mnt) * 60 + sec) * 1000 + ms


def fmt_time(ms: int) -> str:
    """毫秒格式化为 SRT 时间；负数带负号以便预览中识别。"""
    sign = "-" if ms < 0 else ""
    ms = abs(int(round(ms)))
    h, ms = divmod(ms, 3600_000)
    mnt, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{mnt:02d}:{sec:02d},{ms:03d}"


# ---------------------------------------------------------------- SRT 解析

def parse_srt(text: str) -> List[dict]:
    """解析 SRT，保留序号与正文。返回 [{index, start, end, lines}]。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n﻿ ")
    if not text:
        raise ValueError("SRT 内容为空")
    entries = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.strip("\n").split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # 找到包含 --> 的时间行，其上一行（若有）为序号
        ti = next((i for i, ln in enumerate(lines) if ARROW_RE.search(ln)), None)
        if ti is None:
            raise ValueError(f"缺少时间行 '-->'：{lines[0][:40]!r}")
        parts = ARROW_RE.split(lines[ti])
        if len(parts) != 2:
            raise ValueError(f"时间行格式错误：{lines[ti]!r}")
        # 去掉时间行中可能附带的定位信息（如 'X1:.. X2:..'）
        start_s = parts[0].strip()
        end_s = parts[1].strip().split()[0] if parts[1].strip() else ""
        try:
            start, end = parse_time(start_s), parse_time(end_s)
        except ValueError as e:
            raise ValueError(f"第 {len(entries) + 1} 块：{e}")
        index = lines[0].strip() if ti > 0 else str(len(entries) + 1)
        entries.append({
            "index": index,
            "start": start,
            "end": end,
            "lines": lines[ti + 1:],
        })
    if not entries:
        raise ValueError("未解析到任何字幕块")
    return entries


# ---------------------------------------------------------------- 锚点与映射

def build_mapper(anchors_ms: List[tuple]):
    """anchors_ms: [(orig_ms, actual_ms)]，已排序且已校验。
    返回分段线性映射函数：相邻锚点线性插值，两端按最近一段斜率外推。"""
    pts = anchors_ms

    def map_t(t: float) -> float:
        if t <= pts[0][0]:
            (o0, a0), (o1, a1) = pts[0], pts[1]
        elif t >= pts[-1][0]:
            (o0, a0), (o1, a1) = pts[-2], pts[-1]
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= t <= pts[i + 1][0]:
                    (o0, a0), (o1, a1) = pts[i], pts[i + 1]
                    break
        slope = (a1 - a0) / (o1 - o0)
        return a0 + (t - o0) * slope

    return map_t


def validate_anchors(raw: List["Anchor"]) -> List[tuple]:
    """校验并排序锚点，返回 [(orig_ms, actual_ms)]。失败抛 ValueError。"""
    if len(raw) < 2:
        raise ValueError("至少需要 2 个锚点")
    pts = []
    for i, a in enumerate(raw, 1):
        try:
            pts.append((parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            raise ValueError(f"锚点 {i}：{e}")
    pts.sort(key=lambda p: p[0])
    for i in range(1, len(pts)):
        if pts[i][0] == pts[i - 1][0]:
            raise ValueError(f"锚点原时刻重复：{fmt_time(pts[i][0])}")
        if pts[i][1] <= pts[i - 1][1]:
            raise ValueError(
                f"映射后的实际时刻必须严格递增：{fmt_time(pts[i - 1][1])} 之后不能是 {fmt_time(pts[i][1])}"
            )
    return pts


def plan_anchor_suggestion(raw: List["Anchor"]) -> dict:
    """锚点校验失败时的异常定位：求需要剔除锚点最少的修复方案。

    按原时刻排序后，选出「原时刻、实际时刻都严格递增」的最长子序列（至少 2 点）。
    多解时决胜顺序：
      1) 最长（剔除行数最少）；
      2) 首尾原时刻（全局最小/最大原时刻）锚点的保留数量最多；
      3) 剔除项的原输入行号序列字典序最小。
    返回规划字典；无法凑出 2 点时 feasible=False（无法预演）。
    """
    n = len(raw)
    if n < 2:
        return {"feasible": False, "reason": "至少需要 2 个锚点，无法预演；请补充锚点后重试。"}

    parsed = []  # (输入行号, 原时刻ms, 实际时刻ms)
    bad = {}     # 无法解析的行号 -> 原因（必然要剔除）
    for i, a in enumerate(raw, 1):
        try:
            parsed.append((i, parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            bad[i] = str(e)
    if len(parsed) < 2:
        return {
            "feasible": False,
            "reason": "时间格式可解析的有效锚点不足 2 个，无法预演；请按 HH:MM:SS,mmm 修正时间格式。",
        }

    # 按原时刻排序（相同原时刻再按实际时刻、行号兜底，保证确定性）；
    # 转移时同时要求 orig、actual 严格小于，故重复原时刻不会串链。
    nodes = sorted(parsed, key=lambda p: (p[1], p[2], p[0]))
    m = len(nodes)
    row_orig = {r: o for r, o, _ in parsed}
    row_actual = {r: c for r, _, c in parsed}
    min_o = min(o for _, o, _ in parsed)
    max_o = max(o for _, o, _ in parsed)
    all_rows = set(range(1, n + 1))
    orig_count: dict = {}
    for _, o, _ in parsed:
        orig_count[o] = orig_count.get(o, 0) + 1

    def endpoint_hits(chain: tuple) -> int:
        """链中属于首/尾原时刻（全局最小、最大原时刻）的锚点数。"""
        return sum(1 for r in chain if row_orig[r] == min_o or row_orig[r] == max_o)

    def better(c1: tuple, c2: tuple) -> bool:
        """完整保留链的优劣比较：长度 > 首尾保留数 > 剔除行号序列字典序。"""
        if len(c1) != len(c2):
            return len(c1) > len(c2)
        e1, e2 = endpoint_hits(c1), endpoint_hits(c2)
        if e1 != e2:
            return e1 > e2
        rem1 = tuple(sorted(all_rows.difference(c1)))
        rem2 = tuple(sorted(all_rows.difference(c2)))
        return rem1 < rem2

    # O(m^2) DP：dp[i] 为以 nodes[i] 结尾的最优保留链（链元素为输入行号）
    dp = []
    for i in range(m):
        ri, oi, ai = nodes[i]
        best = (ri,)
        for j in range(i):
            rj, oj, aj = nodes[j]
            if oj < oi and aj < ai:
                cand = (*dp[j], ri)
                if better(cand, best):
                    best = cand
        dp.append(best)
    chain = dp[0]
    for c in dp[1:]:
        if better(c, chain):
            chain = c

    if len(chain) < 2:
        return {
            "feasible": False,
            "reason": "找不到两个原时刻与实际时刻都严格递增的锚点，无法预演；请手动修正锚点时刻。",
        }

    kept_rows = set(chain)
    removed = []
    for r in sorted(all_rows - kept_rows):
        if r in bad:
            reason = bad[r]
        elif orig_count[row_orig[r]] > 1:
            reason = f"原时刻 {fmt_time(row_orig[r])} 与其他锚点重复"
        else:
            reason = "实际时刻未随原时刻严格递增，剔除后保留锚点整体最优"
        removed.append({
            "row": r,
            "orig": raw[r - 1].orig,
            "actual": raw[r - 1].actual,
            "reason": reason,
        })
    kept = [
        {"row": r, "orig": fmt_time(row_orig[r]), "actual": fmt_time(row_actual[r])}
        for r in chain
    ]
    pts = [(row_orig[r], row_actual[r]) for r in chain]
    return {
        "feasible": True,
        "remove_rows": sorted(all_rows - kept_rows),
        "kept": kept,
        "removed": removed,
        "_pts": pts,
    }


def calibrate(entries: List[dict], map_t) -> List[dict]:
    """换算每条字幕起止时间并打问题标记。"""
    out = []
    for e in entries:
        ns, ne = int(round(map_t(e["start"]))), int(round(map_t(e["end"])))
        flags = []
        if ns < 0 or ne < 0:
            flags.append("负时间")
        if ne <= ns:
            flags.append("起止倒置")
        out.append({**e, "new_start": ns, "new_end": ne, "flags": flags})
    for i in range(len(out) - 1):
        if out[i]["new_end"] > out[i + 1]["new_start"]:
            out[i]["flags"].append("与下一条重叠")
    return out


# ---------------------------------------------------------------- API

class Anchor(BaseModel):
    orig: str    # 原字幕时刻 HH:MM:SS,mmm
    actual: str  # 视频实际时刻 HH:MM:SS,mmm


class CalibrateRequest(BaseModel):
    srt: str
    anchors: List[Anchor]


def _prepare(req: CalibrateRequest):
    """公共预处理，成功返回 (entries, calibrated)，失败抛 400。

    锚点校验类错误 detail 为 {"type": "anchor_invalid", "message": ...}，
    便于前端据此调用 /api/suggest 获取剔除建议；SRT 解析错误仍返回纯文本。
    """
    try:
        pts = validate_anchors(req.anchors)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail={"type": "anchor_invalid", "message": str(e)}
        )
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return calibrate(entries, build_mapper(pts))


def _preview_payload(rows: List[dict]) -> dict:
    return {
        "ok": all(not r["flags"] for r in rows),
        "problem_count": sum(1 for r in rows if r["flags"]),
        "entries": [
            {
                "index": r["index"],
                "old": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "new": f"{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}",
                "flags": r["flags"],
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/preview")
def preview(req: CalibrateRequest):
    rows = _prepare(req)
    return _preview_payload(rows)


@app.post("/api/suggest")
def suggest(req: CalibrateRequest):
    """锚点校验失败时的异常定位建议。

    返回需要剔除的最少锚点；确认前不改动任何输入，确认后由前端按原流程
    （/api/preview、/api/export）继续。无法凑出 2 个有效锚点时提示无法预演。
    """
    plan = plan_anchor_suggestion(req.anchors)
    if not plan["feasible"]:
        return {"feasible": False, "reason": plan["reason"]}

    pts = plan.pop("_pts")
    # 用保留锚点复用现有校准预览，展示受影响字幕；SRT 本身无法解析时仅提示
    try:
        rows = calibrate(parse_srt(req.srt), build_mapper(pts))
        plan["preview"] = _preview_payload(rows)
        plan["preview_error"] = None
    except ValueError as e:
        plan["preview"] = None
        plan["preview_error"] = str(e)
    return plan


# ---------------------------------------------------------------- 剪辑片段

class KeepInterval(BaseModel):
    start: str  # 原视频保留区间起点 HH:MM:SS,mmm
    end: str    # 原视频保留区间终点 HH:MM:SS,mmm


class ClipRequest(BaseModel):
    srt: str
    intervals: List[KeepInterval]


def validate_intervals(raw: List["KeepInterval"]) -> List[tuple]:
    """校验保留区间并按起点排序，返回 [(start_ms, end_ms, 输入行号)]。

    规则：起点须严格早于终点；区间为左闭右开，故相邻区间端点相触
    （前段终点 == 后段起点）不算重叠，严格交叉才报错。
    """
    rows = [r for r in raw if r.start.strip() or r.end.strip()]
    if not rows:
        raise ValueError("请至少填写 1 个保留区间")
    parsed = []
    for i, r in enumerate(rows, 1):
        try:
            s, e = parse_time(r.start), parse_time(r.end)
        except ValueError as ex:
            raise ValueError(f"保留区间 {i}：{ex}")
        if s >= e:
            raise ValueError(
                f"保留区间 {i}：起点 {fmt_time(s)} 须早于终点 {fmt_time(e)}"
            )
        parsed.append((s, e, i))
    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    # 按起点排序后，相邻区间若不重叠则任意区间都不可能重叠
    for k in range(1, len(parsed)):
        ps, pe, prow = parsed[k - 1]
        s, e, row = parsed[k]
        if s < pe:  # 左闭右开：s == pe 只是端点相触，允许
            raise ValueError(
                f"保留区间重叠：第 {prow} 行（{fmt_time(ps)} – {fmt_time(pe)}）"
                f"与第 {row} 行（{fmt_time(s)} – {fmt_time(e)}）"
            )
    return parsed


def clip_entries(entries: List[dict], intervals: List[tuple]):
    """按左闭右开区间裁剪字幕，返回 (新字幕列表, 保留区间总时长ms)。

    对每条字幕 [start, end) 与每个保留区间求交集：空交集丢弃；跨多个
    区间的字幕按区间拆成多条，正文原样保留。交集时间换算到拼接后的
    连续时间轴：新时间 = 交集时间 - 区间起点 + 此前所有区间时长之和。
    输出按新起点排序（起点相同保持原顺序）并从 1 连续重编号。
    """
    out = []
    offset = 0
    for s, e, _row in intervals:
        for order, entry in enumerate(entries):
            lo, hi = max(entry["start"], s), min(entry["end"], e)
            if lo < hi:  # 左闭右开：lo == hi 为空交集
                out.append({
                    "index": "",
                    "src_index": entry["index"],
                    "src_order": order,
                    "old_start": lo,
                    "old_end": hi,
                    "start": offset + lo - s,
                    "end": offset + hi - s,
                    "lines": entry["lines"],
                })
        offset += e - s
    out.sort(key=lambda r: (r["start"], r["src_order"]))
    for n, r in enumerate(out, 1):
        r["index"] = str(n)
    return out, offset


def _clip_prepare(req: ClipRequest):
    """剪辑功能公共预处理，成功返回 (entries, intervals)，失败抛 400。"""
    try:
        intervals = validate_intervals(req.intervals)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, intervals


@app.post("/api/clip/preview")
def clip_preview(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, total_ms = clip_entries(entries, intervals)
    return {
        "entry_count": len(rows),
        "interval_count": len(intervals),
        "total_duration": fmt_time(total_ms),
        "entries": [
            {
                "index": r["index"],
                "source": r["src_index"],
                "old": f"{fmt_time(r['old_start'])} --> {fmt_time(r['old_end'])}",
                "new": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.post("/api/clip/export")
def clip_export(req: ClipRequest):
    entries, intervals = _clip_prepare(req)
    rows, _ = clip_entries(entries, intervals)
    if not rows:
        raise HTTPException(
            status_code=400, detail="所有字幕均与保留区间无交集，没有可导出的内容"
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['start'])} --> {fmt_time(r['end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="clipped.srt"'},
    )


# ---------------------------------------------------------------- 双语合成

class MergeRequest(BaseModel):
    main: str         # 主轨 SRT
    translation: str  # 译文轨 SRT


MAX_MERGE_EDGES = 2_000_000  # 正时长交叠候选对上限（含残差边开销）


def _overlap_ms(a: dict, b: dict) -> int:
    """两条字幕的正时长交叠毫秒数；端点相触 / 不交叠返回 0。"""
    lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
    return hi - lo if hi > lo else 0


def _validate_track_duration(entries: List[dict], track: str) -> None:
    """任一轨出现空时长（end == start）或起止倒置（end < start）即报错。"""
    for i, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            kind = "空时长" if e["end"] == e["start"] else "起止时间倒置"
            raise ValueError(
                f"{track}第 {i} 条字幕{kind}："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )


def _track_is_ordered(entries: List[dict]) -> bool:
    """轨内条目是否按 (start, end) 均非降排列；正常 SRT 时间轴均满足。"""
    return all(
        entries[i]["start"] <= entries[i + 1]["start"]
        and entries[i]["end"] <= entries[i + 1]["end"]
        for i in range(len(entries) - 1)
    )


def _weight_params(m: int, n: int):
    """三级目标的整数权重参数（详见 pair_entries 文档）。

    每条译文在任一可行配对中至多使用一次（容量 1），故「已用译文序号集合」
    可用无进位的二进制位掩码表示：译文 j（0 起）贡献 2**(n-1-j)，最小序号
    为最高位。等大小集合的升序序号序列字典序越小，该掩码越大；掩码总和
    < 2**n。配对固定奖励取 2**n，2m 份之和仍小于一个交叠毫秒位，从而：
    交叠毫秒数 > 已配数（少漏配）> 序号序列字典序。
    """
    seq_bit = [1 << (n - 1 - j) for j in range(n)]
    shift_seq = n
    match_bonus = 1 << shift_seq
    shift_cnt = shift_seq + (2 * m).bit_length() + 1
    return seq_bit, match_bonus, shift_cnt


def _pair_ordered_dp(mains: List[dict], trans: List[dict]):
    """两轨均 (start,end) 非降时的保序最优配对（带状窗口，典型近线性）。

    f[i][j][k]：第 i 条主轨已配 k∈{0,1,2} 条译文、译文已消费到第 j 条时
    的最优权重。k==0 可跳过译文 j 或闭合当前主轨；k>0 可跳过译文或把译文 j
    配给当前主轨（仅正时长交叠）。两轨有序时交叠矩阵为 Monge 型，保序最优
    即全局最优。

    有序轨上每条主轨的正交叠译文下标是连续区间，且其左/右端点随主轨单调，
    故每行只需在 [L_i, R_i] 窗口内计算状态；轨内互不重叠（正常 SRT）时
    每个译文窗口只被常数个主轨覆盖，总状态 O(m+n)。轨内可有序重叠时窗口变
    宽，最差退化到 O(m*n)，仍正确。
    """
    import bisect

    m, n = len(mains), len(trans)
    seq_bit, bonus, shift_cnt = _weight_params(m, n)
    t_starts = [e["start"] for e in trans]
    t_ends = [e["end"] for e in trans]

    # 主轨 i 的正交叠译文下标区间 first_i .. last_i（TE、TS 均非降）
    first, last = [], []
    for me in mains:
        a = bisect.bisect_right(t_ends, me["start"])      # 第一个 end_j > start_i
        b = bisect.bisect_left(t_starts, me["end"]) - 1  # 最后一个 start_j < end_i
        first.append(min(a, n))
        last.append(b)  # first_i <= last_i 时才存在交叠译文

    # 每行窗口 [L_i, R_i]：左端承接下一主轨首个可能译文，右端覆盖到下一主轨左端
    left = [0] * (m + 1)
    right = [0] * (m + 1)
    for i in range(m):
        if i >= 1:
            left[i] = first[i]
        r = last[i] + 1  # 本行最后交叠译文的下一列（关闭/跳过传播所需）
        if i + 1 < m:
            r = max(r, first[i + 1])
        right[i] = min(r, n)
    left[m] = first[m - 1] if m else 0
    right[m] = n
    for i in range(1, m + 1):  # 窗口左端随主轨单调不减，钳制抖动
        if left[i] < left[i - 1]:
            left[i] = left[i - 1]
    for i in range(m + 1):
        if right[i] < left[i]:
            right[i] = left[i]

    neg = -(1 << 230)
    TAG_SKIP, TAG_MATCH, TAG_ROOT = 0, 1, 254
    TAG_CLOSE0, TAG_INVALID = 2, 255
    tags = {}

    def gix(i, j, k):
        return (i * (n + 1) + j) * 3 + k

    prev = None          # 上一行窗口内的状态 dict: (j,k) -> value
    p_lo = p_hi = 0
    for i in range(m + 1):
        lo, hi = left[i], right[i]
        cur = {}
        si = mains[i]["start"] if i < m else 0
        ei = mains[i]["end"] if i < m else 0

        def pget(j, k):
            if prev is None:
                return 0 if (j == 0 and k == 0) else neg
            if p_lo <= j <= p_hi:
                return prev.get((j, k), neg)
            return neg

        # 窗口首列 k=0：由上一行闭合而来（i==0 且 lo==0 为起点）
        if prev is None and lo == 0:
            cur[(lo, 0)] = 0
            tags[gix(i, lo, 0)] = TAG_ROOT
        else:
            best_v, best_k = neg, 0
            for kp in range(3):
                v = pget(lo, kp)
                if v > best_v:
                    best_v, best_k = v, kp
            cur[(lo, 0)] = best_v
            tags[gix(i, lo, 0)] = TAG_CLOSE0 + best_k
        cur.setdefault((lo, 1), neg)
        cur.setdefault((lo, 2), neg)

        for j in range(lo + 1, hi + 1):
            # k == 0：左邻跳过译文，或上一行在列 j 闭合当前主轨
            best, tag = cur.get((j - 1, 0), neg), TAG_SKIP
            for kp in range(3):
                v = pget(j, kp)
                if v > best:
                    best, tag = v, TAG_CLOSE0 + kp
            cur[(j, 0)] = best
            tags[gix(i, j, 0)] = TAG_INVALID if best == neg else tag

            wc = None
            if i < m:
                ts, te = t_starts[j - 1], t_ends[j - 1]
                xlo = si if si > ts else ts
                xhi = ei if ei < te else te
                if xhi > xlo:
                    wc = ((xhi - xlo) << shift_cnt) + bonus + seq_bit[j - 1]
            for k in (1, 2):
                best, tag = cur.get((j - 1, k), neg), TAG_SKIP
                pv = cur.get((j - 1, k - 1), neg)
                if wc is not None and pv > neg:
                    cand = pv + wc
                    if cand > best:
                        best, tag = cand, TAG_MATCH
                cur[(j, k)] = best
                tags[gix(i, j, k)] = TAG_INVALID if best == neg else tag
        prev, p_lo, p_hi = cur, lo, hi

    assignments = [[] for _ in range(m)]
    i, j, k = m, n, 0
    while i > 0 or j > 0 or k > 0:
        tag = tags.get(gix(i, j, k), TAG_INVALID)
        if k == 0:
            if tag == TAG_SKIP:
                j -= 1
            elif TAG_CLOSE0 <= tag <= TAG_CLOSE0 + 2:
                i -= 1
                k = tag - TAG_CLOSE0
            else:
                raise ValueError("配对回溯失败（内部错误）")
        elif tag == TAG_MATCH:
            assignments[i].append(j - 1)
            j -= 1
            k -= 1
        elif tag == TAG_SKIP:
            j -= 1
        else:
            raise ValueError("配对回溯失败（内部错误）")
    for lst in assignments:
        lst.sort()
    return assignments, {t for lst in assignments for t in lst}


def _pair_mincostflow(mains: List[dict], trans: List[dict]):
    """轨内时间回缩（不有序）时的回退算法：最小费用流求全局最优 b-匹配。

    源 -> 主轨容量 2、主轨 -> 译文容量 1（仅正时长交叠候选，费用为 -权重）、
    译文 -> 汇容量 1；势函数 + Dijkstra 逐次最短路，增广净增益 <= 0 时停止，
    故不强制配满。对任意（含交叉）配对均全局最优。
    """
    import heapq

    m, n = len(mains), len(trans)
    src, sink, nnode = 0, m + n + 1, m + n + 2
    graph = [[] for _ in range(nnode)]
    edge_u, edge_v, cap, edge_cost = [], [], [], []

    def add_edge(u, v, c, w):
        eid = len(edge_u)
        edge_u.extend((u, v)); edge_v.extend((v, u))
        cap.extend((c, 0)); edge_cost.extend((w, -w))
        graph[u].append(eid)
        graph[v].append(eid + 1)
        return eid

    for i in range(m):
        add_edge(src, 1 + i, 2, 0)
    for j in range(n):
        add_edge(1 + m + j, sink, 1, 0)

    seq_bit, bonus, shift_cnt = _weight_params(m, n)
    pair_edges = {}
    edge_count = m + n
    for i, me in enumerate(mains):
        for j, te in enumerate(trans):
            ov = _overlap_ms(me, te)
            if ov <= 0:
                continue
            edge_count += 1
            if edge_count > MAX_MERGE_EDGES:
                raise ValueError(
                    f"两轨正时长交叠候选对过多（已超过 {MAX_MERGE_EDGES}），"
                    "超出配对计算上限，请确认两轨为同一片源后裁短再试"
                )
            weight = (ov << shift_cnt) + bonus + seq_bit[j]
            eid = add_edge(1 + i, 1 + m + j, 1, -weight)
            pair_edges[eid] = (i, j)

    # 初始势函数：初始正向图是 DAG，按节点编号松弛一遍即得自源点最短距
    dist = [None] * nnode
    dist[src] = 0
    for u in range(nnode):
        if dist[u] is None:
            continue
        for eid in graph[u]:
            if cap[eid] and (dist[edge_v[eid]] is None
                             or dist[u] + edge_cost[eid] < dist[edge_v[eid]]):
                dist[edge_v[eid]] = dist[u] + edge_cost[eid]
    potential = [0 if d is None else d for d in dist]

    while True:
        dist = [None] * nnode
        prev_edge = [-1] * nnode
        dist[src] = 0
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for eid in graph[u]:
                if cap[eid] <= 0:
                    continue
                v = edge_v[eid]
                nd = d + edge_cost[eid] + potential[u] - potential[v]
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    prev_edge[v] = eid
                    heapq.heappush(heap, (nd, v))
        if dist[sink] is None:
            break
        for u in range(nnode):
            if dist[u] is not None:
                potential[u] += dist[u]
        if -potential[sink] <= 0:  # 再增广一条不改善任何一级目标
            break
        v = sink
        while v != src:
            eid = prev_edge[v]
            cap[eid] -= 1
            cap[eid ^ 1] += 1
            v = edge_u[eid]

    assignments = [[] for _ in range(m)]
    matched = set()
    for eid, (i, j) in pair_edges.items():
        if cap[eid] == 0:
            assignments[i].append(j)
            matched.add(j)
    for lst in assignments:
        lst.sort()
    return assignments, matched


def pair_entries(mains: List[dict], trans: List[dict]):
    """全局最优配对（不要求轨内条目互不重叠）。

    每条主轨可配 0~2 条译文、每条译文至多使用一次，仅正时长交叠可配对。
    三级目标依次为：最大化总交叠毫秒数；并列时最大化已配译文数（少漏配）；
    再并列时使已用译文原序号（升序）序列字典序最小。

    三级目标编码为单个整数边权（见 _weight_params）：
        交叠毫秒数 << shift_cnt | 每条配对固定奖励 | 译文序号加权奖励。

    - 两轨均按 (start, end) 非降时（正常 SRT 时间轴），交叠矩阵为 Monge 型，
      保序网格 DP 即全局最优，走 O(m*n) 的 _pair_ordered_dp；
    - 轨内时间回缩（条目相互包含 / 错位）时回退到最小费用流
      _pair_mincostflow，对含交叉的配对仍保证全局最优。

    返回 (assignments, matched_set)：assignments[i] 为配给第 i 条主轨的
    译文下标（升序）。
    """
    if _track_is_ordered(mains) and _track_is_ordered(trans):
        return _pair_ordered_dp(mains, trans)
    return _pair_mincostflow(mains, trans)


def _merge_prepare(req: "MergeRequest"):
    """双语合成公共预处理：解析两轨并校验空时长 / 倒置，失败抛 400。"""
    try:
        mains = parse_srt(req.main)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"主轨解析失败：{e}")
    try:
        trans = parse_srt(req.translation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"译文轨解析失败：{e}")
    try:
        _validate_track_duration(mains, "主轨")
        _validate_track_duration(trans, "译文轨")
        assignments, matched = pair_entries(mains, trans)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return mains, trans, assignments, matched


def _entry_brief(seq: int, e: dict):
    return {
        "seq": seq,
        "index": e["index"],
        "time": f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}",
        "text": "\n".join(e["lines"]),
    }


@app.post("/api/merge/preview")
def merge_preview(req: MergeRequest):
    mains, trans, assignments, matched = _merge_prepare(req)
    total_overlap = 0
    entries = []
    for i, me in enumerate(mains):
        paired = []
        for jt in assignments[i]:
            ov = _overlap_ms(me, trans[jt])
            total_overlap += ov
            brief = _entry_brief(jt + 1, trans[jt])
            brief["overlap_ms"] = ov
            paired.append(brief)
        item = _entry_brief(i + 1, me)
        item["translations"] = paired
        entries.append(item)
    unmatched = [_entry_brief(j + 1, trans[j]) for j in range(len(trans)) if j not in matched]
    return {
        "main_count": len(mains),
        "trans_count": len(trans),
        "matched_count": len(matched),
        "unmatched_count": len(trans) - len(matched),
        "total_overlap_ms": total_overlap,
        "total_overlap": fmt_time(total_overlap),
        "entries": entries,
        "unmatched": unmatched,
    }


@app.post("/api/merge/export")
def merge_export(req: MergeRequest):
    mains, trans, assignments, _matched = _merge_prepare(req)
    blocks = []
    for i, me in enumerate(mains):
        body = list(me["lines"])
        for jt in assignments[i]:
            body.extend(trans[jt]["lines"])
        if not body:  # 两轨正文均空时仍保留该主轨块
            body = [""]
        blocks.append(
            f"{i + 1}\n{fmt_time(me['start'])} --> {fmt_time(me['end'])}\n"
            + "\n".join(body)
        )
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="bilingual.srt"'},
    )


@app.post("/api/export")
def export(req: CalibrateRequest):
    rows = _prepare(req)
    bad = [r for r in rows if r["flags"]]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"存在 {len(bad)} 条问题字幕（负时间/起止倒置/重叠），已禁止导出",
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="calibrated.srt"'},
    )


# ---------------------------------------------------------------- 阅读速度修复

MERGE_GAP_MS = 500  # 允许合并的相邻字幕最大间隔（毫秒）


class SpeedRequest(BaseModel):
    srt: str
    max_chars_per_sec: float  # 最大每秒字数（非空白字符）
    max_block_sec: float      # 单块最长秒数


def _speed_params(req: "SpeedRequest"):
    """校验两个限制参数均为正数（有限），返回 (max_cps, max_sec)。"""
    for name, v in (("最大每秒字数", req.max_chars_per_sec),
                    ("单块最长秒数", req.max_block_sec)):
        if v is None or math.isnan(v) or math.isinf(v) or v <= 0:
            raise ValueError(f"{name}必须为正数")
    return float(req.max_chars_per_sec), float(req.max_block_sec)


def _count_nonspace(lines: List[str]) -> int:
    """非空白字符数：空格、制表符、换行等空白均不计。"""
    return sum(1 for ln in lines for ch in ln if not ch.isspace())


def _validate_speed_track(entries: List[dict]) -> None:
    """空时长 / 起止倒置 / 轨内重叠（含时间乱序）即抛 ValueError。"""
    for i, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            kind = "空时长" if e["end"] == e["start"] else "起止倒置"
            raise ValueError(
                f"第 {i} 条字幕（序号 {e['index']}）{kind}："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )
    for i in range(len(entries) - 1):
        cur, nxt = entries[i], entries[i + 1]
        if cur["end"] > nxt["start"]:
            raise ValueError(
                f"轨内重叠：第 {i + 1} 条（序号 {cur['index']}）终点 "
                f"{fmt_time(cur['end'])} 晚于第 {i + 2} 条（序号 {nxt['index']}）"
                f"起点 {fmt_time(nxt['start'])}"
            )


def plan_speed_fix(entries: List[dict], max_cps: float, max_sec: float):
    """阅读速度修复的全局分块规划。

    只允许把「间隔 ≤ 500ms 的连续字幕」合并成一块：合并块取首条起点、末条
    终点，正文按原顺序换行拼接。每块（含未合并单条）须同时满足
    速度（非空白字符数 / 起止跨度秒数）≤ max_cps 与时长 ≤ max_sec。

    目标：合并次数最少（即块数最多）；仍并列时切分位置序列字典序最小
    （切分点尽量靠前，等价于每块尽量短）。后缀 DP：choice[i] 取使
    1 + blocks[j+1] 最大的最小 j——同块数下首切分点最小即保证全局切分
    序列字典序最小（最优子结构：经 j 的方案以后缀最优解 blocks[j+1] 为上界）。

    返回 (plan, reason)：plan 为 [(首条下标, 末条下标), ...]；无可行方案时
    plan 为 None，reason 为无法修复的定位说明。
    """
    n = len(entries)
    chars = [_count_nonspace(e["lines"]) for e in entries]
    csum = [0] * (n + 1)
    for i, c in enumerate(chars):
        csum[i + 1] = csum[i] + c
    # gap_ok[k]：第 k 条与第 k+1 条间隔 ≤ 500ms（无重叠已校验，间隔 ≥ 0）
    gap_ok = [entries[k + 1]["start"] - entries[k]["end"] <= MERGE_GAP_MS
              for k in range(n - 1)]
    # maxreach[i]：以第 i 条为首、满足间隔约束的块最远末条下标（随 i 单调不减）
    maxreach = [0] * n
    r = 0
    for i in range(n):
        if r < i:
            r = i
        while r + 1 < n and gap_ok[r]:
            r += 1
        maxreach[i] = r
    max_dur_ms = max_sec * 1000

    def block_ok(i: int, j: int) -> bool:
        dur_ms = entries[j]["end"] - entries[i]["start"]
        if dur_ms > max_dur_ms:
            return False
        return (csum[j + 1] - csum[i]) * 1000 <= max_cps * dur_ms

    blocks = [0] * (n + 1)  # blocks[i]：后缀 i..n-1 的最多块数，-1 表不可行
    choice = [-1] * n
    for i in range(n - 1, -1, -1):
        best, bj = -1, -1
        for j in range(i, maxreach[i] + 1):
            # 块时长随 j 单调增，超限后不必再试更长的块
            if entries[j]["end"] - entries[i]["start"] > max_dur_ms:
                break
            if blocks[j + 1] < 0 or not block_ok(i, j):
                continue
            cand = 1 + blocks[j + 1]
            if cand > best:  # j 递增遍历，严格更大才更新 ⇒ 并列取最小 j
                best, bj = cand, j
        blocks[i] = best
        choice[i] = bj

    if blocks[0] >= 0:
        plan = []
        i = 0
        while i < n:
            j = choice[i]
            plan.append((i, j))
            i = j + 1
        return plan, None

    # 无法修复：前缀可达分析，定位第一个任何可行分块都覆盖不到的字幕
    reach = [False] * (n + 1)  # reach[p]：前 p 条（0..p-1）可被完整分块
    reach[0] = True
    for p in range(1, n + 1):
        for i in range(p - 1, -1, -1):
            if maxreach[i] < p - 1:
                break  # maxreach 随 i 单调不减，更小的 i 也够不到 p-1
            if reach[i] and block_ok(i, p - 1):
                reach[p] = True
                break
    stuck = max(p for p in range(n + 1) if reach[p])
    # reach[stuck] 可达而 reach[stuck+1] 不可达 ⇒ 该条单块必然已超限
    e = entries[stuck]
    dur_ms = e["end"] - e["start"]
    c = chars[stuck]
    pos = (f"第 {stuck + 1} 条字幕（序号 {e['index']}，"
           f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}）")
    if dur_ms > max_dur_ms:
        reason = (f"{pos}单条时长 {dur_ms / 1000:.3f} 秒已超过单块最长 "
                  f"{max_sec} 秒，合并只会更长，无法修复")
    elif c * 1000 > max_cps * dur_ms:
        left = stuck > 0 and gap_ok[stuck - 1]
        right = stuck < n - 1 and gap_ok[stuck]
        if not (left or right):
            reason = (f"{pos}速度 {c * 1000 / dur_ms:.2f} 字/秒超过 "
                      f"{max_cps} 字/秒，且与前后字幕间隔均超过 500 毫秒，"
                      "无法合并降速，无法修复")
        else:
            reason = (f"{pos}速度 {c * 1000 / dur_ms:.2f} 字/秒超过 "
                      f"{max_cps} 字/秒，与相邻字幕合并后仍无法同时满足"
                      "速度与时长限制，无法修复")
    else:
        reason = f"{pos}附近无法得到满足速度与时长限制的分块方案，无法修复"
    return None, reason


def _speed_prepare(req: "SpeedRequest"):
    """阅读速度修复公共预处理，返回 (entries, max_cps, max_sec, plan, reason)。"""
    try:
        max_cps, max_sec = _speed_params(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    plan, reason = plan_speed_fix(entries, max_cps, max_sec)
    return entries, max_cps, max_sec, plan, reason


def _speed_blocks(entries: List[dict], plan: List[tuple]) -> List[dict]:
    """按分块方案汇总每块的来源序号、时间、速度、合并正文。"""
    out = []
    for no, (i, j) in enumerate(plan, 1):
        grp = entries[i:j + 1]
        start, end = grp[0]["start"], grp[-1]["end"]
        dur_ms = end - start
        c = sum(_count_nonspace(e["lines"]) for e in grp)
        lines = [ln for e in grp for ln in e["lines"]]
        out.append({
            "index": no,
            "sources": [e["index"] for e in grp],
            "start": fmt_time(start),
            "end": fmt_time(end),
            "duration_sec": round(dur_ms / 1000, 3),
            "chars": c,
            "speed": round(c * 1000 / dur_ms, 2),
            "merged": j > i,
            "lines": lines,
            "text": "\n".join(lines),
        })
    return out


@app.post("/api/speed/preview")
def speed_preview(req: SpeedRequest):
    entries, max_cps, max_sec, plan, reason = _speed_prepare(req)
    if plan is None:
        return {"feasible": False, "reason": reason}
    blocks = _speed_blocks(entries, plan)
    return {
        "feasible": True,
        "entry_count": len(entries),
        "block_count": len(blocks),
        "merge_count": len(entries) - len(blocks),
        "max_chars_per_sec": max_cps,
        "max_block_sec": max_sec,
        "blocks": [{k: v for k, v in b.items() if k != "lines"} for b in blocks],
    }


@app.post("/api/speed/export")
def speed_export(req: SpeedRequest):
    entries, _cps, _sec, plan, reason = _speed_prepare(req)
    if plan is None:
        raise HTTPException(status_code=400, detail=f"无法修复：{reason}")
    blocks = _speed_blocks(entries, plan)
    content = "\n\n".join(
        f"{b['index']}\n{b['start']} --> {b['end']}\n" + "\n".join(b["lines"])
        for b in blocks
    ) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="speed_fixed.srt"'},
    )
