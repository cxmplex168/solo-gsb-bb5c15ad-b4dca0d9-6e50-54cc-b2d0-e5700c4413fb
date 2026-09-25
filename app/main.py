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

class SpeedRequest(BaseModel):
    srt: str
    max_cps: float           # 最大每秒字数（非空白字符 / 起止跨度秒数）
    max_block_seconds: float # 单块最长秒数


MAX_MERGE_GAP_MS = 500  # 仅允许合并间隔不超过 500ms 的相邻字幕


def _non_ws_chars(lines: List[str]) -> int:
    """正文中的非空白字符数（每个非空白字符计一字）。"""
    return sum(1 for ln in lines for ch in ln if not ch.isspace())


def _speed_params(req: "SpeedRequest"):
    """校验限速参数（均须为正数），返回 (max_cps, max_block_ms)。失败抛 ValueError。"""
    if not math.isfinite(req.max_cps) or req.max_cps <= 0:
        raise ValueError("最大每秒字数须为正数")
    if not math.isfinite(req.max_block_seconds) or req.max_block_seconds <= 0:
        raise ValueError("单块最长秒数须为正数")
    return req.max_cps, req.max_block_seconds * 1000.0


def _validate_speed_track(entries: List[dict]) -> None:
    """空时长（终点=起点）、起止倒置（终点<起点）或轨内重叠即报错。

    逐条保证终点严格大于起点后，再检查相邻两条：前条终点 > 后条起点即重叠。
    相邻不重叠时起点、终点均严格递增，故任意两条都不可能重叠。
    """
    for i, e in enumerate(entries, 1):
        if e["end"] <= e["start"]:
            kind = "空时长" if e["end"] == e["start"] else "起止时间倒置"
            raise ValueError(
                f"第 {i} 条字幕（序号 {e['index']}）{kind}："
                f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}"
            )
    for i in range(len(entries) - 1):
        a, b = entries[i], entries[i + 1]
        if a["end"] > b["start"]:
            raise ValueError(
                f"第 {i + 1} 条（{fmt_time(a['start'])} --> {fmt_time(a['end'])}）"
                f"与第 {i + 2} 条（{fmt_time(b['start'])} --> {fmt_time(b['end'])}）轨内重叠"
            )


def plan_speed_fix(entries: List[dict], max_cps: float, max_block_ms: float):
    """阅读速度修复的全局分块规划。

    把按时间有序的字幕划分为若干连续块，规则：
      - 仅相邻间隔 ≤ 500ms 的字幕可并入同一块（间隔更大处必须切开）；
      - 每块取首条起点、末条终点为起止，正文按原顺序换行拼接；
      - 每块 速度 = 非空白字数 / 起止跨度 ≤ max_cps，且跨度 ≤ max_block_ms。

    目标：合并次数（= 条数 - 块数）最少优先；仍并列时切分位置序列
    （相邻两条之间的切开位置，按条目序号升序）字典序最小。

    后缀 DP：dp[i] = (合并次数, 切分位置序列) 为第 i 条起的最优后缀方案，
    choice[i] 记录首块末条下标。约束逐块独立、后缀最优子结构成立；
    终点严格递增，故跨度随块尾单调，超时长即可停止扩展。
    返回 (blocks, None)；无可行方案时返回 (None, 原因)。
    """
    n = len(entries)
    prefix = [0] * (n + 1)  # 非空白字数前缀和
    for i, e in enumerate(entries):
        prefix[i + 1] = prefix[i] + _non_ws_chars(e["lines"])

    # reach[i]：从第 i 条起允许合并到的最远下标（块内相邻间隔均 ≤ 500ms）
    reach = list(range(n))
    for i in range(n - 2, -1, -1):
        gap = entries[i + 1]["start"] - entries[i]["end"]
        reach[i] = reach[i + 1] if gap <= MAX_MERGE_GAP_MS else i

    dp = [None] * (n + 1)
    dp[n] = (0, ())
    choice = [None] * n
    for i in range(n - 1, -1, -1):
        best = None
        start_i = entries[i]["start"]
        for j in range(i, reach[i] + 1):
            span = entries[j]["end"] - start_i
            if span > max_block_ms:
                break  # 终点严格递增，跨度只会更大
            if dp[j + 1] is None:
                continue
            chars = prefix[j + 1] - prefix[i]
            if chars * 1000.0 / span > max_cps:
                continue
            merges = (j - i) + dp[j + 1][0]
            cuts = ((j + 1,) + dp[j + 1][1]) if j + 1 < n else ()
            if best is None or (merges, cuts) < (best[0], best[1]):
                best = (merges, cuts, j)
        if best is not None:
            dp[i] = best[:2]
            choice[i] = best[2]

    if dp[0] is None:
        return None, _speed_infeasible_reason(entries, dp, reach, max_cps, max_block_ms)

    blocks = []
    i = 0
    while i < n:
        j = choice[i]
        merged = entries[i:j + 1]
        blocks.append({
            "sources": [e["index"] for e in merged],
            "src_first": i + 1,
            "src_last": j + 1,
            "start": entries[i]["start"],
            "end": entries[j]["end"],
            "chars": prefix[j + 1] - prefix[i],
            "lines": [ln for e in merged for ln in e["lines"]],
        })
        i = j + 1
    return blocks, None


def _speed_infeasible_reason(entries: List[dict], dp: list, reach: list,
                             max_cps: float, max_block_ms: float) -> str:
    """定位最靠后的不可行起点，给出无法修复的具体原因。

    bad 之后的位置均可行（bad 是最大的不可行下标），因此从 bad 出发的
    每个候选块若满足速度与时长，dp[bad] 就不会不可行——即候选块必然全部
    超速（时长、间隔已由枚举范围保证），或单条自身即超时长。
    """
    n = len(entries)
    bad = max(i for i in range(n) if dp[i] is None)
    e = entries[bad]
    head = (f"第 {bad + 1} 条字幕（序号 {e['index']}，"
            f"{fmt_time(e['start'])} --> {fmt_time(e['end'])}）")
    own_ms = e["end"] - e["start"]
    if own_ms > max_block_ms:
        return (
            f"{head}自身时长 {own_ms / 1000:.3f} 秒已超过单块最长 "
            f"{max_block_ms / 1000:g} 秒，合并只会更长，无法修复"
        )
    chars = 0
    best_cps = None
    for j in range(bad, reach[bad] + 1):
        chars += _non_ws_chars(entries[j]["lines"])
        span = entries[j]["end"] - e["start"]
        if span > max_block_ms:
            break
        cps = chars * 1000.0 / span
        if best_cps is None or cps < best_cps:
            best_cps = cps
    if best_cps is None:  # 防御：候选为空时（单条超时长已在上面拦截）
        return f"{head}附近不存在可行分块，无法修复"
    return (
        f"{head}起即使尽量向后合并（间隔 ≤ {MAX_MERGE_GAP_MS}ms、单块 ≤ "
        f"{max_block_ms / 1000:g} 秒），速度最低仍为 {best_cps:.2f} 字/秒，"
        f"超过上限 {max_cps:g} 字/秒，无法修复"
    )


def _speed_prepare(req: "SpeedRequest"):
    """阅读速度修复公共预处理：校验参数、解析并校验轨道，失败抛 400。"""
    try:
        max_cps, max_block_ms = _speed_params(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        entries = parse_srt(req.srt)
        _validate_speed_track(entries)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return entries, max_cps, max_block_ms


def _speed_block_payload(b: dict, seq: int) -> dict:
    span = b["end"] - b["start"]
    return {
        "seq": seq,
        "sources": b["sources"],
        "merged": b["src_first"] != b["src_last"],
        "start": fmt_time(b["start"]),
        "end": fmt_time(b["end"]),
        "duration_ms": span,
        "chars": b["chars"],
        "cps": round(b["chars"] * 1000.0 / span, 2),
        "text": "\n".join(b["lines"]),
    }


@app.post("/api/speed/preview")
def speed_preview(req: SpeedRequest):
    entries, max_cps, max_block_ms = _speed_prepare(req)
    blocks, reason = plan_speed_fix(entries, max_cps, max_block_ms)
    if blocks is None:
        return {"feasible": False, "reason": reason, "entry_count": len(entries)}
    return {
        "feasible": True,
        "entry_count": len(entries),
        "block_count": len(blocks),
        "merge_count": len(entries) - len(blocks),
        "max_cps": max_cps,
        "max_block_seconds": req.max_block_seconds,
        "blocks": [_speed_block_payload(b, seq) for seq, b in enumerate(blocks, 1)],
    }


@app.post("/api/speed/export")
def speed_export(req: SpeedRequest):
    entries, max_cps, max_block_ms = _speed_prepare(req)
    blocks, reason = plan_speed_fix(entries, max_cps, max_block_ms)
    if blocks is None:
        raise HTTPException(status_code=400, detail=f"无法修复：{reason}")
    content = "\n\n".join(
        f"{seq}\n{fmt_time(b['start'])} --> {fmt_time(b['end'])}\n"
        + "\n".join(b["lines"] or [""])
        for seq, b in enumerate(blocks, 1)
    ) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="speed_fixed.srt"'},
    )
