"""
server.py
=========
本文件是笔记本端车道感知与控制桥接服务，主链路为：
1) 收图（socket）-> 2) UNet 分割 -> 3) 车道跟踪 -> 4) 输出 deviation/severity。

给新同学的阅读顺序建议：
1. 先看 `process_lane_lock_on`（单帧控制主流程）
2. 再看 `run_server` 与 `run_client_session`（网络/推理/可视化）
3. 最后看配置区（每个参数都在注释中给出了调参方向）
"""
import socket
import struct
import cv2
import numpy as np
import torch
import time
from PIL import Image
import os
import warnings

# 兼容不同NumPy版本的RankWarning导入路径
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning

warnings.simplefilter('ignore', RankWarning)

try:
    from Unet import UNet
    from data import transform
except ImportError:
    print("Error: Ensure Unet.py and data.py are in the current directory.")
    exit()

# --- Configuration ---
# 总说明：本文件负责「图像分割 -> 车道跟踪 -> 偏差/转弯强度输出」。
# 处理目标是让小车稳定居中，同时在直道与急弯切换时兼顾平滑与响应。

# [TUNING] GEOMETRY
# 几何参数：主要控制“单边线推中线”时的基准偏置。
# A moderate lane width is more stable for very tight O-turns and roundabouts.
LANE_WIDTH_PIXELS = 165
# Additional center bias (pixels)
BIAS_OFFSET = 0

# Smoothing
# 平滑参数：值越大越稳，值越小响应越快。
HISTORY_WEIGHT = 0.55
GHOST_MODE_LIMIT = 10

# Masking
# ROI掩码：只保留与车辆近场控制最相关的区域，抑制远处噪声。
MASK_BOTTOM_ROW = 244
TOP_MASK_ROW = 35
# 形态学闭运算核：复用避免每帧重复分配。
POST_CLOSE_KERNEL = np.ones((3, 3), np.uint8)
ZONE_START = 120
ZONE_END = 236

# Tracking
# 跟踪参数：通过多条水平切片逐层追踪已锁定边线。
SLICE_Y_LEVELS = [230, 216, 202, 188, 174, 160, 146]
SLICE_HALF_HEIGHT = 8
BASE_SEARCH_MARGIN = 75
MAX_SEARCH_MARGIN = 135
# SEARCH 状态下重新锁边的最小点数确认。
# 只靠一个点就承诺锁到某一侧，极容易在急弯/盲弯里被错误大连通域带偏。
SEARCH_RELOCK_MIN_POINTS = 2
SIDE_SWITCH_MARGIN = 46
SIDE_SWITCH_CONFIRM = 3
# 左右锁定切换至少需要这么多层有效点，避免只靠 1~2 个噪点就翻边。
SIDE_SWITCH_MIN_POINTS = 3
MIN_BLOB_AREA = 35.0
MAX_POINT_JUMP = 44.0

# Sharp-turn handling
# 急弯策略：根据曲率选择不同lookahead并调节控制灵敏度。
LOOKAHEAD_Y_STRAIGHT = 198
LOOKAHEAD_Y_TURN = 184
LOOKAHEAD_Y_SHARP = 168
LOOKAHEAD_Y_UTURN = 154
CURVE_TURN_THRESHOLD = 18.0
CURVE_SHARP_THRESHOLD = 36.0
UTURN_CURVE_THRESHOLD = 44.0
MAX_TARGET_STEP = 26.0
HEADING_GAIN = 18.0
MAX_HEADING_SLOPE = 1.6
UTURN_TURN_IND_THRESHOLD = 0.78
UTURN_HEADING_THRESHOLD = 0.95
UTURN_HOLD_FRAMES = 10
UTURN_HEADING_GAIN_SCALE = 1.28
UTURN_HEADING_LIMIT = 3.3
UTURN_TARGET_STEP_LIMIT = 56.0
UTURN_HISTORY_MAX = 0.30
UTURN_SEVERITY_BOOST = 0.18

# Adaptive transition handling (straight -> sharp turn)
# 过渡策略：直道进入急弯时，动态放宽限制，避免“该转时转不进去”。
TURN_INDICATOR_DECAY = 0.84
# 弱观测（仅 0~1 个点）时，转弯记忆衰减放慢，避免刚进盲弯就把 Turn 清零。
TURN_INDICATOR_DECAY_WEAK = 0.94
TURN_ENTRY_RATE_SCALE = 14.0
TURN_SPAN_SCALE = 40.0
HEADING_LIMIT_SHARP = 2.7
MAX_TARGET_STEP_SHARP = 44.0
HISTORY_WEIGHT_SHARP = 0.40

# Curve-fit robustness
# 稳健拟合：离群点过滤 + 加权拟合 + 拟合质量门控。
FIT_INLIER_THRESHOLD = 22.0
FIT_MIN_INLIER_RATIO = 0.50
FIT_MAX_RMSE = 11.0
FIT_HISTORY_WEIGHT = 0.60
SINGLE_POINT_HISTORY_BLEND = 0.62
SINGLE_POINT_HISTORY_TURN_BLEND = 0.82
SINGLE_POINT_HISTORY_TURN_IND = 0.35

# Fit coherence (拟合一致性)
# 检测新旧拟合的几何偏差，自动调节"巡线 vs 曲线跟踪"的比重。
# divergence < LOW → coherence ≈ 1（信任历史，追踪连续曲线）
# divergence > HIGH → coherence ≈ 0（信任当前观测，直接巡线）
COHERENCE_DIVERGE_LOW = 15.0
COHERENCE_DIVERGE_HIGH = 40.0
COHERENCE_SMOOTH = 0.60
COHERENCE_FIT_HIST_MIN = 0.05
COHERENCE_TARGET_HIST_MIN = 0.15

# Anti lane-hugging
# 防骑线：优先融合双边线中心，并强制保持离边安全距离。
LANE_WIDTH_MIN = 135.0
LANE_WIDTH_MAX = 210.0
LANE_WIDTH_SMOOTH = 0.72
OPP_MIN_GAP = 86.0
OPP_SEARCH_WIDTH = 172.0
OPP_MIN_HITS = 2
MIN_EDGE_CLEARANCE = 24.0
DUAL_CENTER_BLEND = 0.88
DUAL_HEADING_SCALE = 0.72
TURN_HUG_EXTRA_CLEARANCE = 12.0
TURN_HUG_CORR_GAIN = 0.72
TURN_HUG_CORR_LIMIT = 14.0
# 单边线主动反向偏置：
# 当只看到一根线时，不直接把“半个车道宽”当作中心，
# 而是再额外往另一侧偏一点，主动给车辆留出更多搜索空间。
SINGLE_LINE_OPPOSITE_BIAS_BASE = 16.0
SINGLE_LINE_OPPOSITE_BIAS_TURN = 12.0
SINGLE_LINE_OPPOSITE_BIAS_MAX = 32.0
DUAL_TRACK_MIN_HITS = 3
DUAL_TRACK_MIN_GAP = 64.0
DUAL_TRACK_MAX_GAP = 240.0

# Junction / ramp dashed-line handling
MIN_BLOB_AREA_DASH = 12.0
DASH_MIN_PIXELS = 16
RAMP_TURN_THRESHOLD = 0.55
JUNCTION_AMBIG_SCORE_GAP = 10.0
JUNCTION_AMBIG_X_GAP = 24.0
JUNCTION_AMBIG_REQUIRED = 2
JUNCTION_COOLDOWN_FRAMES = 4
JUNCTION_JUMP_TIGHTEN = 0.78

# Sharp fork / blind-corner recovery
# 当当前锁定边线在急弯分岔处变得不可靠时：
# 1) 优先根据远场哪一侧“连续白色像素更强”来给目标一个方向偏置；
# 2) 如果前方几乎失去可追踪信息，则输出更激进的恢复指令。
FAR_FIELD_END_Y = 182
BRANCH_RELOCK_TURN_IND = 0.72
BRANCH_RELOCK_MAX_POINTS = 4
BRANCH_RELOCK_STRENGTH_RATIO = 1.35
BRANCH_RELOCK_MIN_PIXELS = 120
BRANCH_BIAS_PIXELS = 34.0
BRANCH_HISTORY_MAX = 0.34
BRANCH_STEP_LIMIT = 54.0
# 强弯错边强制重锁：
# 有些场景下虽然采样点不算少，但其实整条边已经锁错了。
# 当 turn/curve 都很高、对侧线缺失、远场另一侧明显更强、且几何方向也支持时，
# 允许直接把锁定边切回正确一侧。
BRANCH_FORCE_RELOCK_TURN_IND = 0.92
BRANCH_FORCE_RELOCK_CURVE = 52.0
BRANCH_FORCE_RELOCK_RATIO = 1.18
BRANCH_FORCE_RELOCK_HEADING = 0.10
BRANCH_FORCE_RELOCK_SLOPE = 0.14
RECOVERY_TURN_IND = 0.90
RECOVERY_MAX_POINTS = 1
RECOVERY_MIN_FAR_PIXELS = 70
RECOVERY_HOLD_FRAMES = 6
RECOVERY_TARGET_BIAS = 84.0
RECOVERY_STEP_LIMIT = 74.0
RECOVERY_HISTORY_MAX = 0.18
RECOVERY_SENTINEL_DEVIATION = 175.0


class LaneTracker:
    """
    车道跟踪状态机（核心）：
    - SEARCH：未锁定边线，等待初始化。
    - LOCKED_LEFT / LOCKED_RIGHT：锁定单边线并推算车道中心。

    新同学建议先读 `process_lane_lock_on`，再回看各子步骤。
    """
    # 车道跟踪状态机：
    # 1) SEARCH：丢线或初始化，先抓到一条可靠边线
    # 2) LOCKED_LEFT/RIGHT：持续跟踪某一侧边线，并推算中心目标
    def __init__(self):
        # 上一帧目标中心x（用于平滑和步长限制）
        self.last_target = 128.0
        self.mode = "SEARCH"  # SEARCH, LOCKED_LEFT, LOCKED_RIGHT
        self.tracked_x = 128.0  # The x-position of the line we are tracking
        self.lost_counter = 0
        self.side_switch_counter = 0
        # 最近一次稳定锁定的边线侧。用于急弯/盲弯中短时丢线后的“按上一侧优先重锁”。
        self.preferred_mode = None
        # 上一帧拟合曲线系数，用于弱观测时回退
        self.last_fit_coef = None
        # 上一帧中心线拟合系数。新的主控制逻辑优先跟踪“中心线”而不是“某一侧边线”。
        self.last_center_coef = None
        self.fit_fail_counter = 0
        self.prev_curve_score = 0.0
        # 入弯意图强度（0~1）：用于直道->急弯过渡
        self.turn_indicator = 0.0
        # 动态估计车道宽，防止固定宽度导致长期贴线
        self.estimated_lane_width = float(LANE_WIDTH_PIXELS)
        self.junction_cooldown = 0
        # U-turn 保护冷却：短时保持更激进转向，避免冲出弯道
        self.uturn_cooldown = 0
        # 极端急弯/盲拐角恢复：在短时丢失前方可跟踪信息时，保持同一转向方向。
        self.recovery_cooldown = 0
        self.recovery_direction = 1.0
        # 拟合一致性（0~1）：新旧拟合几何偏差越大 → 越低 → 越信任当前观测（巡线模式）。
        self.fit_coherence = 1.0
        self.vis_data = {}
    def get_biggest_blob_centroid(self, img_slice):
        """
        在给定切片中寻找最大白色连通域的中心。

        返回：
        - (centroid_x, area)
        - 若未找到有效连通域，则返回 (None, 0)。
        """
        contours, _ = cv2.findContours(img_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0

        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)

        if area < 50:
            return None, 0

        M = cv2.moments(biggest)
        if M["m00"] == 0:
            return None, 0

        cx = int(M["m10"] / M["m00"])
        return cx, area

    def get_blob_candidates(self, img_slice, min_area=MIN_BLOB_AREA):
        """
        提取切片中的候选连通域中心点，并按面积从大到小排序。

        返回：[(cx, area), ...]
        """
        contours, _ = cv2.findContours(img_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            candidates.append((cx, float(area)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    @staticmethod
    def _fit_poly(ys, xs, weights=None):
        """
        拟合 x=f(y)。

        说明：
        - 点数 >= 4 时使用二次拟合；
        - 点数较少时退化为一次拟合，并补成 [0, b, c] 的形式，
          便于后续统一按二次多项式求值。
        """
        deg = 2 if len(ys) >= 4 else 1
        coef = np.polyfit(ys, xs, deg, w=weights)
        if deg == 1:
            coef = np.array([0.0, coef[0], coef[1]], dtype=np.float32)
        else:
            coef = np.array(coef, dtype=np.float32)
        return coef

    @staticmethod
    def _nearest_point_x(points, y_target, default_x):
        """返回离 y_target 最近采样点的 x；若无点则回退 default_x。"""
        if not points:
            return float(default_x)
        nearest = min(points, key=lambda p: abs(float(p[1]) - float(y_target)))
        return float(nearest[0])

    @staticmethod
    def _dynamic_control_params(turn_indicator):
        """
        根据 turn_indicator 动态计算控制参数。

        turn_indicator 越高表示弯道意图越强：
        - 允许更大的目标步长（更敢转）
        - 降低历史平滑权重（更灵活）
        - 提高航向修正增益（更快贴合弯道）
        """
        heading_limit = MAX_HEADING_SLOPE + ((HEADING_LIMIT_SHARP - MAX_HEADING_SLOPE) * turn_indicator)
        step_limit = MAX_TARGET_STEP + ((MAX_TARGET_STEP_SHARP - MAX_TARGET_STEP) * turn_indicator)
        history_weight = HISTORY_WEIGHT - ((HISTORY_WEIGHT - HISTORY_WEIGHT_SHARP) * turn_indicator)
        heading_gain = HEADING_GAIN * (0.90 + (0.35 * turn_indicator))
        return heading_limit, step_limit, history_weight, heading_gain

    def _far_field_balance(self, binary_warped, center_screen):
        """
        统计远场区域左右两半的有效像素数量。

        用途：
        - 分岔时判断哪一侧仍然保留更连续的“前方路线证据”；
        - 前方几乎变成水平线或完全丢失时，辅助触发恢复逻辑。
        """
        far_roi = binary_warped[ZONE_START:FAR_FIELD_END_Y, :]
        left_strength = int(np.count_nonzero(far_roi[:, :center_screen]))
        right_strength = int(np.count_nonzero(far_roi[:, center_screen:]))
        return left_strength, right_strength

    def _dominant_half_blob_x(self, binary_warped, center_screen, prefer_left):
        """
        在整块跟踪 ROI 的半幅区域内，寻找最大的候选连通域中心。

        返回全局 x 坐标；若没有可靠候选则返回 None。
        """
        roi = binary_warped[ZONE_START:ZONE_END, :]
        if prefer_left:
            half = roi[:, :center_screen]
            offset = 0
        else:
            half = roi[:, center_screen:]
            offset = center_screen

        candidates = self.get_blob_candidates(half, min_area=MIN_BLOB_AREA_DASH)
        if not candidates:
            return None
        return float(offset + candidates[0][0])

    def _pick_half_candidate(self, candidates, offset, prev_x=None):
        """
        从半幅候选中挑一个最可信的点。

        - 若存在上一层位置 prev_x，则优先连续性；
        - 否则优先面积最大的候选。
        """
        if not candidates:
            return None

        if prev_x is None:
            return float(offset + candidates[0][0])

        best_x = None
        best_score = -1e9
        for cx_local, area in candidates[:6]:
            x_global = float(offset + cx_local)
            score = (0.045 * area) - (1.25 * abs(x_global - prev_x))
            if score > best_score:
                best_score = score
                best_x = x_global
        return best_x

    def find_dual_lines_symmetric(self, binary_warped, y_levels, center_screen, lookahead_y):
        """
        对称地同时搜索左右两根边线。

        这条路径不依赖当前 `LOCKED_LEFT/RIGHT` 状态，适合在单边锁错时
        重新用“双边证据”把目标中心拉回车道中间。

        返回：
        - left_x, right_x: lookahead 处左右边线位置
        - measured_width: 估计车道宽
        - left_hits, right_hits: 左右命中层数
        - left_points, right_points: 左右采样点
        """
        h, _ = binary_warped.shape
        left_points = []
        right_points = []
        left_prev = None
        right_prev = None
        width_samples = []

        for y_center in y_levels:
            if y_center < ZONE_START or y_center >= ZONE_END:
                continue

            y0 = max(0, int(y_center - SLICE_HALF_HEIGHT))
            y1 = min(h, int(y_center + SLICE_HALF_HEIGHT))
            if y1 <= y0:
                continue

            row = binary_warped[y0:y1, :]
            left_half = row[:, :center_screen]
            right_half = row[:, center_screen:]

            left_candidates = self.get_blob_candidates(left_half, min_area=MIN_BLOB_AREA_DASH)
            right_candidates = self.get_blob_candidates(right_half, min_area=MIN_BLOB_AREA_DASH)

            left_x = self._pick_half_candidate(left_candidates, 0.0, prev_x=left_prev)
            right_x = self._pick_half_candidate(right_candidates, float(center_screen), prev_x=right_prev)

            if left_x is None or right_x is None:
                continue

            lane_gap = float(right_x - left_x)
            if lane_gap < DUAL_TRACK_MIN_GAP or lane_gap > DUAL_TRACK_MAX_GAP:
                continue

            left_points.append((left_x, float(y_center)))
            right_points.append((right_x, float(y_center)))
            width_samples.append(lane_gap)
            left_prev = left_x
            right_prev = right_x

        if len(left_points) < 2 or len(right_points) < 2:
            return None, None, None, len(left_points), len(right_points), left_points, right_points

        left_ys = np.array([p[1] for p in left_points], dtype=np.float32)
        left_xs = np.array([p[0] for p in left_points], dtype=np.float32)
        right_ys = np.array([p[1] for p in right_points], dtype=np.float32)
        right_xs = np.array([p[0] for p in right_points], dtype=np.float32)

        left_coef = self._fit_poly(left_ys, left_xs, weights=None)
        right_coef = self._fit_poly(right_ys, right_xs, weights=None)
        left_x = float(np.polyval(left_coef, lookahead_y))
        right_x = float(np.polyval(right_coef, lookahead_y))
        measured_width = float(np.median(width_samples)) if width_samples else None
        return left_x, right_x, measured_width, len(left_points), len(right_points), left_points, right_points

    def _turn_direction_from_geometry(self, heading, slope_hint):
        """
        将几何信息映射为恢复时的转向方向。

        返回：
        - +1.0 : 期望右转（最终会输出正 deviation）
        - -1.0 : 期望左转
        """
        if abs(heading) > 0.08:
            return 1.0 if heading < 0.0 else -1.0
        if abs(slope_hint) > 0.05:
            return 1.0 if slope_hint < 0.0 else -1.0
        return self.recovery_direction

    def find_opposite_line(self, binary_warped, mode, y_levels, ref_points, ref_coef, lookahead_y):
        """
        从当前锁定边线出发，搜索对侧边线。

        返回：
        - opp_line_x：lookahead 处对侧边线 x
        - measured_width：估计车道宽（可能为 None）
        - hit_count：命中层数
        - opp_points：对侧候选点
        """
        if mode not in ("LOCKED_LEFT", "LOCKED_RIGHT"):
            return None, None, 0, []

        h, w = binary_warped.shape
        expected_w = float(np.clip(self.estimated_lane_width, LANE_WIDTH_MIN, LANE_WIDTH_MAX))
        sign = 1.0 if mode == "LOCKED_LEFT" else -1.0
        opp_points = []

        for y_center in y_levels:
            if y_center < ZONE_START or y_center >= ZONE_END:
                continue

            y0 = max(0, int(y_center - SLICE_HALF_HEIGHT))
            y1 = min(h, int(y_center + SLICE_HALF_HEIGHT))
            if y1 <= y0:
                continue

            if ref_coef is not None:
                ref_x = float(np.polyval(ref_coef, y_center))
            else:
                ref_x = self._nearest_point_x(ref_points, y_center, self.tracked_x)

            exp_x = ref_x + (sign * expected_w)
            x_min = int(max(0, exp_x - (OPP_SEARCH_WIDTH * 0.5)))
            x_max = int(min(w, exp_x + (OPP_SEARCH_WIDTH * 0.5)))

            if mode == "LOCKED_LEFT":
                x_min = max(x_min, int(ref_x + OPP_MIN_GAP))
            else:
                x_max = min(x_max, int(ref_x - OPP_MIN_GAP))

            if x_max <= x_min:
                continue

            row = binary_warped[y0:y1, :]
            window = row[:, x_min:x_max]
            candidates = self.get_blob_candidates(window)
            if not candidates:
                continue

            best_x = None
            best_score = -1e9
            for cx_local, area in candidates[:5]:
                x_global = float(x_min + cx_local)
                lane_gap = abs(x_global - ref_x)
                if lane_gap < OPP_MIN_GAP:
                    continue

                dx_exp = abs(x_global - exp_x)
                score = (-1.35 * dx_exp) + (0.03 * area)
                if score > best_score:
                    best_score = score
                    best_x = x_global

            if best_x is not None:
                opp_points.append((best_x, float(y_center)))

        if not opp_points:
            return None, None, 0, []

        ys = np.array([p[1] for p in opp_points], dtype=np.float32)
        xs = np.array([p[0] for p in opp_points], dtype=np.float32)
        opp_coef = self._fit_poly(ys, xs, weights=None)
        opp_line_x = float(np.polyval(opp_coef, lookahead_y))

        widths = []
        for ox, oy in opp_points:
            if ref_coef is not None:
                rx = float(np.polyval(ref_coef, oy))
            else:
                rx = self._nearest_point_x(ref_points, oy, self.tracked_x)
            gap = abs(float(ox) - float(rx))
            if (LANE_WIDTH_MIN * 0.7) <= gap <= (LANE_WIDTH_MAX * 1.3):
                widths.append(gap)

        measured_width = float(np.median(widths)) if widths else None
        return opp_line_x, measured_width, len(opp_points), opp_points

    def robust_curve_fit(self, points):
        """
        稳健曲线拟合（离群点过滤 + 近场加权）。

        返回：
        (coef, curve_score, rmse, inlier_ratio, fit_ok, near_far_span, slope_hint)
        """
        if len(points) < 2:
            return None, 0.0, 0.0, 0.0, False, 0.0, 0.0

        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)

        coef0 = self._fit_poly(ys, xs, weights=None)
        pred0 = np.polyval(coef0, ys)
        residual = np.abs(xs - pred0)
        inlier_mask = residual <= FIT_INLIER_THRESHOLD

        if np.count_nonzero(inlier_mask) >= 3:
            ys_fit = ys[inlier_mask]
            xs_fit = xs[inlier_mask]
        else:
            ys_fit = ys
            xs_fit = xs

        y_min = float(np.min(ys_fit))
        y_max = float(np.max(ys_fit))
        y_span = max(1.0, y_max - y_min)
        # Near-field points (larger y) are slightly more important for control.
        weights = 0.65 + 0.35 * ((ys_fit - y_min) / y_span)
        coef = self._fit_poly(ys_fit, xs_fit, weights=weights)

        pred = np.polyval(coef, ys_fit)
        rmse = float(np.sqrt(np.mean(np.square(xs_fit - pred)))) if len(xs_fit) > 0 else 0.0
        inlier_ratio = float(np.count_nonzero(inlier_mask)) / float(max(1, len(xs)))

        near_far_span = abs(float(xs[0] - xs[-1]))
        spread = float(np.max(xs) - np.min(xs))
        curve_score = max(near_far_span, spread * 0.8)

        denom = max(1.0, float(abs(ys[0] - ys[-1])))
        slope_hint = float((xs[0] - xs[-1]) / denom)

        fit_ok = (len(ys_fit) >= 3) and (rmse <= FIT_MAX_RMSE) and (inlier_ratio >= FIT_MIN_INLIER_RATIO)
        return coef, curve_score, rmse, inlier_ratio, fit_ok, near_far_span, slope_hint

    def _resolve_track_from_points(self, points, history_coef, lookahead_y, heading_limit, prev_turn_indicator):
        """
        将“当前采样点 + 历史拟合”融合成一条可直接用于控制的轨迹。

        这个辅助函数既可以处理：
        - 单边边线轨迹
        - 双边中点构成的中心线轨迹

        返回 dict，字段包括：
        - x / heading：lookahead 处的横向位置与切线
        - coef：最终使用的拟合系数
        - fit_source：new / weak / history / point / point_hist / none
        - curve_score / near_far_span / slope_hint：给转弯判断使用
        """
        result = {
            'x': None,
            'heading': 0.0,
            'coef': None,
            'fit_source': "none",
            'curve_score': 0.0,
            'rmse': 0.0,
            'inlier_ratio': 0.0,
            'fit_ok': False,
            'near_far_span': 0.0,
            'slope_hint': 0.0,
            'points': list(points),
        }

        if len(points) >= 2:
            coef, curve_score, rmse, inlier_ratio, fit_ok, near_far_span, slope_hint = self.robust_curve_fit(points)

            if coef is not None and fit_ok and history_coef is not None:
                scaled_fit_hw = COHERENCE_FIT_HIST_MIN + (
                    (FIT_HISTORY_WEIGHT - COHERENCE_FIT_HIST_MIN) * self.fit_coherence)
                coef = (history_coef * scaled_fit_hw) + (coef * (1.0 - scaled_fit_hw))

            if coef is not None and (not fit_ok) and history_coef is not None:
                coef = history_coef.copy()
                fit_source = "history"
            elif coef is not None:
                fit_source = "new" if fit_ok else "weak"
            else:
                fit_source = "point"

            if coef is not None:
                x = float(np.polyval(coef, lookahead_y))
                heading = float((2.0 * coef[0] * lookahead_y) + coef[1])
                heading = float(np.clip(heading, -heading_limit, heading_limit))
                if fit_source == "weak":
                    turn_hint = max(self.turn_indicator, prev_turn_indicator)
                    heading *= (0.55 + (0.25 * turn_hint))

                result.update({
                    'x': x,
                    'heading': heading,
                    'coef': coef,
                    'fit_source': fit_source,
                    'curve_score': float(curve_score),
                    'rmse': float(rmse),
                    'inlier_ratio': float(inlier_ratio),
                    'fit_ok': bool(fit_ok),
                    'near_far_span': float(near_far_span),
                    'slope_hint': float(slope_hint),
                })
                return result

        if len(points) == 1:
            x = float(points[0][0])
            fit_source = "point"
            curve_score = 0.0
            slope_hint = 0.0
            heading = 0.0

            if history_coef is not None:
                history_turn = max(self.turn_indicator, prev_turn_indicator)
                history_x = float(np.polyval(history_coef, lookahead_y))
                history_heading = float((2.0 * history_coef[0] * lookahead_y) + history_coef[1])
                history_heading = float(np.clip(history_heading, -heading_limit, heading_limit))

                y_near = float(SLICE_Y_LEVELS[0])
                y_far = float(SLICE_Y_LEVELS[-1])
                hist_near = float(np.polyval(history_coef, y_near))
                hist_far = float(np.polyval(history_coef, y_far))
                denom = max(1.0, abs(y_near - y_far))
                slope_hint = float((hist_near - hist_far) / denom)
                curve_score = abs(hist_near - hist_far) * 0.9

                hist_blend = SINGLE_POINT_HISTORY_TURN_BLEND if history_turn > SINGLE_POINT_HISTORY_TURN_IND else SINGLE_POINT_HISTORY_BLEND
                hist_blend *= self.fit_coherence
                x = (x * (1.0 - hist_blend)) + (history_x * hist_blend)
                heading = history_heading
                fit_source = "point_hist"

            result.update({
                'x': x,
                'heading': heading,
                'coef': history_coef.copy() if history_coef is not None else None,
                'fit_source': fit_source,
                'curve_score': float(curve_score),
                'rmse': 0.0,
                'inlier_ratio': 0.0,
                'fit_ok': False,
                'near_far_span': float(curve_score),
                'slope_hint': float(slope_hint),
            })
            return result

        if history_coef is not None:
            x = float(np.polyval(history_coef, lookahead_y))
            heading = float((2.0 * history_coef[0] * lookahead_y) + history_coef[1])
            heading = float(np.clip(heading, -heading_limit, heading_limit))
            result.update({
                'x': x,
                'heading': heading,
                'coef': history_coef.copy(),
                'fit_source': "history",
            })
            return result

        return result

    def process_lane_lock_on(self, binary_warped):
        """
        单帧控制主流程（最关键入口）。

        输入：
        - binary_warped: 车道分割二值图（0/255）。

        输出：
        - deviation: 目标中心相对图像中心的偏差（像素）。
        - turn_severity: 转弯/风险强度（0~1），供客户端降速。
        """
        # 主流程：
        # A. 单边线跟踪取点 -> B. 稳健拟合 -> C. 推算目标中心 -> D. 平滑与限幅 -> E. 输出偏差/强度
        h, w = binary_warped.shape
        center_screen = w // 2

        # 1. 近场ROI：聚焦控制关键区域，减少远处干扰
        roi = binary_warped[ZONE_START:ZONE_END, :]
        points = []  # (x, y)
        uturn_active = self.uturn_cooldown > 0
        prev_turn_indicator = float(self.turn_indicator)
        weak_turn_indicator_floor = 0.0
        search_lock_source = None

        # 2. 分层切片跟踪：按“预测位置 + 候选评分”选每层最可信点
        if self.mode == "SEARCH":
            # 急弯/盲弯中如果刚刚短时丢线，优先沿上一条稳定锁定边重锁，
            # 避免被外侧那条“面积更大但路线不对”的边线瞬间抢走。
            prefer_memory_relock = (
                self.preferred_mode in ("LOCKED_LEFT", "LOCKED_RIGHT")
                and (
                    self.turn_indicator > RAMP_TURN_THRESHOLD
                    or self.prev_curve_score > (CURVE_TURN_THRESHOLD * 0.8)
                    or self.uturn_cooldown > 0
                )
            )
            if prefer_memory_relock:
                relock_x = self._dominant_half_blob_x(
                    binary_warped,
                    center_screen,
                    prefer_left=(self.preferred_mode == "LOCKED_LEFT")
                )
                if relock_x is not None:
                    self.tracked_x = float(relock_x)
                    self.mode = self.preferred_mode
                    self.lost_counter = 0
                    search_lock_source = "preferred"

            if self.mode == "SEARCH":
                cx, area = self.get_biggest_blob_centroid(roi)
                if cx is not None:
                    self.tracked_x = float(cx)
                    self.mode = "LOCKED_LEFT" if cx < center_screen else "LOCKED_RIGHT"
                    self.lost_counter = 0
                    search_lock_source = "blob"

        if self.mode != "SEARCH":
            current_x = float(self.tracked_x)
            junction_anchor = self.junction_cooldown > 0
            junction_ambiguous_slices = 0
            ramp_mode = ((self.turn_indicator > RAMP_TURN_THRESHOLD) or (self.prev_curve_score > (CURVE_TURN_THRESHOLD * 0.9))) and (not junction_anchor)

            for y_center in SLICE_Y_LEVELS:
                if y_center < ZONE_START or y_center >= ZONE_END:
                    continue

                y0 = max(0, y_center - SLICE_HALF_HEIGHT)
                y1 = min(h, y_center + SLICE_HALF_HEIGHT)
                if y1 <= y0:
                    continue

                row = binary_warped[y0:y1, :]
                margin = int(BASE_SEARCH_MARGIN + min(25.0, abs(current_x - center_screen) * 0.25))
                margin = min(margin, MAX_SEARCH_MARGIN)
                x_min = max(0, int(current_x - margin))
                x_max = min(w, int(current_x + margin))
                if x_max <= x_min:
                    continue

                window = row[:, x_min:x_max]
                adaptive_min_area = MIN_BLOB_AREA_DASH if (ramp_mode and junction_ambiguous_slices == 0) else MIN_BLOB_AREA
                candidates = self.get_blob_candidates(window, min_area=adaptive_min_area)

                predicted_x = current_x
                if self.last_fit_coef is not None:
                    history_x = float(np.polyval(self.last_fit_coef, y_center))
                    if junction_anchor:
                        predicted_x = history_x
                    else:
                        predicted_x = (0.65 * current_x) + (0.35 * history_x)
                predicted_x = float(np.clip(predicted_x, 0.0, float(w - 1)))

                # 在匝道虚线场景中，连通域可能很小，允许用窗口内非零点重心补点。
                if (not candidates) and ramp_mode and junction_ambiguous_slices == 0:
                    nz = np.nonzero(window)
                    if len(nz[0]) >= DASH_MIN_PIXELS:
                        cx_local = float(np.mean(nz[1]))
                        candidates = [(cx_local, float(len(nz[0])))]

                if not candidates:
                    continue


                scored = []
                for cx_local, area in candidates[:6]:
                    x_global = float(x_min + cx_local)
                    dx = abs(x_global - predicted_x)
                    score = (-1.8 * dx) + (0.035 * area)

                    # 侧边先验：抑制跟踪点跨到另一侧车道线。
                    side_gate = 12 if junction_anchor else 28
                    if self.mode == "LOCKED_LEFT" and x_global > center_screen + side_gate:
                        score -= (x_global - (center_screen + side_gate)) * (1.15 if junction_anchor else 0.9)
                    elif self.mode == "LOCKED_RIGHT" and x_global < center_screen - side_gate:
                        score -= ((center_screen - side_gate) - x_global) * (1.15 if junction_anchor else 0.9)

                    if junction_anchor:
                        score -= 0.25 * dx

                    scored.append((score, x_global, area))

                if not scored:
                    continue

                scored.sort(key=lambda item: item[0], reverse=True)
                best_score, best_x, _ = scored[0]

                if len(scored) >= 2:
                    score_gap = scored[0][0] - scored[1][0]
                    x_gap = abs(scored[0][1] - scored[1][1])
                    if score_gap < JUNCTION_AMBIG_SCORE_GAP and x_gap > JUNCTION_AMBIG_X_GAP:
                        junction_ambiguous_slices += 1
                        if junction_anchor and self.last_fit_coef is not None:
                            history_x = float(np.polyval(self.last_fit_coef, y_center))
                            best_x = min([scored[i][1] for i in range(min(3, len(scored)))], key=lambda x: abs(x - history_x))

                jump_limit = MAX_POINT_JUMP + min(14.0, abs(predicted_x - center_screen) * 0.2)
                if junction_anchor:
                    jump_limit *= JUNCTION_JUMP_TIGHTEN
                if abs(best_x - predicted_x) > jump_limit:
                    continue

                points.append((best_x, float(y_center)))
                if junction_anchor:
                    current_x = (current_x * 0.75) + (best_x * 0.25)
                else:
                    current_x = best_x

            if points:
                self.tracked_x = float(points[0][0])  # nearest point
                self.lost_counter = 0
                if self.mode in ("LOCKED_LEFT", "LOCKED_RIGHT") and len(points) >= 3:
                    self.preferred_mode = self.mode
                # 若本帧是靠“最大连通域”从 SEARCH 新锁到一侧，但后续只能拿到 0~1 个点，
                # 说明这次锁边证据太弱，很可能就是你图里那种被外侧大边线误导的情况。
                # 此时不接受这次新锁定，回退到 SEARCH，宁可短时 ghost，也不直接承诺错误方向。
                if search_lock_source == "blob" and len(points) < SEARCH_RELOCK_MIN_POINTS:
                    self.mode = "SEARCH"
                    self.side_switch_counter = 0
                    points = []
                # 若当前只有极少点、却突然锁到了“与上一条稳定边相反”的那一侧，
                # 也直接判定这次锁边不可靠。
                if (
                    len(points) <= 1
                    and self.preferred_mode in ("LOCKED_LEFT", "LOCKED_RIGHT")
                    and self.mode in ("LOCKED_LEFT", "LOCKED_RIGHT")
                    and self.mode != self.preferred_mode
                ):
                    self.mode = "SEARCH"
                    self.side_switch_counter = 0
                    points = []
                # 只剩 0~1 个点时，说明已经进入弱观测阶段。
                # 这时不要立刻把转弯意图“忘干净”，否则后续分岔/恢复/重锁逻辑都会失效，
                # 很容易退化成局部单点跟踪并错误锁到外侧边线。
                if len(points) <= 1 and prev_turn_indicator > 0.0:
                    weak_turn_indicator_floor = max(
                        weak_turn_indicator_floor,
                        prev_turn_indicator * TURN_INDICATOR_DECAY_WEAK
                    )
            else:
                self.lost_counter += 1
                if self.lost_counter > GHOST_MODE_LIMIT:
                    self.mode = "SEARCH"
                    # 上一帧拟合曲线系数，用于弱观测时回退
                    self.last_fit_coef = None
                    self.last_center_coef = None

            junction_detected = junction_ambiguous_slices >= JUNCTION_AMBIG_REQUIRED
            if junction_detected:
                self.junction_cooldown = JUNCTION_COOLDOWN_FRAMES
            else:
                self.junction_cooldown = max(0, self.junction_cooldown - 1)
            junction_active = junction_anchor or junction_detected
        else:
            junction_active = self.junction_cooldown > 0
            self.junction_cooldown = max(0, self.junction_cooldown - 1)

        # 3. 连续跨中心线确认后才切换左右锁定，避免单帧误判
        if junction_active:
            # 交叉口歧义阶段：冻结左右切换，防止被隔壁车道线误导。
            self.side_switch_counter = 0
        else:
            if len(points) < SIDE_SWITCH_MIN_POINTS:
                self.side_switch_counter = 0
            elif self.mode == "LOCKED_LEFT" and self.tracked_x > center_screen + SIDE_SWITCH_MARGIN:
                self.side_switch_counter += 1
            elif self.mode == "LOCKED_RIGHT" and self.tracked_x < center_screen - SIDE_SWITCH_MARGIN:
                self.side_switch_counter += 1
            else:
                self.side_switch_counter = 0

            if self.side_switch_counter >= SIDE_SWITCH_CONFIRM:
                self.mode = "LOCKED_RIGHT" if self.mode == "LOCKED_LEFT" else "LOCKED_LEFT"
                self.side_switch_counter = 0

        # 4. 几何估计：得到参考边线位置 line_x、切线 heading、曲率评分 curve_score
        line_x = float(self.tracked_x)
        heading = 0.0
        curve_score = 0.0
        fit_rmse = 0.0
        fit_inlier_ratio = 0.0
        fit_source = "none"
        fit_coef_to_draw = None
        lookahead_y = LOOKAHEAD_Y_TURN
        slope_hint = 0.0

        # 对侧边线/防骑线状态缓存
        opp_line_x = None
        opp_hits = 0
        opp_points = []
        dual_ok = False
        dual_left_x = None
        dual_right_x = None
        dual_left_hits = 0
        dual_right_hits = 0
        dual_left_points = []
        dual_right_points = []
        dual_center_x = None
        dual_heading = 0.0
        branch_bias_direction = 0.0
        branch_bias_amount = 0.0
        recovery_active = False
        recovery_spin = False
        far_left_strength = 0
        far_right_strength = 0
        pending_mode_switch = None
        pending_tracked_x = None

        # 直道->弯道过渡记忆：无强证据时逐帧衰减，避免过度激进
        self.turn_indicator = float(np.clip(self.turn_indicator * TURN_INDICATOR_DECAY, 0.0, 1.0))
        turn_indicator_frame = float(self.turn_indicator)
        if weak_turn_indicator_floor > 0.0:
            turn_indicator_frame = max(turn_indicator_frame, weak_turn_indicator_floor)
            self.turn_indicator = turn_indicator_frame

        heading_limit_current, target_step_limit, history_weight_current, dynamic_heading_gain = \
            self._dynamic_control_params(turn_indicator_frame)

        if len(points) >= 2:
            coef, curve_score, fit_rmse, fit_inlier_ratio, fit_ok, near_far_span, slope_hint = self.robust_curve_fit(points)

            # 入弯意图 = 绝对曲率 + 近远横向跨度 + 曲率上升速度
            curve_norm = min(1.0, curve_score / max(1.0, CURVE_SHARP_THRESHOLD))
            span_norm = min(1.0, near_far_span / TURN_SPAN_SCALE)
            slope_norm = min(1.0, abs(slope_hint) / 1.35)
            entry_rate = min(1.0, max(0.0, curve_score - self.prev_curve_score) / TURN_ENTRY_RATE_SCALE)
            turn_raw = max(curve_norm, span_norm * 0.90, slope_norm * 0.75)

            turn_indicator_frame = max(turn_raw, self.turn_indicator + (entry_rate * 0.75))
            turn_indicator_frame = float(np.clip(turn_indicator_frame, 0.0, 1.0))
            self.turn_indicator = turn_indicator_frame
            self.prev_curve_score = curve_score

            heading_limit_current, target_step_limit, history_weight_current, dynamic_heading_gain = \
                self._dynamic_control_params(turn_indicator_frame)

            # 拟合一致性更新：本帧原始拟合 vs 上一帧系数在参考 Y 处的偏差。
            if coef is not None and fit_ok and self.last_fit_coef is not None:
                ref_y = float(LOOKAHEAD_Y_TURN)
                new_x = float(np.polyval(coef, ref_y))
                hist_x = float(np.polyval(self.last_fit_coef, ref_y))
                divergence = abs(new_x - hist_x)
                raw_coherence = 1.0 - float(np.clip(
                    (divergence - COHERENCE_DIVERGE_LOW) / max(1.0, COHERENCE_DIVERGE_HIGH - COHERENCE_DIVERGE_LOW),
                    0.0, 1.0))
                self.fit_coherence = float(
                    (self.fit_coherence * COHERENCE_SMOOTH) + (raw_coherence * (1.0 - COHERENCE_SMOOTH)))

            if coef is not None and fit_ok and self.last_fit_coef is not None:
                scaled_fit_hw = COHERENCE_FIT_HIST_MIN + (
                    (FIT_HISTORY_WEIGHT - COHERENCE_FIT_HIST_MIN) * self.fit_coherence)
                coef = (self.last_fit_coef * scaled_fit_hw) + (coef * (1.0 - scaled_fit_hw))

            if coef is not None and (not fit_ok) and self.last_fit_coef is not None:
                coef = self.last_fit_coef.copy()
                self.fit_fail_counter += 1
                fit_source = "history"
            elif coef is not None:
                self.fit_fail_counter = 0
                fit_source = "new" if fit_ok else "weak"
            else:
                fit_source = "point"

            if coef is not None:
                self.last_fit_coef = coef
                fit_coef_to_draw = coef

                if curve_score > CURVE_SHARP_THRESHOLD:
                    lookahead_y = LOOKAHEAD_Y_SHARP
                elif curve_score > CURVE_TURN_THRESHOLD:
                    lookahead_y = LOOKAHEAD_Y_TURN
                else:
                    lookahead_y = LOOKAHEAD_Y_STRAIGHT

                line_x = float(np.polyval(coef, lookahead_y))
                heading = float((2.0 * coef[0] * lookahead_y) + coef[1])
                heading = float(np.clip(heading, -heading_limit_current, heading_limit_current))
                if fit_source == "weak":
                    heading *= (0.55 + (0.25 * turn_indicator_frame))
        elif len(points) == 1:
            line_x = float(points[0][0])
            fit_source = "point"
            self.prev_curve_score *= 0.70
            curve_score = max(curve_score, self.prev_curve_score)
            if self.last_fit_coef is not None and self.mode != "SEARCH":
                history_turn = max(turn_indicator_frame, prev_turn_indicator)
                if uturn_active or history_turn > 0.72 or self.prev_curve_score > CURVE_SHARP_THRESHOLD:
                    lookahead_y = LOOKAHEAD_Y_UTURN
                else:
                    lookahead_y = LOOKAHEAD_Y_TURN

                history_x = float(np.polyval(self.last_fit_coef, lookahead_y))
                history_heading = float((2.0 * self.last_fit_coef[0] * lookahead_y) + self.last_fit_coef[1])
                history_heading = float(np.clip(history_heading, -heading_limit_current, heading_limit_current))

                y_near = float(SLICE_Y_LEVELS[0])
                y_far = float(SLICE_Y_LEVELS[-1])
                hist_near = float(np.polyval(self.last_fit_coef, y_near))
                hist_far = float(np.polyval(self.last_fit_coef, y_far))
                denom = max(1.0, abs(y_near - y_far))
                slope_hint = float((hist_near - hist_far) / denom)
                curve_score = max(curve_score, abs(hist_near - hist_far) * 0.9)

                hist_blend = SINGLE_POINT_HISTORY_TURN_BLEND if history_turn > SINGLE_POINT_HISTORY_TURN_IND else SINGLE_POINT_HISTORY_BLEND
                hist_blend *= self.fit_coherence
                line_x = (line_x * (1.0 - hist_blend)) + (history_x * hist_blend)
                heading = history_heading
                fit_coef_to_draw = self.last_fit_coef
                fit_source = "point_hist"
        elif self.last_fit_coef is not None and self.mode != "SEARCH":
            lookahead_y = LOOKAHEAD_Y_TURN
            line_x = float(np.polyval(self.last_fit_coef, lookahead_y))
            heading = float((2.0 * self.last_fit_coef[0] * lookahead_y) + self.last_fit_coef[1])
            heading = float(np.clip(heading, -heading_limit_current, heading_limit_current))
            fit_coef_to_draw = self.last_fit_coef
            fit_source = "history"
            self.prev_curve_score *= 0.70
        else:
            self.prev_curve_score *= 0.70

        # U-turn 保护：当曲率/转向意图持续很高时，短时切入更激进的转弯策略。
        heading_mag = abs(heading)
        uturn_context = (
            (self.mode in ("LOCKED_LEFT", "LOCKED_RIGHT"))
            and (not junction_active)
            and (
                (turn_indicator_frame > UTURN_TURN_IND_THRESHOLD and curve_score > UTURN_CURVE_THRESHOLD)
                or (curve_score > (CURVE_SHARP_THRESHOLD * 0.9) and heading_mag > UTURN_HEADING_THRESHOLD)
            )
        )
        if uturn_context:
            self.uturn_cooldown = UTURN_HOLD_FRAMES
        else:
            self.uturn_cooldown = max(0, self.uturn_cooldown - 1)
        uturn_active = self.uturn_cooldown > 0

        if uturn_active:
            lookahead_y = min(lookahead_y, LOOKAHEAD_Y_UTURN)
            ref_coef_for_uturn = fit_coef_to_draw if fit_coef_to_draw is not None else self.last_fit_coef
            if ref_coef_for_uturn is not None:
                line_x = float(np.polyval(ref_coef_for_uturn, lookahead_y))
                heading = float((2.0 * ref_coef_for_uturn[0] * lookahead_y) + ref_coef_for_uturn[1])

            heading_limit_current = max(heading_limit_current, UTURN_HEADING_LIMIT)
            heading = float(np.clip(heading, -heading_limit_current, heading_limit_current))
            dynamic_heading_gain *= UTURN_HEADING_GAIN_SCALE
            target_step_limit = max(target_step_limit, UTURN_TARGET_STEP_LIMIT)
            history_weight_current = min(history_weight_current, UTURN_HISTORY_MAX)

        # 远场证据统计：
        # 急弯分岔时，如果当前锁定边线已经不连贯，哪一半仍保留更多前方白色像素，
        # 往往就更接近真实应走的方向。
        far_left_strength, far_right_strength = self._far_field_balance(binary_warped, center_screen)

        # 4.1 优先尝试双边对称搜索：
        #     只要左右两根边线都能稳定命中，就直接用“双边中心”作为主目标。
        if not junction_active:
            (
                dual_left_x,
                dual_right_x,
                dual_measured_width,
                dual_left_hits,
                dual_right_hits,
                dual_left_points,
                dual_right_points,
            ) = self.find_dual_lines_symmetric(
                binary_warped=binary_warped,
                y_levels=SLICE_Y_LEVELS,
                center_screen=center_screen,
                lookahead_y=lookahead_y,
            )

            dual_ok = (
                dual_left_x is not None
                and dual_right_x is not None
                and min(dual_left_hits, dual_right_hits) >= DUAL_TRACK_MIN_HITS
                and (dual_right_x - dual_left_x) > DUAL_TRACK_MIN_GAP
            )

            if dual_ok:
                dual_center_x = 0.5 * (dual_left_x + dual_right_x)
                dual_center_points = [
                    (0.5 * (lp[0] + rp[0]), lp[1])
                    for lp, rp in zip(dual_left_points, dual_right_points)
                ]
                if len(dual_center_points) >= 2:
                    dual_ys = np.array([p[1] for p in dual_center_points], dtype=np.float32)
                    dual_xs = np.array([p[0] for p in dual_center_points], dtype=np.float32)
                    dual_coef = self._fit_poly(dual_ys, dual_xs, weights=None)
                    dual_center_x = float(np.polyval(dual_coef, lookahead_y))
                    dual_heading = float((2.0 * dual_coef[0] * lookahead_y) + dual_coef[1])
                    dual_heading = float(np.clip(dual_heading, -heading_limit_current, heading_limit_current))

                if dual_measured_width is not None:
                    dual_measured_width = float(np.clip(dual_measured_width, LANE_WIDTH_MIN, LANE_WIDTH_MAX))
                    self.estimated_lane_width = (
                        (self.estimated_lane_width * LANE_WIDTH_SMOOTH)
                        + (dual_measured_width * (1.0 - LANE_WIDTH_SMOOTH))
                    )

                if self.mode == "LOCKED_LEFT":
                    line_x = float(dual_left_x)
                    self.tracked_x = line_x
                    opp_line_x = float(dual_right_x)
                    opp_hits = int(dual_right_hits)
                    opp_points = list(dual_right_points)
                elif self.mode == "LOCKED_RIGHT":
                    line_x = float(dual_right_x)
                    self.tracked_x = line_x
                    opp_line_x = float(dual_left_x)
                    opp_hits = int(dual_left_hits)
                    opp_points = list(dual_left_points)
                elif self.preferred_mode == "LOCKED_LEFT":
                    line_x = float(dual_left_x)
                    opp_line_x = float(dual_right_x)
                elif self.preferred_mode == "LOCKED_RIGHT":
                    line_x = float(dual_right_x)
                    opp_line_x = float(dual_left_x)

        # 4.2 检测对侧边线并更新动态车道宽（防骑线核心）
        if junction_active:
            # 路口阶段优先单边历史连续性，避免把隔壁车道当成本车道。
            opp_line_x = None
            opp_hits = 0
            opp_points = []
        elif (not dual_ok) and self.mode in ("LOCKED_LEFT", "LOCKED_RIGHT"):
            ref_coef = fit_coef_to_draw if fit_coef_to_draw is not None else self.last_fit_coef
            opp_line_x, measured_width, opp_hits, opp_points = self.find_opposite_line(
                binary_warped=binary_warped,
                mode=self.mode,
                y_levels=SLICE_Y_LEVELS,
                ref_points=points,
                ref_coef=ref_coef,
                lookahead_y=lookahead_y,
            )

            if measured_width is not None:
                measured_width = float(np.clip(measured_width, LANE_WIDTH_MIN, LANE_WIDTH_MAX))
                self.estimated_lane_width = (
                    (self.estimated_lane_width * LANE_WIDTH_SMOOTH)
                    + (measured_width * (1.0 - LANE_WIDTH_SMOOTH))
                )
            else:
                # 双边观测不可靠时，缓慢回归默认车道宽，避免瞬时抖动
                self.estimated_lane_width = (self.estimated_lane_width * 0.98) + (LANE_WIDTH_PIXELS * 0.02)

        dynamic_lane_width = float(np.clip(self.estimated_lane_width, LANE_WIDTH_MIN, LANE_WIDTH_MAX))

        # 只有“命中数够 + 几何关系合理”才允许对侧线参与控制，避免单点误检把目标拉飞。
        opp_valid = False
        if dual_ok:
            opp_valid = True
        elif opp_line_x is not None and opp_hits >= OPP_MIN_HITS:
            if self.mode == "LOCKED_LEFT" and opp_line_x > (line_x + (OPP_MIN_GAP * 0.6)):
                opp_valid = True
            elif self.mode == "LOCKED_RIGHT" and opp_line_x < (line_x - (OPP_MIN_GAP * 0.6)):
                opp_valid = True
        if not opp_valid:
            opp_line_x = None

        # 4.3 中心线优先：
        #     双边稳定时直接取中线；
        #     单边稳定时再将边线平移为中心线；
        #     历史中心线只作为最后兜底。
        edge_line_x = float(line_x)
        edge_fit_source = fit_source
        shift_mode = self.mode if self.mode in ("LOCKED_LEFT", "LOCKED_RIGHT") else self.preferred_mode
        center_points = []
        center_source = "CENTER_NONE"
        single_line_bias = 0.0

        if dual_ok:
            center_points = [
                (0.5 * (lp[0] + rp[0]), float(lp[1]))
                for lp, rp in zip(dual_left_points, dual_right_points)
            ]
            center_source = "CENTER_DUAL"
        elif shift_mode == "LOCKED_LEFT" and points:
            opp_map = {int(round(py)): float(px) for px, py in opp_points}
            if not opp_valid:
                single_line_bias = min(
                    SINGLE_LINE_OPPOSITE_BIAS_MAX,
                    SINGLE_LINE_OPPOSITE_BIAS_BASE + (turn_indicator_frame * SINGLE_LINE_OPPOSITE_BIAS_TURN)
                )
            center_shift = (dynamic_lane_width * 0.5) + BIAS_OFFSET + single_line_bias
            for px, py in points:
                center_x = float(px + center_shift)
                opp_x = opp_map.get(int(round(py)))
                if opp_valid and opp_x is not None and opp_x > px:
                    midpoint_x = 0.5 * (float(px) + opp_x)
                    center_x = (center_x * (1.0 - DUAL_CENTER_BLEND)) + (midpoint_x * DUAL_CENTER_BLEND)
                center_points.append((center_x, float(py)))
            center_source = "CENTER_LEFT"
        elif shift_mode == "LOCKED_RIGHT" and points:
            opp_map = {int(round(py)): float(px) for px, py in opp_points}
            if not opp_valid:
                single_line_bias = min(
                    SINGLE_LINE_OPPOSITE_BIAS_MAX,
                    SINGLE_LINE_OPPOSITE_BIAS_BASE + (turn_indicator_frame * SINGLE_LINE_OPPOSITE_BIAS_TURN)
                )
            center_shift = (dynamic_lane_width * 0.5) + BIAS_OFFSET + single_line_bias
            for px, py in points:
                center_x = float(px - center_shift)
                opp_x = opp_map.get(int(round(py)))
                if opp_valid and opp_x is not None and opp_x < px:
                    midpoint_x = 0.5 * (float(px) + opp_x)
                    center_x = (center_x * (1.0 - DUAL_CENTER_BLEND)) + (midpoint_x * DUAL_CENTER_BLEND)
                center_points.append((center_x, float(py)))
            center_source = "CENTER_RIGHT"
        elif self.last_center_coef is not None:
            center_source = "CENTER_HISTORY"

        center_track = self._resolve_track_from_points(
            points=center_points,
            history_coef=self.last_center_coef,
            lookahead_y=lookahead_y,
            heading_limit=heading_limit_current,
            prev_turn_indicator=prev_turn_indicator,
        )
        if center_track['coef'] is not None:
            self.last_center_coef = center_track['coef']

        center_track_available = center_track['x'] is not None
        if center_track_available:
            line_x = float(center_track['x'])
            heading = float(center_track['heading'])
            curve_score = float(center_track['curve_score'])
            slope_hint = float(center_track['slope_hint'])
            fit_source = center_track['fit_source']
            fit_rmse = float(center_track['rmse'])
            fit_inlier_ratio = float(center_track['inlier_ratio'])
            fit_coef_to_draw = center_track['coef']
            control_points_count = len(center_points)
        else:
            control_points_count = len(points)

        # 急弯分岔重导向：
        # 当当前锁定边线在急弯里只剩很少点，但远场另一侧仍然有更强的连续证据时，
        # 先给目标点一个“方向偏置”，并为下一帧提前切换锁定边。
        weak_tracking_now = (control_points_count <= BRANCH_RELOCK_MAX_POINTS) or (fit_source in ("history", "weak", "point", "point_hist", "none"))
        strong_curve_wrong_side = (
            (not junction_active)
            and (not dual_ok)
            and opp_line_x is None
            and turn_indicator_frame > BRANCH_FORCE_RELOCK_TURN_IND
            and curve_score > BRANCH_FORCE_RELOCK_CURVE
        )
        if (not junction_active) and (not dual_ok) and turn_indicator_frame > BRANCH_RELOCK_TURN_IND and weak_tracking_now:
            if self.mode == "LOCKED_RIGHT":
                if far_left_strength > max(BRANCH_RELOCK_MIN_PIXELS, far_right_strength * BRANCH_RELOCK_STRENGTH_RATIO):
                    branch_bias_direction = 1.0
                    branch_bias_amount = BRANCH_BIAS_PIXELS
                    pending_mode_switch = "LOCKED_LEFT"
                    relock_x = self._dominant_half_blob_x(binary_warped, center_screen, prefer_left=True)
                    if relock_x is not None:
                        pending_tracked_x = relock_x
            elif self.mode == "LOCKED_LEFT":
                if far_right_strength > max(BRANCH_RELOCK_MIN_PIXELS, far_left_strength * BRANCH_RELOCK_STRENGTH_RATIO):
                    branch_bias_direction = -1.0
                    branch_bias_amount = BRANCH_BIAS_PIXELS
                    pending_mode_switch = "LOCKED_RIGHT"
                    relock_x = self._dominant_half_blob_x(binary_warped, center_screen, prefer_left=False)
                    if relock_x is not None:
                        pending_tracked_x = relock_x
        elif strong_curve_wrong_side:
            turning_left_hint = (heading > BRANCH_FORCE_RELOCK_HEADING) or (slope_hint > BRANCH_FORCE_RELOCK_SLOPE)
            turning_right_hint = (heading < -BRANCH_FORCE_RELOCK_HEADING) or (slope_hint < -BRANCH_FORCE_RELOCK_SLOPE)
            if self.mode == "LOCKED_RIGHT":
                left_dominant = far_left_strength > max(
                    BRANCH_RELOCK_MIN_PIXELS,
                    far_right_strength * BRANCH_FORCE_RELOCK_RATIO
                )
                memory_support = self.preferred_mode in (None, "LOCKED_LEFT")
                if left_dominant and turning_left_hint and memory_support:
                    branch_bias_direction = 1.0
                    branch_bias_amount = BRANCH_BIAS_PIXELS
                    pending_mode_switch = "LOCKED_LEFT"
                    relock_x = self._dominant_half_blob_x(binary_warped, center_screen, prefer_left=True)
                    if relock_x is not None:
                        pending_tracked_x = relock_x
            elif self.mode == "LOCKED_LEFT":
                right_dominant = far_right_strength > max(
                    BRANCH_RELOCK_MIN_PIXELS,
                    far_left_strength * BRANCH_FORCE_RELOCK_RATIO
                )
                memory_support = self.preferred_mode in (None, "LOCKED_RIGHT")
                if right_dominant and turning_right_hint and memory_support:
                    branch_bias_direction = -1.0
                    branch_bias_amount = BRANCH_BIAS_PIXELS
                    pending_mode_switch = "LOCKED_RIGHT"
                    relock_x = self._dominant_half_blob_x(binary_warped, center_screen, prefer_left=False)
                    if relock_x is not None:
                        pending_tracked_x = relock_x

        # 极端盲拐角恢复：
        # 当前方几乎看不到可持续跟踪的点时，不再继续“小角度向前冲”，
        # 而是保持上一次推断的转弯方向，强制做找线恢复。
        recovery_trigger = (
            (not junction_active)
            and (not dual_ok)
            and turn_indicator_frame > RECOVERY_TURN_IND
            and control_points_count <= RECOVERY_MAX_POINTS
            and max(far_left_strength, far_right_strength) < RECOVERY_MIN_FAR_PIXELS
            and fit_source in ("history", "weak", "point", "point_hist", "none")
        )
        if recovery_trigger:
            self.recovery_cooldown = RECOVERY_HOLD_FRAMES
            self.recovery_direction = self._turn_direction_from_geometry(heading, slope_hint)
        else:
            self.recovery_cooldown = max(0, self.recovery_cooldown - 1)

        recovery_active = self.recovery_cooldown > 0

        # 5. 控制目标跟中心线，而不是直接跟参考边线。
        target_x = float(self.last_target)
        lower = 8.0
        upper = float(w - 8.0)

        if dual_ok and dual_left_x is not None and dual_right_x is not None:
            lower = float(dual_left_x + MIN_EDGE_CLEARANCE)
            upper = float(dual_right_x - MIN_EDGE_CLEARANCE)
            if upper <= lower:
                lower = float(dual_center_x - 2.0)
                upper = float(dual_center_x + 2.0)

            if center_track_available:
                target_x = float(line_x + (heading * dynamic_heading_gain * DUAL_HEADING_SCALE))
            else:
                target_x = float(dual_center_x)

        elif center_track_available:
            heading_corr = heading * dynamic_heading_gain
            # 丢线 / 弱观测时，heading*gain 可能大幅超过中心偏移量，
            # 把目标推到屏幕中心的错误一侧（例如左转弯中反而右转）。
            # 此时将修正量限制在"中心偏移幅度的 50%"以内，
            # 确保修正方向不会逆转 deviation 的符号。
            if control_points_count <= 1 or fit_source in ("history", "point_hist"):
                center_offset = line_x - float(center_screen)
                if abs(center_offset) > 4.0:
                    corr_cap = abs(center_offset) * 0.50
                    heading_corr = float(np.clip(heading_corr, -corr_cap, corr_cap))
            target_x = float(line_x + heading_corr)

            if shift_mode == "LOCKED_LEFT":
                lower = float(edge_line_x + MIN_EDGE_CLEARANCE)
                upper = float((opp_line_x - MIN_EDGE_CLEARANCE) if (opp_line_x is not None and opp_line_x > edge_line_x) else (w - 8.0))
            elif shift_mode == "LOCKED_RIGHT":
                lower = float((opp_line_x + MIN_EDGE_CLEARANCE) if (opp_line_x is not None and opp_line_x < edge_line_x) else 8.0)
                upper = float(edge_line_x - MIN_EDGE_CLEARANCE)
            else:
                half_width = max(10.0, (dynamic_lane_width * 0.5) - MIN_EDGE_CLEARANCE)
                lower = float(line_x - half_width)
                upper = float(line_x + half_width)

            if upper <= lower:
                lower = float(line_x - 2.0)
                upper = float(line_x + 2.0)
        else:
            target_x = (self.last_target * 0.96) + (center_screen * 0.04)
            center_source = "CENTER_GHOST"

        if branch_bias_direction != 0.0:
            target_x += (branch_bias_direction * branch_bias_amount)
            target_step_limit = max(target_step_limit, BRANCH_STEP_LIMIT)
            history_weight_current = min(history_weight_current, BRANCH_HISTORY_MAX)

        if recovery_active:
            target_x = center_screen + (self.recovery_direction * RECOVERY_TARGET_BIAS)
            target_step_limit = max(target_step_limit, RECOVERY_STEP_LIMIT)
            history_weight_current = min(history_weight_current, RECOVERY_HISTORY_MAX)

        target_x = float(np.clip(target_x, lower, upper))
        target_x = float(np.clip(target_x, 8.0, w - 8.0))

        if dual_ok and dual_left_x is not None and dual_right_x is not None:
            lane_left = float(dual_left_x)
            lane_right = float(dual_right_x)
        elif shift_mode == "LOCKED_LEFT":
            lane_left = float(edge_line_x)
            lane_right = float(opp_line_x if opp_line_x is not None else (edge_line_x + dynamic_lane_width))
        elif shift_mode == "LOCKED_RIGHT":
            lane_left = float(opp_line_x if opp_line_x is not None else (edge_line_x - dynamic_lane_width))
            lane_right = float(edge_line_x)
        elif center_track_available:
            lane_left = float(line_x - (dynamic_lane_width * 0.5))
            lane_right = float(line_x + (dynamic_lane_width * 0.5))
        else:
            lane_left = float(center_screen - (dynamic_lane_width * 0.5))
            lane_right = float(center_screen + (dynamic_lane_width * 0.5))

        clearance_left = max(0.0, target_x - lane_left)
        clearance_right = max(0.0, lane_right - target_x)
        min_clearance = min(clearance_left, clearance_right)

        # 6. 时域平滑 + 步长限幅：抑制控制突变，避免机械冲击
        #    coherence 低（几何突变）→ 降低平滑权重 → 更快跟随新目标
        effective_hist_w = COHERENCE_TARGET_HIST_MIN + (
            (history_weight_current - COHERENCE_TARGET_HIST_MIN) * self.fit_coherence)
        effective_hist_w = max(COHERENCE_TARGET_HIST_MIN, effective_hist_w)
        target_step = float(np.clip(target_x - self.last_target, -target_step_limit, target_step_limit))
        bounded_target = self.last_target + target_step
        final_target = (self.last_target * effective_hist_w) + (bounded_target * (1.0 - effective_hist_w))
        self.last_target = final_target

        deviation = final_target - center_screen

        # 7. 输出转弯强度（给客户端降速）：融合偏差、曲率、丢线、骑线风险
        dev_score = min(1.0, abs(deviation) / 110.0)
        curve_score_norm = min(1.0, curve_score / 55.0)
        lost_score = min(1.0, float(self.lost_counter) / float(max(1, GHOST_MODE_LIMIT)))
        hug_score = max(0.0, (MIN_EDGE_CLEARANCE + 6.0 - min_clearance) / (MIN_EDGE_CLEARANCE + 6.0))
        heading_score = min(1.0, abs(heading) / max(1.0, UTURN_HEADING_LIMIT))

        # 跟踪不确定性：点数太少或只能靠历史时，主动提高severity，防止转弯中突然加速。
        point_quality = min(1.0, float(control_points_count) / float(max(1, len(SLICE_Y_LEVELS))))
        uncertainty = 1.0 - point_quality
        if fit_source == "history":
            uncertainty = max(uncertainty, 0.62)
        elif fit_source in ("point", "point_hist"):
            uncertainty = max(uncertainty, 0.78 if fit_source == "point" else 0.68)
        elif fit_source == "weak":
            uncertainty = max(uncertainty, 0.45)
        if center_source == "CENTER_DUAL":
            uncertainty *= 0.82

        turn_severity = max(dev_score, curve_score_norm * 0.7, lost_score * 0.7, turn_indicator_frame * 0.85, hug_score * 0.6, heading_score * 0.55, uncertainty * 0.65)
        if uturn_active:
            turn_severity = min(1.0, turn_severity + UTURN_SEVERITY_BOOST)

        # 恢复模式的最高级兜底：
        # 若前方几乎没有可靠点，就给客户端发送一个超大偏差作为“原地强制找线”信号。
        if recovery_active and control_points_count <= RECOVERY_MAX_POINTS and max(far_left_strength, far_right_strength) < RECOVERY_MIN_FAR_PIXELS:
            deviation = self.recovery_direction * RECOVERY_SENTINEL_DEVIATION
            turn_severity = 1.0
            recovery_spin = True

        self.vis_data = {
            'target': int(final_target),
            'tracked_x': int(self.tracked_x) if self.mode != "SEARCH" else -1,
            'opp_x': int(opp_line_x) if opp_line_x is not None else -1,
            'mode': center_source,
            'lock_mode': self.mode,
            'curve': round(curve_score, 1),
            'points': len(points),
            'ctrl_points': int(control_points_count),
            'fit_source': fit_source,
            'edge_fit_source': edge_fit_source,
            'fit_rmse': round(fit_rmse, 1),
            'fit_inlier': round(fit_inlier_ratio, 2),
            'coherence': round(self.fit_coherence, 2),
            'turn_ind': round(turn_indicator_frame, 2),
            'h_lim': round(heading_limit_current, 2),
            'step_lim': round(target_step_limit, 1),
            'hist_w': round(history_weight_current, 2),
            'lane_w': round(dynamic_lane_width, 1),
            'single_bias': round(single_line_bias, 1),
            'dual_hits': int(min(dual_left_hits, dual_right_hits)) if dual_ok else int(opp_hits),
            'dual_ok': 1 if dual_ok else 0,
            'junction': 1 if junction_active else 0,
            'uturn': 1 if uturn_active else 0,
            'uturn_cd': int(self.uturn_cooldown),
            'recover': 1 if recovery_active else 0,
            'spin': 1 if recovery_spin else 0,
            'recover_dir': int(self.recovery_direction),
            'pref_mode': self.preferred_mode if self.preferred_mode is not None else "NONE",
            'pend_mode': pending_mode_switch if pending_mode_switch is not None else "NONE",
            'far_l': int(far_left_strength),
            'far_r': int(far_right_strength),
            'clear_l': round(clearance_left, 1),
            'clear_r': round(clearance_right, 1),
            'fit_coef': fit_coef_to_draw.tolist() if fit_coef_to_draw is not None else [],
            'fit_points': [(int(p[0]), int(p[1])) for p in center_points],
            'raw_points': [(int(p[0]), int(p[1])) for p in points],
            'opp_points': [(int(p[0]), int(p[1])) for p in opp_points],
            'dual_left_points': [(int(p[0]), int(p[1])) for p in dual_left_points],
            'dual_right_points': [(int(p[0]), int(p[1])) for p in dual_right_points],
        }

        if pending_mode_switch is not None:
            self.mode = pending_mode_switch
            self.preferred_mode = pending_mode_switch
        if pending_tracked_x is not None:
            self.tracked_x = float(pending_tracked_x)

        return deviation, float(turn_severity)




def recv_exact(conn, size):
    """从 TCP 连接中精确读取固定字节数；连接中断则返回 None。"""
    buf = b''
    while len(buf) < size:
        data = conn.recv(size - len(buf))
        if not data:
            return None
        buf += data
    return buf


def load_unet_model(device, weights_path=r'params/min_loss.pth'):
    """
    加载 UNet 模型权重。

    参数：
    - device: torch.device，CPU 或 CUDA。
    - weights_path: 训练得到的权重文件路径。

    返回：
    - net: eval() 状态的 UNet 模型。
    """
    print("Loading Deep Learning Model...")
    net = UNet().to(device)
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        net.load_state_dict(state)
        print("Weights loaded.")
    else:
        raise FileNotFoundError("UNet weights not found: {}".format(weights_path))
    net.eval()
    return net


def run_unet_inference(frame_bgr, net, device):
    """
    对单帧 BGR 图像做 UNet 推理并返回二值掩码（0/255）。

    注意：
    - 这里输入被 resize 到 256x256，与训练时预处理保持一致。
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out = net(transform(pil_img.resize((256, 256))).unsqueeze(0).to(device))
        out = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255
    return out


def postprocess_lane_mask(out):
    """
    对分割输出做轻量后处理，提升边线连续性并抑制无效区域噪声。
    """
    out = cv2.medianBlur(out, 5)
    _, out = cv2.threshold(out, 80, 255, cv2.THRESH_BINARY)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, POST_CLOSE_KERNEL, iterations=1)

    # 掩码裁剪：去除底部和顶部高噪声区域。
    out[MASK_BOTTOM_ROW:, :] = 0
    out[0:TOP_MASK_ROW, :] = 0
    return out

def build_debug_visualization(out, tracker, deviation, t_start):
    """
    绘制调试窗口（拟合曲线、目标点、状态文本等）。

    该函数仅用于可视化，不参与控制计算。
    """
    vis_img = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    vis_data = tracker.vis_data
    img_w = vis_img.shape[1]

    center_x = img_w // 2
    cv2.line(vis_img, (center_x, 0), (center_x, 256), (100, 100, 100), 1)

    if vis_data:
        for px, py in vis_data.get('raw_points', []):
            cv2.circle(vis_img, (px, py), 2, (90, 255, 90), -1)

        for px, py in vis_data.get('fit_points', []):
            cv2.circle(vis_img, (px, py), 2, (0, 200, 255), -1)

        for px, py in vis_data.get('opp_points', []):
            cv2.circle(vis_img, (px, py), 2, (255, 100, 255), -1)

        for px, py in vis_data.get('dual_left_points', []):
            cv2.circle(vis_img, (px, py), 2, (0, 220, 220), -1)

        for px, py in vis_data.get('dual_right_points', []):
            cv2.circle(vis_img, (px, py), 2, (220, 220, 0), -1)

        fit_coef = vis_data.get('fit_coef', [])
        if len(fit_coef) == 3:
            coef = np.array(fit_coef, dtype=np.float32)
            curve_points = []
            for yy in range(ZONE_START, ZONE_END):
                xx = int(np.clip(np.polyval(coef, yy), 0, img_w - 1))
                curve_points.append((xx, yy))
            if len(curve_points) >= 2:
                cv2.polylines(vis_img, [np.array(curve_points, dtype=np.int32)], False, (255, 0, 0), 2)

        cv2.circle(vis_img, (vis_data['target'], 180), 8, (0, 0, 255), -1)

        if vis_data['tracked_x'] > 0:
            cv2.circle(vis_img, (vis_data['tracked_x'], 180), 6, (0, 255, 0), -1)

        if vis_data.get('opp_x', -1) > 0:
            cv2.circle(vis_img, (vis_data['opp_x'], 180), 6, (255, 0, 255), -1)

        fps = 1.0 / max(1e-4, (time.time() - t_start))
        cv2.putText(vis_img, f"Dev: {deviation:.1f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(vis_img, f"Mode: {vis_data['mode']}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 1)
        cv2.putText(vis_img, f"Lock: {vis_data.get('lock_mode', 'SEARCH')}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 255, 255), 1)
        cv2.putText(vis_img, f"Curve: {vis_data['curve']}  Raw:{vis_data['points']} Ctrl:{vis_data.get('ctrl_points', 0)}", (5, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
        cv2.putText(vis_img,
                    f"Fit:{vis_data['fit_source']} Edge:{vis_data.get('edge_fit_source', 'none')} RMSE:{vis_data['fit_rmse']} In:{vis_data['fit_inlier']} Coh:{vis_data.get('coherence', 1.0)}",
                    (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        cv2.putText(vis_img,
                    f"Turn:{vis_data['turn_ind']} Hlim:{vis_data['h_lim']} Step:{vis_data['step_lim']}",
                    (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 160), 1)
        cv2.putText(vis_img,
                    f"LaneW:{vis_data['lane_w']} SBias:{vis_data.get('single_bias', 0)} Dual:{vis_data['dual_ok']} Hits:{vis_data['dual_hits']} J:{vis_data['junction']} U:{vis_data.get('uturn', 0)}/{vis_data.get('uturn_cd', 0)}",
                    (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 255), 1)
        cv2.putText(vis_img,
                    f"ClearL:{vis_data.get('clear_l', 0)} ClearR:{vis_data.get('clear_r', 0)} Hw:{vis_data['hist_w']}",
                    (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 235, 255), 1)
        cv2.putText(vis_img,
                    f"Rec:{vis_data.get('recover', 0)} Spin:{vis_data.get('spin', 0)} Far:{vis_data.get('far_l', 0)}/{vis_data.get('far_r', 0)} Pref:{vis_data.get('pref_mode', 'NONE')} Pend:{vis_data.get('pend_mode', 'NONE')}",
                    (5, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 150), 1)
        cv2.putText(vis_img, f"FPS: {fps:.1f}", (5, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (160, 255, 160), 1)

    return vis_img


def run_client_session(conn, net, device, tracker):
    """
    处理一个客户端连接会话。

    返回：
    - should_stop=True 表示用户按 q 请求退出整个 server。
    - should_stop=False 表示仅当前连接结束，继续等待下一个客户端。
    """
    while True:
        t_start = time.time()
        size_data = recv_exact(conn, 4)
        if not size_data:
            return False

        size = struct.unpack(">I", size_data)[0]
        img_data = recv_exact(conn, size)
        if not img_data:
            return False

        frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        out = run_unet_inference(frame, net, device)
        out = postprocess_lane_mask(out)
        cv2.imshow("Laptop: Lane Detection Output", out)

        deviation, severity = tracker.process_lane_lock_on(out)
        payload = struct.pack(">ff", float(deviation), float(severity))
        conn.sendall(payload)

        vis_img = build_debug_visualization(out, tracker, deviation, t_start)
        cv2.imshow("Laptop: Lane Fit", vis_img)
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) == ord('q'):
            return True

def run_server(host="0.0.0.0", port=8000, weights_path=r'params/min_loss.pth'):
    """
    server 主入口：
    - 初始化模型和跟踪器
    - 监听客户端连接
    - 循环处理每个连接会话
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_unet_model(device, weights_path=weights_path)

    tracker = LaneTracker()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许服务重启后快速复用端口，避免 TIME_WAIT 导致的 bind 失败。
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Server ready on port {port}...")

    should_stop = False
    try:
        while not should_stop:
            conn, addr = server.accept()
            print(f"Connected: {addr}")
            try:
                should_stop = run_client_session(conn, net, device, tracker)
            except Exception as e:
                print(e)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        server.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_server()
